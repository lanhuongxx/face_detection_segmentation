"""Đánh giá pipeline thật của Hương (RetinaFace + U-Net) trên tập val/test của Hương.

Mỗi ảnh đo trong 2 không gian, mỗi không gian một CSV cùng schema với ``evaluate.py``:

- ``orig_clean``: ảnh gốc 1024, GT = mask mặt đã làm sạch (đúng target lúc train).
  Đây là phép đo theo evaluation_plan 3.1b (không gian ảnh gốc).
- ``crop256_clean``: cắt khung vuông 256 quanh mặt như notebook của Hương; so được với
  Dice 0.9788 / IoU 0.9590 của Hương.

Lưu ý: GT là mask ĐÃ LÀM SẠCH của Hương (trừ tóc, bỏ tua mảnh), không phải nhãn thô.

    python -m src.evaluation.evaluate_pipeline --data_root data/raw/CelebAMask-HQ \
        --split test --weights models/unet_best.pt --out_dir outputs/pipe_test --save_masks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .huong_protocol import MARGIN, SIZE, clean_face_mask, crop_mask, huong_split, ids_with_skin
from .seg_metrics import dice_score, iou_score, pixel_precision_recall

FIELDS = ("image_id", "dice", "iou", "precision", "recall", "area_gt", "area_pred",
          "detected", "n_faces", "fallback")
SPACES = ("orig_clean", "crop256_clean")


def _row(image_id: str, pred: np.ndarray, gt: np.ndarray, extra: dict) -> dict:
    precision, recall = pixel_precision_recall(pred, gt)
    return {"image_id": image_id, "dice": float(dice_score(pred, gt)),
            "iou": float(iou_score(pred, gt)), "precision": float(precision),
            "recall": float(recall), "area_gt": int(gt.sum()), "area_pred": int(pred.sum()),
            **extra}


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def evaluate_ids(ids: Sequence[int], image_dir: Path, mask_dir: Path, engine,
                 save_dir: Path | None = None) -> dict[str, list[dict]]:
    """``engine`` là ``PipelineEngine``; dùng RetinaFace thật để lấy bbox."""
    rows: dict[str, list[dict]] = {space: [] for space in SPACES}
    for image_id in ids:
        path = next((p for ext in (".jpg", ".png", ".jpeg")
                     if (p := image_dir / f"{image_id}{ext}").is_file()), None)
        if path is None:
            raise FileNotFoundError(f"không thấy ảnh {image_id} trong {image_dir}")
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        clean = clean_face_mask(mask_dir, image_id, (image.shape[1], image.shape[0]))
        gt_box = _bbox(clean)
        out = engine.mask_from_detector(image)
        extra = {"detected": int(out.n_faces > 0), "n_faces": out.n_faces,
                 "fallback": int(out.fallback)}
        name = f"{image_id:05d}"
        if save_dir is not None:  # để error_analysis.py dùng lại: pred/, gt_clean/
            for sub, arr in (("pred", out.mask), ("gt_clean", clean)):
                (save_dir / sub).mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr.astype(np.uint8) * 255).save(save_dir / sub / f"{name}.png")
        rows["orig_clean"].append(_row(name, out.mask, clean, extra))
        box = out.box or gt_box  # không phát hiện được: cắt theo GT để vẫn có số trong crop
        if box is None:
            crop_pred = crop_gt = np.zeros((SIZE, SIZE), bool)
        else:
            crop_pred, crop_gt = crop_mask(out.mask, box), crop_mask(clean, box)
        rows["crop256_clean"].append(_row(name, crop_pred, crop_gt, extra))
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    keys = ("dice", "iou", "precision", "recall")
    out = {f"macro_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    out.update(n=len(rows), detect_rate=float(np.mean([r["detected"] for r in rows])),
               fallback_rate=float(np.mean([r["fallback"] for r in rows])),
               n_faces_mean=float(np.mean([r["n_faces"] for r in rows])))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, required=True,
                        help="thư mục có CelebA-HQ-img và CelebAMask-HQ-mask-anno")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_masks", action="store_true",
                        help="ghi pred/ và gt_clean/ vào out_dir cho error_analysis.py")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args(argv)

    image_dir = args.data_root / "CelebA-HQ-img"
    mask_dir = args.data_root / "CelebAMask-HQ-mask-anno"
    ids = huong_split(ids_with_skin(mask_dir))[args.split]
    if args.limit:
        ids = ids[:args.limit]
    from .pipeline_adapter import PipelineEngine
    engine = PipelineEngine(args.config, weights=args.weights, device=args.device)
    rows = evaluate_ids(ids, image_dir, mask_dir, engine,
                        args.out_dir if args.save_masks else None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for space, items in rows.items():
        with open(args.out_dir / f"{space}.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(items)
        summary[space] = summarize(items)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"split": args.split, "margin": MARGIN,
                    **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

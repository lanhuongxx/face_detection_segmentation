"""Đánh giá pipeline thật của Hương (RetinaFace + U-Net) trên tập val/test của Hương.

Mỗi ảnh được đo trong hai không gian, mỗi không gian ghi một CSV cùng dạng với ``evaluate.py``:

- ``orig_clean``: ảnh gốc 1024, GT là mask mặt đã làm sạch (đúng loại target lúc train).
  Đây là phép đo theo evaluation_plan mục 3.1b (đo trong không gian ảnh gốc).
- ``crop256_clean``: cắt khung vuông 256 quanh mặt như notebook của Hương, nên so được
  với Dice 0.9788 và IoU 0.9590 mà Hương báo.

Lưu ý: GT ở đây là mask đã làm sạch của Hương (trừ tóc, bỏ các phần mảnh), không phải nhãn thô.

    python -m src.evaluation.evaluate_pipeline --data_root data/raw/CelebAMask-HQ \\
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
    """Tạo một dòng kết quả cho một ảnh: các metric và diện tích mask, cộng thêm ``extra``."""
    precision, recall = pixel_precision_recall(pred, gt)
    return {"image_id": image_id, "dice": float(dice_score(pred, gt)),
            "iou": float(iou_score(pred, gt)), "precision": float(precision),
            "recall": float(recall), "area_gt": int(gt.sum()), "area_pred": int(pred.sum()),
            **extra}


def _bbox(mask: np.ndarray):
    """Bbox [x1, y1, x2, y2] bao quanh vùng True của mask, mask rỗng thì trả None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def evaluate_ids(ids: Sequence[int], image_dir: Path, mask_dir: Path, engine,
                 save_dir: Path | None = None) -> dict[str, list[dict]]:
    """Chạy pipeline trên từng id và trả kết quả theo hai không gian đo.

    ``engine`` là ``PipelineEngine``, bbox lấy từ RetinaFace thật.
    """
    rows: dict[str, list[dict]] = {space: [] for space in SPACES}
    for image_id in ids:
        # ảnh tên theo id không có số 0 đứng đầu, thử lần lượt các đuôi file thường gặp
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
        if save_dir is not None:  # lưu mask ra file để error_analysis.py đọc lại (thư mục pred/ và gt_clean/)
            for sub, arr in (("pred", out.mask), ("gt_clean", clean)):
                (save_dir / sub).mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr.astype(np.uint8) * 255).save(save_dir / sub / f"{name}.png")
        # không gian 1: đo trên ảnh gốc
        rows["orig_clean"].append(_row(name, out.mask, clean, extra))
        # không gian 2: đo trong khung cắt 256. Nếu detector không thấy mặt thì tạm cắt theo bbox
        # của GT, để ảnh đó vẫn có số đo trong khung cắt (việc miss đã được ghi ở cột detected)
        box = out.box or gt_box
        if box is None:
            crop_pred = crop_gt = np.zeros((SIZE, SIZE), bool)
        else:
            crop_pred, crop_gt = crop_mask(out.mask, box), crop_mask(clean, box)
        rows["crop256_clean"].append(_row(name, crop_pred, crop_gt, extra))
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    """Tóm tắt một không gian đo: trung bình các metric, tỉ lệ phát hiện được và tỉ lệ dùng elip dự phòng."""
    keys = ("dice", "iou", "precision", "recall")
    out = {f"macro_{k}": float(np.mean([r[k] for r in rows])) for k in keys}
    out.update(n=len(rows), detect_rate=float(np.mean([r["detected"] for r in rows])),
               fallback_rate=float(np.mean([r["fallback"] for r in rows])),
               n_faces_mean=float(np.mean([r["n_faces"] for r in rows])))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy từ dòng lệnh: đánh giá tập val hoặc test, ghi hai CSV và một file summary.json."""
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
    # import muộn để chỉ khi thật sự chạy mới cần có code pipeline của Hương
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
    # ghi kèm split và margin để sau này nhìn file json biết số này đo trên tập nào, margin bao nhiêu
    (args.out_dir / "summary.json").write_text(
        json.dumps({"split": args.split, "margin": MARGIN,
                    **summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

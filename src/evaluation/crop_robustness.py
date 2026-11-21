"""So sánh GT crop và predicted crop: lỗi detection ảnh hưởng mask bao nhiêu (YE-3).

Với mỗi ảnh test của Hương (CelebAMask-HQ, một mặt) chạy ``FacePipeline`` thật với:

- ``gt_crop``: bbox lấy từ mask GT (cận trên, detection hoàn hảo).
- ``pred_crop``: bbox RetinaFace thật (không phát hiện được = mask rỗng, tính là miss).
- ``jitter_*``: bbox GT bị lệch/đổi cỡ có chủ đích, mô phỏng detection sai.

Dice/IoU tính trong không gian ảnh gốc so với GT đã làm sạch của Hương.

    python -m src.evaluation.crop_robustness --data_root data/raw/CelebAMask-HQ \
        --split test --weights models/unet_best.pt --out_dir outputs/crop
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from PIL import Image

from .huong_protocol import clean_face_mask, huong_split, ids_with_skin
from .pipeline_adapter import MaskOutput
from .seg_metrics import dice_score, iou_score

Box = tuple[float, float, float, float]

# Độ lệch mô phỏng: (tên, dx, dy, scale) với dx/dy tính theo tỉ lệ cạnh bbox.
JITTERS = (
    ("jitter_shift_x10", 0.10, 0.0, 1.0),
    ("jitter_shift_y10", 0.0, 0.10, 1.0),
    ("jitter_shift_20", 0.20, 0.20, 1.0),
    ("jitter_scale_0.8", 0.0, 0.0, 0.8),
    ("jitter_scale_1.2", 0.0, 0.0, 1.2),
)


def mask_bbox(mask: np.ndarray) -> Box | None:
    """Bbox [x1, y1, x2, y2] (x2, y2 exclusive) của vùng True; None nếu mask rỗng."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def jitter_box(box: Box, dx: float, dy: float, scale: float) -> Box:
    """Dịch tâm (theo tỉ lệ cạnh) và đổi cỡ quanh tâm."""
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2 + dx * w, (box[1] + box[3]) / 2 + dy * h
    return cx - w * scale / 2, cy - h * scale / 2, cx + w * scale / 2, cy + h * scale / 2


def evaluate_image(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    engine: Any,
    jitters: Sequence[tuple[str, float, float, float]] = JITTERS,
) -> list[dict[str, Any]]:
    """Chạy mọi cách cắt cho một ảnh RGB; mỗi cách là một dòng kết quả.

    ``engine`` là ``PipelineEngine`` (có ``mask_from_boxes`` và ``mask_from_detector``).
    """
    gt_box = mask_bbox(gt_mask)
    if gt_box is None:
        raise ValueError("mask GT rỗng, không tạo được GT crop")
    outputs: list[tuple[str, MaskOutput, bool]] = [
        ("gt_crop", engine.mask_from_boxes(image_rgb, [gt_box]), True)
    ]
    out = engine.mask_from_detector(image_rgb)
    outputs.append(("pred_crop", out, out.n_faces > 0))
    for name, dx, dy, scale in jitters:
        box = jitter_box(gt_box, dx, dy, scale)
        outputs.append((name, engine.mask_from_boxes(image_rgb, [box]), True))
    rows = []
    for name, out, detected in outputs:
        rows.append(
            {
                "setting": name,
                "detected": int(detected),
                "n_faces": out.n_faces,
                "fallback": int(out.fallback),
                "box_iou": box_iou(out.box, gt_box) if out.box is not None else 0.0,
                "dice": float(dice_score(out.mask, gt_mask)),
                "iou": float(iou_score(out.mask, gt_mask)),
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tổng hợp theo cách cắt, kèm mức giảm Dice so với GT crop."""
    names = list(dict.fromkeys(r["setting"] for r in rows))
    base = [r["dice"] for r in rows if r["setting"] == "gt_crop"]
    base_mean = float(np.mean(base)) if base else float("nan")
    out = []
    for name in names:
        subset = [r for r in rows if r["setting"] == name]
        dice = np.array([r["dice"] for r in subset])
        out.append(
            {
                "setting": name,
                "n": len(subset),
                "miss_rate": 1.0 - float(np.mean([r["detected"] for r in subset])),
                "fallback_rate": float(np.mean([r["fallback"] for r in subset])),
                "n_faces_mean": float(np.mean([r["n_faces"] for r in subset])),
                "box_iou_mean": float(np.mean([r["box_iou"] for r in subset])),
                "dice_mean": float(dice.mean()),
                "dice_median": float(np.median(dice)),
                "iou_mean": float(np.mean([r["iou"] for r in subset])),
                "dice_drop_vs_gt": base_mean - float(dice.mean()),
            }
        )
    return out


def _write(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, required=True,
                        help="thư mục có CelebA-HQ-img và CelebAMask-HQ-mask-anno")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/crop"))
    args = parser.parse_args(argv)

    from .pipeline_adapter import PipelineEngine

    image_dir = args.data_root / "CelebA-HQ-img"
    mask_dir = args.data_root / "CelebAMask-HQ-mask-anno"
    ids = huong_split(ids_with_skin(mask_dir))[args.split]
    if args.limit:
        ids = ids[: args.limit]
    engine = PipelineEngine(args.config, args.weights, args.device)
    all_rows = []
    for image_id in ids:
        path = next((p for ext in (".jpg", ".png") if (p := image_dir / f"{image_id}{ext}").is_file()), None)
        if path is None:
            parser.error(f"không thấy ảnh {image_id} trong {image_dir}")
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        gt = clean_face_mask(mask_dir, image_id, (image.shape[1], image.shape[0]))
        for row in evaluate_image(image, gt, engine):
            all_rows.append({"image_id": f"{image_id:05d}", **row})

    summary = summarize(all_rows)
    _write(args.out_dir / "crop_per_image.csv", all_rows)
    _write(args.out_dir / "crop_summary.csv", summary)
    print(f"{'setting':<18}{'n':>6}{'miss':>8}{'fallbk':>8}{'boxIoU':>9}{'Dice':>9}{'drop':>9}")
    for r in summary:
        print(f"{r['setting']:<18}{r['n']:>6}{r['miss_rate']:>8.3f}{r['fallback_rate']:>8.3f}"
              f"{r['box_iou_mean']:>9.3f}{r['dice_mean']:>9.3f}{r['dice_drop_vs_gt']:>9.3f}")
    print(f"Kết quả: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

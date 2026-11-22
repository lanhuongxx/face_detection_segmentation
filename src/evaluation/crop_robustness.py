"""So sánh cắt bằng bbox GT với cắt bằng bbox dự đoán: lỗi detection làm mask tệ đi bao nhiêu (YE-3).

Với mỗi ảnh test của Hương (CelebAMask-HQ, mỗi ảnh một mặt), chạy ``FacePipeline`` thật theo
nhiều cách cắt khác nhau:

- ``gt_crop``: bbox lấy từ mask GT, tức detection hoàn hảo nên đây là mức tốt nhất có thể đạt.
- ``pred_crop``: bbox do RetinaFace thật dự đoán (không phát hiện được thì mask rỗng, tính là miss).
- ``jitter_*``: bbox GT bị cố ý dịch hoặc phóng to/thu nhỏ, để mô phỏng detection sai.

Dice và IoU tính trong không gian ảnh gốc, so với mask GT đã làm sạch theo cách của Hương.

    python -m src.evaluation.crop_robustness --data_root data/raw/CelebAMask-HQ \\
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

# Các kiểu làm lệch bbox: (tên, dx, dy, scale). dx và dy tính theo tỉ lệ chiều rộng/cao của bbox,
# ví dụ 0.10 nghĩa là dịch đi 10% cạnh. scale là hệ số phóng to hoặc thu nhỏ quanh tâm.
JITTERS = (
    ("jitter_shift_x10", 0.10, 0.0, 1.0),
    ("jitter_shift_y10", 0.0, 0.10, 1.0),
    ("jitter_shift_20", 0.20, 0.20, 1.0),
    ("jitter_scale_0.8", 0.0, 0.0, 0.8),
    ("jitter_scale_1.2", 0.0, 0.0, 1.2),
)


def mask_bbox(mask: np.ndarray) -> Box | None:
    """Tìm bbox [x1, y1, x2, y2] bao quanh vùng True (x2, y2 là biên ngoài). Mask rỗng thì trả None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(a: Box, b: Box) -> float:
    """IoU giữa hai bbox: diện tích phần giao chia cho diện tích phần hợp."""
    # chiều rộng và cao của phần giao, nếu hai box không chồng nhau thì bằng 0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def jitter_box(box: Box, dx: float, dy: float, scale: float) -> Box:
    """Dịch tâm bbox đi (dx * rộng, dy * cao) rồi đổi kích thước theo scale quanh tâm mới."""
    w, h = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2 + dx * w, (box[1] + box[3]) / 2 + dy * h
    return cx - w * scale / 2, cy - h * scale / 2, cx + w * scale / 2, cy + h * scale / 2


def evaluate_image(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    engine: Any,
    jitters: Sequence[tuple[str, float, float, float]] = JITTERS,
) -> list[dict[str, Any]]:
    """Thử mọi cách cắt trên một ảnh RGB, mỗi cách cho ra một dòng kết quả.

    ``engine`` là ``PipelineEngine`` (có hai hàm ``mask_from_boxes`` và ``mask_from_detector``).
    """
    gt_box = mask_bbox(gt_mask)
    if gt_box is None:
        raise ValueError("mask GT rỗng, không tạo được GT crop")
    # gt_crop luôn tính là phát hiện được (True) vì bbox lấy thẳng từ GT
    outputs: list[tuple[str, MaskOutput, bool]] = [
        ("gt_crop", engine.mask_from_boxes(image_rgb, [gt_box]), True)
    ]
    out = engine.mask_from_detector(image_rgb)
    # pred_crop chỉ tính là phát hiện được khi detector thật tìm ra ít nhất một mặt
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
                # không có bbox (không thấy mặt) thì cho box_iou = 0
                "box_iou": box_iou(out.box, gt_box) if out.box is not None else 0.0,
                "dice": float(dice_score(out.mask, gt_mask)),
                "iou": float(iou_score(out.mask, gt_mask)),
            }
        )
    return rows


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tổng hợp kết quả theo từng cách cắt, kèm mức Dice giảm so với gt_crop."""
    # dict.fromkeys để lấy tên cách cắt không trùng mà vẫn giữ thứ tự xuất hiện
    names = list(dict.fromkeys(r["setting"] for r in rows))
    # Dice trung bình của gt_crop làm mốc để so
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
                # miss_rate = tỉ lệ ảnh không phát hiện được mặt
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
    """Ghi danh sách dòng ra CSV, lấy tên cột từ khóa của dòng đầu tiên."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy từ dòng lệnh: duyệt các ảnh của tập được chọn, ghi CSV theo ảnh và CSV tổng hợp."""
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

    # import ở đây để các hàm bên trên dùng được mà không cần code pipeline của Hương
    from .pipeline_adapter import PipelineEngine

    image_dir = args.data_root / "CelebA-HQ-img"
    mask_dir = args.data_root / "CelebAMask-HQ-mask-anno"
    # chia tập đúng như Hương để val/test ở đây trùng với val/test lúc train
    ids = huong_split(ids_with_skin(mask_dir))[args.split]
    if args.limit:
        ids = ids[: args.limit]
    engine = PipelineEngine(args.config, args.weights, args.device)
    all_rows = []
    for image_id in ids:
        # ảnh CelebA-HQ tên là 0.jpg, 1.jpg... (không có số 0 đứng đầu), thử .jpg rồi tới .png
        path = next((p for ext in (".jpg", ".png") if (p := image_dir / f"{image_id}{ext}").is_file()), None)
        if path is None:
            parser.error(f"không thấy ảnh {image_id} trong {image_dir}")
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        # mask GT làm sạch ở đúng cỡ ảnh gốc, (rộng, cao) = (shape[1], shape[0])
        gt = clean_face_mask(mask_dir, image_id, (image.shape[1], image.shape[0]))
        for row in evaluate_image(image, gt, engine):
            all_rows.append({"image_id": f"{image_id:05d}", **row})

    summary = summarize(all_rows)
    _write(args.out_dir / "crop_per_image.csv", all_rows)
    _write(args.out_dir / "crop_summary.csv", summary)
    # in bảng tóm tắt ra màn hình cho tiện xem nhanh
    print(f"{'setting':<18}{'n':>6}{'miss':>8}{'fallbk':>8}{'boxIoU':>9}{'Dice':>9}{'drop':>9}")
    for r in summary:
        print(f"{r['setting']:<18}{r['n']:>6}{r['miss_rate']:>8.3f}{r['fallback_rate']:>8.3f}"
              f"{r['box_iou_mean']:>9.3f}{r['dice_mean']:>9.3f}{r['dice_drop_vs_gt']:>9.3f}")
    print(f"Kết quả: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

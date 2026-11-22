"""Phân tích lỗi segmentation: xem ví dụ TP/FP/FN và điều kiện ảnh nào làm model tệ đi.

Đầu vào là CSV theo ảnh do ``evaluate.py`` tạo, cùng thư mục ảnh gốc, mask GT và mask dự đoán.
Các chỉ số điều kiện ảnh được tính thẳng từ ảnh nên không cần gán nhãn thêm:

- ``brightness``: độ sáng trung bình (0-255), thấp nghĩa là ảnh tối.
- ``sharpness``: phương sai Laplacian, thấp nghĩa là ảnh mờ (chỉ so sánh được khi cùng kích thước ảnh).
- ``contrast``: độ lệch chuẩn của mức xám.
- ``face_ratio``: diện tích mask GT chia diện tích ảnh, thấp nghĩa là mặt nhỏ trong khung.

Mỗi điều kiện được chia thành 3 nhóm low/mid/high theo tertile (chia ba theo phân vị) của chính
tập đang đánh giá. Ngưỡng chia được ghi vào CSV, không tự đặt một ngưỡng cố định.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import _load_mask
from .seg_metrics import _binarize

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CONDITIONS = ("brightness", "sharpness", "contrast", "face_ratio")
# màu phủ lên ảnh: TP xanh lá, FP đỏ, FN xanh dương
_COLORS = {"tp": (0, 200, 0), "fp": (230, 30, 30), "fn": (40, 90, 255)}
# cạnh (pixel) của mỗi ô trong ảnh ví dụ
_PANEL = 256


def find_image(image_dir: Path, image_id: str) -> Path:
    """Tìm ảnh gốc theo image_id (id không có đuôi file, ảnh có thể nằm trong thư mục con)."""
    # CelebA-HQ đặt tên ảnh là "0.jpg" còn mask và CSV dùng "00000",
    # nên thử thêm cả tên đã bỏ các số 0 đứng đầu
    names = [image_id]
    if image_id.isdigit() and str(int(image_id)) != image_id:
        names.append(str(int(image_id)))
    for name in names:
        for suffix in _IMAGE_SUFFIXES:
            candidate = image_dir / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Không thấy ảnh cho {image_id!r} trong {image_dir}")


def _find_mask(directory: Path, image_id: str) -> Path:
    """Tìm file mask của một id trong thư mục, thử lần lượt các đuôi file thường gặp."""
    for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"):
        candidate = directory / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Không thấy mask cho {image_id!r} trong {directory}")


def laplacian_variance(gray: np.ndarray) -> float:
    """Phương sai của Laplacian (4 láng giềng), dùng làm độ nét. Tự tính bằng NumPy cho khỏi cần OpenCV."""
    g = gray.astype(np.float64)
    # ảnh nhỏ hơn 3x3 thì không có pixel nào đủ láng giềng để tính
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    # Laplacian = tổng 4 pixel xung quanh trừ 4 lần pixel giữa, chỉ tính cho phần lõi ảnh
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    )
    return float(lap.var())


def image_conditions(image_rgb: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Tính 4 chỉ số điều kiện (sáng, nét, tương phản, tỉ lệ mặt) cho một ảnh và mask GT bool."""
    # đổi RGB sang xám theo công thức chuẩn 0.299 R + 0.587 G + 0.114 B
    gray = (
        image_rgb[..., 0] * 0.299 + image_rgb[..., 1] * 0.587 + image_rgb[..., 2] * 0.114
    )
    return {
        "brightness": float(gray.mean()),
        "sharpness": laplacian_variance(gray),
        "contrast": float(gray.std()),
        "face_ratio": float(gt_mask.sum()) / gt_mask.size,
    }


def error_overlay(
    image_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Phủ màu lên ảnh: TP xanh lá, FP đỏ (model tô thừa), FN xanh dương (model bỏ sót)."""
    out = image_rgb.astype(np.float32).copy()
    for key, region in (("tp", gt & pred), ("fp", pred & ~gt), ("fn", gt & ~pred)):
        color = np.asarray(_COLORS[key], dtype=np.float32)
        # trộn màu với ảnh gốc theo alpha để vẫn nhìn thấy nội dung ảnh bên dưới
        out[region] = (1 - alpha) * out[region] + alpha * color
    return out.clip(0, 255).astype(np.uint8)


def _panel(array: np.ndarray, label: str) -> Image.Image:
    """Đổi mảng thành một ô ảnh vuông _PANEL x _PANEL, ghi chữ nhãn ở góc trên bên trái."""
    image = Image.fromarray(array).convert("RGB").resize((_PANEL, _PANEL))
    ImageDraw.Draw(image).text((4, 4), label, fill=(255, 255, 0))
    return image


def make_example_sheet(
    rows: Sequence[dict], image_dir: Path, gt_dir: Path, pred_dir: Path, path: Path
) -> None:
    """Ghép các ảnh ví dụ thành một tấm. Mỗi hàng gồm: ảnh | GT | dự đoán | lỗi (TP xanh lá, FP đỏ, FN xanh dương)."""
    if not rows:
        return
    sheet = Image.new("RGB", (_PANEL * 4, _PANEL * len(rows)), "white")
    for r, row in enumerate(rows):
        image_id = row["image_id"]
        image = np.asarray(Image.open(find_image(image_dir, image_id)).convert("RGB"))
        gt = _binarize(_load_mask(_find_mask(gt_dir, image_id)))
        pred = _binarize(_load_mask(_find_mask(pred_dir, image_id)))
        # mask bool (True/False) nhân 255 và nhân ra 3 kênh để hiển thị như ảnh xám
        panels = (
            _panel(image, f"{image_id}"),
            _panel(np.repeat((gt * 255).astype(np.uint8)[..., None], 3, 2), "GT"),
            _panel(np.repeat((pred * 255).astype(np.uint8)[..., None], 3, 2),
                   f"pred dice={float(row['dice']):.3f}"),
            _panel(error_overlay(image, gt, pred), "TP green / FP red / FN blue"),
        )
        for c, panel in enumerate(panels):
            sheet.paste(panel, (c * _PANEL, r * _PANEL))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def tertile_groups(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Chia thành 3 nhóm 0/1/2 (low/mid/high) theo tertile, trả về nhãn từng phần tử và hai ngưỡng."""
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    labels = np.where(values <= low, 0, np.where(values <= high, 1, 2))
    return labels, (float(low), float(high))


def analyze(
    metrics_csv: Path,
    image_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    k: int = 6,
) -> dict:
    """Chạy toàn bộ phân tích, ghi CSV và ảnh PNG vào out_dir rồi trả về một dict tóm tắt."""
    with metrics_csv.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV không có dòng nào: {metrics_csv}")
    # csv.DictReader đọc mọi thứ ra chuỗi nên đổi các cột số về float
    for row in rows:
        for key in ("dice", "iou", "precision", "recall", "area_gt", "area_pred"):
            row[key] = float(row[key])

    # bước 1: tính điều kiện ảnh (sáng, nét, tương phản, tỉ lệ mặt) cho từng ảnh
    for row in rows:
        image = np.asarray(Image.open(find_image(image_dir, row["image_id"])).convert("RGB"))
        gt = _binarize(_load_mask(_find_mask(gt_dir, row["image_id"])))
        row.update(image_conditions(image, gt))

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "dice", "iou", "precision", "recall", "area_gt", "area_pred",
              *CONDITIONS]
    with (out_dir / "per_image_conditions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k_: row[k_] for k_ in fields} for row in rows)

    # bước 2: với mỗi điều kiện, chia ảnh thành 3 nhóm rồi xem Dice, IoU... của từng nhóm
    dice = np.array([r["dice"] for r in rows])
    iou = np.array([r["iou"] for r in rows])
    prec = np.array([r["precision"] for r in rows])
    rec = np.array([r["recall"] for r in rows])
    group_names = ("low", "mid", "high")
    table = []
    for condition in CONDITIONS:
        values = np.array([r[condition] for r in rows])
        labels, (lo, hi) = tertile_groups(values)
        for index, name in enumerate(group_names):
            selected = labels == index
            # nhóm nào trống (ví dụ nhiều giá trị trùng nhau làm tertile dồn lại) thì bỏ qua
            if not selected.any():
                continue
            table.append(
                {
                    "condition": condition, "group": name,
                    "cutoff_low": f"{lo:.4g}", "cutoff_high": f"{hi:.4g}",
                    "n": int(selected.sum()),
                    "dice_mean": f"{dice[selected].mean():.4f}",
                    "dice_median": f"{np.median(dice[selected]):.4f}",
                    "iou_mean": f"{iou[selected].mean():.4f}",
                    "precision_mean": f"{prec[selected].mean():.4f}",
                    "recall_mean": f"{rec[selected].mean():.4f}",
                }
            )
    with (out_dir / "error_by_condition.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    # bước 3: chọn k ảnh tệ nhất, k ảnh ở giữa (quanh trung vị) và k ảnh tốt nhất theo Dice
    k = max(1, min(k, len(rows)))
    order = sorted(rows, key=lambda r: r["dice"])
    mid = len(order) // 2
    # lấy k ảnh quanh vị trí giữa, kẹp lại để không chạy ra ngoài danh sách
    start = max(0, min(mid - k // 2, len(order) - k))
    picks = {
        "worst": order[:k],
        "median": order[start:start + k],
        "best": order[-k:][::-1],
    }
    for name, chosen in picks.items():
        make_example_sheet(chosen, image_dir, gt_dir, pred_dir, out_dir / f"examples_{name}.png")
    with (out_dir / "worst_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k_: r[k_] for k_ in fields} for r in picks["worst"])

    # bước 4: xem lỗi chủ yếu là tô thừa hay bỏ sót.
    # precision < recall nghĩa là FP nhiều hơn FN (tô thừa), ngược lại là bỏ sót
    over = int(np.sum(prec < rec))
    under = int(np.sum(rec < prec))
    return {
        "n_images": len(rows),
        "dice_mean": float(dice.mean()),
        "over_segmentation_images": over,
        "under_segmentation_images": under,
        "worst_ids": [r["image_id"] for r in picks["worst"]],
        "out_dir": str(out_dir),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy từ dòng lệnh: phân tích rồi in kết quả chính ra màn hình."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="CSV theo ảnh từ evaluate.py")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/eval/error_analysis"))
    parser.add_argument("--k", type=int, default=6, help="số ví dụ mỗi nhóm")
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.metrics, args.image_dir, args.gt_dir, args.pred_dir,
                          args.out_dir, args.k)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"{summary['n_images']} ảnh, Dice trung bình {summary['dice_mean']:.4f}")
    print(f"Tô thừa nhiều hơn bỏ sót: {summary['over_segmentation_images']} ảnh; "
          f"bỏ sót nhiều hơn tô thừa: {summary['under_segmentation_images']} ảnh")
    print(f"Ảnh tệ nhất: {', '.join(summary['worst_ids'])}")
    print(f"Kết quả: {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

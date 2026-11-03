"""Đánh giá các cặp mask segmentation và ghi metric theo từng ảnh ra CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .seg_metrics import confusion_counts, dice_score, iou_score, pixel_precision_recall

_SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".npy"}
# Thứ tự cột được cố định để file CSV luôn có cùng schema.
_CSV_FIELDS = (
    "image_id", "dice", "iou", "precision", "recall", "area_gt", "area_pred"
)


def _index_directory(directory: Path) -> dict[str, Path]:
    """Lập chỉ mục file mask theo đường dẫn tương đối đã bỏ phần mở rộng."""
    if not directory.is_dir():
        raise ValueError(f"mask directory does not exist: {directory}")
    indexed: dict[str, Path] = {}
    # rglob hỗ trợ cả cấu trúc thư mục con; sort để kết quả ổn định giữa các lần chạy.
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        # Giữ đường dẫn thư mục con trong key để tránh trùng tên ảnh ở hai thư mục.
        # Bỏ extension để prediction .png vẫn ghép được với ground truth .bmp/.tif.
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in indexed:
            raise ValueError(f"duplicate mask name {key!r} in {directory}")
        indexed[key] = path
    return indexed


def _directory_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[str, Path, Path]]:
    """Ghép prediction với ground truth bằng tên tương đối, không tự bỏ qua file thiếu."""
    pred_files = _index_directory(pred_dir)
    gt_files = _index_directory(gt_dir)
    pred_names, gt_names = set(pred_files), set(gt_files)
    # Kiểm tra chênh lệch hai tập tên riêng để nêu rõ phía nào đang thiếu mask.
    missing_gt = sorted(pred_names - gt_names)
    missing_pred = sorted(gt_names - pred_names)
    if missing_gt or missing_pred:
        details = []
        if missing_gt:
            details.append(f"missing ground truth for: {', '.join(missing_gt)}")
        if missing_pred:
            details.append(f"missing prediction for: {', '.join(missing_pred)}")
        raise ValueError("; ".join(details))
    if not pred_files:
        raise ValueError("no supported mask files found")
    # Sắp xếp theo image_id để CSV có thứ tự tái lập, không phụ thuộc thứ tự filesystem.
    return [
        (name, pred_files[name], gt_files[name]) for name in sorted(pred_names)
    ]


def _manifest_pairs(manifest: Path) -> list[tuple[str, Path, Path]]:
    """Đọc cặp mask từ CSV có các cột image_id, pred_path và gt_path."""
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    pairs = []
    image_ids: set[str] = set()
    # utf-8-sig cũng đọc được CSV UTF-8 có BOM thường được tạo bởi spreadsheet.
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "pred_path", "gt_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "manifest must have image_id, pred_path, and gt_path columns"
            )
        # CSV có một dòng header nên dòng dữ liệu đầu tiên là dòng số 2.
        for line_number, row in enumerate(reader, start=2):
            # strip bỏ khoảng trắng thừa quanh ID và đường dẫn do thao tác tạo CSV.
            image_id = (row["image_id"] or "").strip()
            pred_path = (row["pred_path"] or "").strip()
            gt_path = (row["gt_path"] or "").strip()
            if not image_id or not pred_path or not gt_path:
                raise ValueError(f"manifest has an empty value on line {line_number}")
            if image_id in image_ids:
                raise ValueError(f"duplicate image_id {image_id!r} in manifest")
            image_ids.add(image_id)
            # Đường dẫn tương đối trong manifest được tính từ thư mục chứa manifest.
            pairs.append(
                (
                    image_id,
                    _resolve_manifest_path(manifest.parent, pred_path),
                    _resolve_manifest_path(manifest.parent, gt_path),
                )
            )
    if not pairs:
        raise ValueError("manifest contains no mask pairs")
    return pairs


def _resolve_manifest_path(base: Path, value: str) -> Path:
    """Giữ nguyên đường dẫn tuyệt đối, còn đường dẫn tương đối thì nối với base."""
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_mask(path: Path) -> np.ndarray:
    """Đọc một mask 2D từ .npy hoặc ảnh; không chuyển mask thành nhị phân tại đây."""
    if not path.is_file():
        raise ValueError(f"mask file does not exist: {path}")
    if path.suffix.lower() == ".npy":
        # Tắt pickle để chỉ nạp dữ liệu mảng, không thực thi dữ liệu object tùy ý.
        mask = np.load(path, allow_pickle=False)
    else:
        # Chuyển ảnh sang grayscale để có một mặt phẳng mask đơn kênh.
        with Image.open(path) as image:
            mask = np.asarray(image.convert("L"))
    # Metric chỉ nhận một mask 2D; ảnh nhiều kênh thường là đầu vào không đúng định dạng.
    if mask.ndim != 2:
        raise ValueError(f"mask file must contain one 2D mask: {path} has shape {mask.shape}")
    return mask


def evaluate_pairs(
    pairs: Sequence[tuple[str, Path, Path]],
) -> tuple[list[dict[str, str | int | float]], tuple[int, int, int]]:
    """Tính metric cho từng cặp và đồng thời cộng TP/FP/FN để tính micro."""
    rows = []
    total_tp = total_fp = total_fn = 0
    for image_id, pred_path, gt_path in pairs:
        try:
            pred = _load_mask(pred_path)
            gt = _load_mask(gt_path)
            # confusion_counts mặc định trả mảng một phần tử cho một ảnh (shape H,W).
            tp, fp, fn, _ = confusion_counts(pred, gt)
            # Cộng confusion counts thô; không cộng Dice/IoU từng ảnh để tính micro.
            total_tp += int(tp[0])
            total_fp += int(fp[0])
            total_fn += int(fn[0])
            # Với một ảnh, per_image vẫn trả vector một phần tử nên lấy chỉ số 0.
            precision, recall = pixel_precision_recall(
                pred, gt, reduction="per_image"
            )
            rows.append(
                {
                    "image_id": image_id,
                    "dice": float(dice_score(pred, gt, reduction="per_image")[0]),
                    "iou": float(iou_score(pred, gt, reduction="per_image")[0]),
                    "precision": float(precision[0]),
                    "recall": float(recall[0]),
                    # Diện tích mask là số pixel dương: GT=TP+FN, prediction=TP+FP.
                    "area_gt": int(tp[0] + fn[0]),
                    "area_pred": int(tp[0] + fp[0]),
                }
            )
        except ValueError as error:
            raise ValueError(f"image {image_id!r}: {error}") from error
    # Trả về cả dữ liệu theo ảnh (macro) lẫn số đếm gộp (micro).
    return rows, (total_tp, total_fp, total_fn)


def _write_csv(output: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Tạo thư mục cha nếu cần rồi ghi header và kết quả theo từng ảnh."""
    # Cho phép --out trỏ tới một thư mục chưa được tạo trước đó.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    rows: list[dict[str, str | int | float]],
    micro_counts: tuple[int, int, int],
) -> None:
    """In thống kê macro trên ảnh và metric micro trên toàn bộ pixel."""
    print("Per-image summary (population std):")
    for field in _CSV_FIELDS[1:]:
        # Thống kê mỗi cột CSV theo ảnh; std mặc định của NumPy là population std.
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        print(
            f"{field}: mean={values.mean():.6f}, std={values.std():.6f}, "
            f"median={np.median(values):.6f}, min={values.min():.6f}, "
            f"max={values.max():.6f}"
        )

    # Micro gộp TP/FP/FN trên toàn bộ ảnh rồi mới thế vào công thức metric.
    tp, fp, fn = micro_counts
    dice_denominator = 2 * tp + fp + fn
    # Nếu cả tập không có pixel dương ở cả prediction và GT, quy ước Dice/IoU = 1.
    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator else 1.0
    # Nếu cả prediction và GT đều rỗng thì precision/recall bằng 1; còn lại mẫu số 0 bằng 0.
    both_empty = tp + fp + fn == 0
    precision = tp / (tp + fp) if tp + fp else (1.0 if both_empty else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if both_empty else 0.0)
    print(
        "Micro: "
        f"dice={dice:.6f}, iou={iou:.6f}, "
        f"precision={precision:.6f}, recall={recall:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Điểm vào CLI; chọn một trong hai nguồn: manifest hoặc hai thư mục mask."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    # Hai lựa chọn nguồn loại trừ nhau để tránh cách hiểu mơ hồ về danh sách cặp.
    source.add_argument("--manifest", type=Path, help="CSV with image_id,pred_path,gt_path")
    source.add_argument("--pred_dir", type=Path, help="directory containing predicted masks")
    parser.add_argument("--gt_dir", type=Path, help="directory containing ground-truth masks")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/seg_metrics.csv"),
        help="output CSV path (default: outputs/eval/seg_metrics.csv)",
    )
    # argv=None nghĩa là argparse đọc tham số trực tiếp từ dòng lệnh.
    args = parser.parse_args(argv)
    try:
        if args.manifest:
            if args.gt_dir:
                raise ValueError("--gt_dir cannot be combined with --manifest")
            pairs = _manifest_pairs(args.manifest)
        else:
            # Khi không dùng manifest, bắt buộc phải cung cấp đủ hai thư mục.
            if args.pred_dir is None or args.gt_dir is None:
                raise ValueError("provide both --pred_dir and --gt_dir, or --manifest")
            pairs = _directory_pairs(args.pred_dir, args.gt_dir)
        # Tính một lượt, lưu kết quả per-image và các count tổng để tránh đọc lại mask.
        rows, micro_counts = evaluate_pairs(pairs)
        _write_csv(args.out, rows)
        print(f"Wrote {len(rows)} image results to {args.out}")
        _print_summary(rows, micro_counts)
    except (OSError, ValueError) as error:
        # parser.error in ra thông báo CLI dễ đọc và kết thúc với exit code khác 0.
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    # Cho phép chạy bằng: python -m src.evaluation.evaluate ...
    raise SystemExit(main())

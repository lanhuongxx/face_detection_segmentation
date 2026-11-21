"""Đánh giá các cặp mask segmentation (dự đoán và ground truth), ghi metric từng ảnh ra CSV.

Có hai cách đưa dữ liệu vào: một manifest CSV liệt kê từng cặp mask, hoặc hai thư mục
(pred_dir và gt_dir) có mask trùng tên. Sau khi chạy xong sẽ có một file CSV theo ảnh
và một đoạn thống kê tóm tắt in ra màn hình.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .seg_metrics import confusion_counts, dice_score, iou_score, pixel_precision_recall

_SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".npy"}
# cố định thứ tự cột để file CSV lần nào cũng cùng một dạng
_CSV_FIELDS = (
    "image_id", "dice", "iou", "precision", "recall", "area_gt", "area_pred"
)


def _index_directory(directory: Path) -> dict[str, Path]:
    """Quét một thư mục mask, trả về dict {tên mask bỏ đuôi file: đường dẫn}."""
    if not directory.is_dir():
        raise ValueError(f"mask directory does not exist: {directory}")
    indexed: dict[str, Path] = {}
    # rglob để vào được cả thư mục con, sorted để thứ tự không đổi giữa các lần chạy
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        # key giữ cả đường dẫn thư mục con để hai ảnh trùng tên ở hai thư mục khác nhau
        # không bị nhầm. Bỏ đuôi file để mask dự đoán .png vẫn ghép được với GT .bmp/.tif
        key = path.relative_to(directory).with_suffix("").as_posix()
        if key in indexed:
            raise ValueError(f"duplicate mask name {key!r} in {directory}")
        indexed[key] = path
    return indexed


def _directory_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[str, Path, Path]]:
    """Ghép mask dự đoán với GT theo tên. Thiếu bên nào thì báo lỗi chứ không bỏ qua."""
    pred_files = _index_directory(pred_dir)
    gt_files = _index_directory(gt_dir)
    pred_names, gt_names = set(pred_files), set(gt_files)
    # so sánh hai chiều riêng để biết chính xác bên nào đang thiếu mask
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
    # sắp xếp theo tên để CSV ra luôn cùng thứ tự, không phụ thuộc hệ điều hành
    return [
        (name, pred_files[name], gt_files[name]) for name in sorted(pred_names)
    ]


def _manifest_pairs(manifest: Path) -> list[tuple[str, Path, Path]]:
    """Đọc danh sách cặp mask từ CSV có ba cột image_id, pred_path, gt_path."""
    if not manifest.is_file():
        raise ValueError(f"manifest does not exist: {manifest}")
    pairs = []
    image_ids: set[str] = set()
    # utf-8-sig để đọc được cả file CSV có BOM (Excel hay thêm vào khi lưu UTF-8)
    with manifest.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "pred_path", "gt_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "manifest must have image_id, pred_path, and gt_path columns"
            )
        # dòng 1 là header nên dòng dữ liệu đầu tiên là dòng 2 (dùng để báo lỗi cho đúng)
        for line_number, row in enumerate(reader, start=2):
            # strip để bỏ khoảng trắng dư hay bị dính lúc soạn CSV bằng tay
            image_id = (row["image_id"] or "").strip()
            pred_path = (row["pred_path"] or "").strip()
            gt_path = (row["gt_path"] or "").strip()
            if not image_id or not pred_path or not gt_path:
                raise ValueError(f"manifest has an empty value on line {line_number}")
            if image_id in image_ids:
                raise ValueError(f"duplicate image_id {image_id!r} in manifest")
            image_ids.add(image_id)
            # đường dẫn tương đối trong manifest được hiểu là tính từ thư mục chứa manifest
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
    """Đường dẫn tuyệt đối giữ nguyên, đường dẫn tương đối thì nối vào base."""
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_mask(path: Path) -> np.ndarray:
    """Đọc một mask 2D từ file .npy hoặc file ảnh. Hàm này chưa đổi mask sang nhị phân."""
    if not path.is_file():
        raise ValueError(f"mask file does not exist: {path}")
    if path.suffix.lower() == ".npy":
        # allow_pickle=False để chỉ đọc dữ liệu mảng, tránh chạy code lạ nằm trong file
        mask = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            if image.mode.startswith("I;16") or image.mode == "I":
                # PNG 16-bit (hay gặp khi lưu bản đồ xác suất 0..65535).
                # Nếu để convert("L") thì Pillow cắt giá trị ở 255, mọi pixel đều thành 255
                # và Dice bị sai mà không báo lỗi gì. Quy ước ở đây: 16-bit dùng đủ dải
                # 0..65535 nên chia cho 257 để về 0..255
                raw = np.asarray(image).astype(np.float64)
                mask = np.clip(np.rint(raw / 257.0), 0, 255).astype(np.uint8)
            elif image.mode == "F":
                mask = np.asarray(image, dtype=np.float32)  # ảnh float thì giữ nguyên xác suất 0..1
            else:
                # các trường hợp còn lại đổi sang xám để có mask một kênh
                mask = np.asarray(image.convert("L"))
    # metric chỉ nhận mask 2D, ảnh nhiều kênh thường là do đưa nhầm file
    if mask.ndim != 2:
        raise ValueError(f"mask file must contain one 2D mask: {path} has shape {mask.shape}")
    return mask


def evaluate_pairs(
    pairs: Sequence[tuple[str, Path, Path]],
) -> tuple[list[dict[str, str | int | float]], tuple[int, int, int]]:
    """Tính metric cho từng cặp mask, đồng thời cộng dồn TP/FP/FN để tính micro."""
    rows = []
    total_tp = total_fp = total_fn = 0
    for image_id, pred_path, gt_path in pairs:
        try:
            pred = _load_mask(pred_path)
            gt = _load_mask(gt_path)
            # một ảnh thì confusion_counts trả về mảng có đúng một phần tử
            tp, fp, fn, _ = confusion_counts(pred, gt)
            # micro cần cộng số đếm thô (TP, FP, FN), không cộng điểm Dice hay IoU từng ảnh
            total_tp += int(tp[0])
            total_fp += int(fp[0])
            total_fn += int(fn[0])
            # per_image cũng trả mảng một phần tử nên bên dưới lấy chỉ số 0
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
                    # diện tích mask = số pixel dương: GT = TP + FN, dự đoán = TP + FP
                    "area_gt": int(tp[0] + fn[0]),
                    "area_pred": int(tp[0] + fp[0]),
                }
            )
        except ValueError as error:
            # gắn thêm id ảnh vào lỗi để biết ngay ảnh nào gây ra
            raise ValueError(f"image {image_id!r}: {error}") from error
    # trả cả số liệu từng ảnh (để tính macro) lẫn tổng số đếm (để tính micro)
    return rows, (total_tp, total_fp, total_fn)


def _write_csv(output: Path, rows: list[dict[str, str | int | float]]) -> None:
    """Ghi kết quả từng ảnh ra CSV, tự tạo thư mục cha nếu chưa có."""
    # để --out có thể trỏ vào thư mục chưa tồn tại
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    rows: list[dict[str, str | int | float]],
    micro_counts: tuple[int, int, int],
) -> None:
    """In thống kê theo ảnh (macro) và metric gộp toàn bộ pixel (micro)."""
    print("Per-image summary (population std):")
    for field in _CSV_FIELDS[1:]:
        # thống kê từng cột metric trên các ảnh. np.std mặc định là độ lệch chuẩn tổng thể
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        print(
            f"{field}: mean={values.mean():.6f}, std={values.std():.6f}, "
            f"median={np.median(values):.6f}, min={values.min():.6f}, "
            f"max={values.max():.6f}"
        )

    # micro: cộng TP/FP/FN của tất cả ảnh trước, rồi mới thay vào công thức
    tp, fp, fn = micro_counts
    dice_denominator = 2 * tp + fp + fn
    # cả tập không có pixel dương nào (ở cả dự đoán lẫn GT) thì quy ước Dice và IoU bằng 1
    dice = 2 * tp / dice_denominator if dice_denominator else 1.0
    iou_denominator = tp + fp + fn
    iou = tp / iou_denominator if iou_denominator else 1.0
    # tương tự với precision và recall: cả hai mask đều rỗng thì bằng 1,
    # còn mẫu số 0 vì lý do khác thì bằng 0
    both_empty = tp + fp + fn == 0
    precision = tp / (tp + fp) if tp + fp else (1.0 if both_empty else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if both_empty else 0.0)
    print(
        "Micro: "
        f"dice={dice:.6f}, iou={iou:.6f}, "
        f"precision={precision:.6f}, recall={recall:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Chạy từ dòng lệnh. Chọn một trong hai nguồn dữ liệu: --manifest hoặc cặp thư mục."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    # hai cách nhập loại trừ nhau, để khỏi bị nhập lẫn rồi không rõ đang dùng danh sách nào
    source.add_argument("--manifest", type=Path, help="CSV with image_id,pred_path,gt_path")
    source.add_argument("--pred_dir", type=Path, help="directory containing predicted masks")
    parser.add_argument("--gt_dir", type=Path, help="directory containing ground-truth masks")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/seg_metrics.csv"),
        help="output CSV path (default: outputs/eval/seg_metrics.csv)",
    )
    # argv=None thì argparse tự đọc tham số từ dòng lệnh
    args = parser.parse_args(argv)
    try:
        if args.manifest:
            if args.gt_dir:
                raise ValueError("--gt_dir cannot be combined with --manifest")
            pairs = _manifest_pairs(args.manifest)
        else:
            # không dùng manifest thì phải đưa đủ cả hai thư mục
            if args.pred_dir is None or args.gt_dir is None:
                raise ValueError("provide both --pred_dir and --gt_dir, or --manifest")
            pairs = _directory_pairs(args.pred_dir, args.gt_dir)
        # chỉ đọc mask một lượt: vừa lấy số liệu từng ảnh vừa lấy tổng số đếm
        rows, micro_counts = evaluate_pairs(pairs)
        _write_csv(args.out, rows)
        print(f"Wrote {len(rows)} image results to {args.out}")
        _print_summary(rows, micro_counts)
    except (OSError, ValueError) as error:
        # parser.error in thông báo gọn gàng rồi thoát với mã lỗi khác 0
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    # để chạy được bằng: python -m src.evaluation.evaluate ...
    raise SystemExit(main())

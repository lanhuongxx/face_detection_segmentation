"""Test phần đọc mask, ghép cặp và ghi CSV của evaluate.py (cả CLI lẫn hàm bên trong)."""

import csv

import numpy as np
import pytest
from PIL import Image

from src.evaluation.evaluate import _directory_pairs, evaluate_pairs, main


def test_directory_cli_writes_per_image_metrics(tmp_path, capsys):
    """Chạy CLI với hai thư mục mask, kiểm tra CSV theo ảnh và phần thống kê in ra màn hình."""
    # dự đoán là .png còn GT là .tif nhưng cùng tên "face", vẫn phải ghép được với nhau
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    Image.fromarray(np.array([[255, 255], [0, 0]], dtype=np.uint8)).save(
        pred_dir / "face.png"
    )
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(
        gt_dir / "face.tif"
    )
    output = tmp_path / "out" / "metrics.csv"

    # gọi thẳng main để test đúng parser và đúng luồng ghi file của CLI
    assert main(
        [
            "--pred_dir", str(pred_dir),
            "--gt_dir", str(gt_dir),
            "--out", str(output),
        ]
    ) == 0

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    # dự đoán 2 pixel dương, trong đó đúng 1 pixel (GT chỉ có 1 pixel dương):
    # TP = 1, FP = 1, FN = 0 nên Dice = 2/3, IoU = 1/2, precision = 1/2, recall = 1
    assert rows == [
        {
            "image_id": "face",
            "dice": "0.6666666666666666",
            "iou": "0.5",
            "precision": "0.5",
            "recall": "1.0",
            "area_gt": "1",
            "area_pred": "2",
        }
    ]
    summary = capsys.readouterr().out
    # CLI phải in cả thống kê theo ảnh (macro) lẫn metric gộp toàn tập (micro)
    assert "dice: mean=" in summary
    assert "Micro: dice=" in summary


def test_directory_pairing_reports_missing_masks(tmp_path):
    """Có dự đoán mà không có GT tương ứng thì phải báo lỗi, không được lặng lẽ bỏ qua."""
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()
    (pred_dir / "missing.png").touch()

    with pytest.raises(ValueError, match="missing ground truth"):
        _directory_pairs(pred_dir, gt_dir)


def test_manifest_cli_resolves_paths_relative_to_manifest(tmp_path):
    """Đường dẫn tương đối trong manifest phải được tính từ thư mục chứa file CSV."""
    pred_path = tmp_path / "pred.png"
    gt_path = tmp_path / "gt.png"
    Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(pred_path)
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(gt_path)
    manifest = tmp_path / "pairs.csv"
    manifest.write_text(
        "image_id,pred_path,gt_path\nsample,pred.png,gt.png\n",
        encoding="utf-8",
    )
    output = tmp_path / "metrics.csv"

    assert main(
        ["--manifest", str(manifest), "--out", str(output)]
    ) == 0

    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    # hai mask không chung pixel nào nên Dice = 0
    assert rows[0]["image_id"] == "sample"
    assert rows[0]["dice"] == "0.0"


def test_evaluate_pairs_shape_error_includes_image_id(tmp_path):
    """Hai mask khác shape thì lỗi phải kèm image_id để biết ảnh nào bị."""
    pred_path = tmp_path / "pred.npy"
    gt_path = tmp_path / "gt.npy"
    np.save(pred_path, np.zeros((2, 3), dtype=np.uint8))
    np.save(gt_path, np.zeros((3, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="image 'face-001'.*same shape"):
        evaluate_pairs([("face-001", pred_path, gt_path)])

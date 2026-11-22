"""Test phân tích lỗi (error_analysis) và bảng tổng hợp (report_tables) trên bộ dữ liệu giả nhỏ."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation import error_analysis, evaluate, report_tables
from src.evaluation.error_analysis import error_overlay, image_conditions, tertile_groups
from src.evaluation.seg_metrics import confusion_counts


def build_dataset(root: Path, n=9, size=32):
    """Tạo n ảnh giả cùng mask GT và mask dự đoán, ba thư mục images, gt, pred trong root.

    Ảnh i càng lớn thì dự đoán càng lệch so với GT, nên Dice giảm dần theo i.
    """
    # seed cố định để dữ liệu giống nhau giữa các lần chạy
    rng = np.random.default_rng(0)
    for d in ("images", "gt", "pred"):
        (root / d).mkdir()
    for i in range(n):
        # độ sáng ảnh tăng dần theo i để các nhóm điều kiện (sáng/tối) có sự khác biệt
        image = (rng.random((size, size, 3)) * (40 + 25 * i)).astype(np.uint8)
        gt = np.zeros((size, size), np.uint8)
        gt[8:24, 8:24 - i % 3 * 2] = 255  # diện tích mặt khác nhau giữa các ảnh
        pred = np.zeros_like(gt)
        shift = i  # i càng lớn, dự đoán càng lệch -> Dice giảm dần
        pred[8 + shift:24 + shift, 8:24] = 255
        Image.fromarray(image).save(root / "images" / f"{i:02d}.png")
        Image.fromarray(gt).save(root / "gt" / f"{i:02d}.png")
        Image.fromarray(pred).save(root / "pred" / f"{i:02d}.png")


def test_full_flow_evaluate_then_analyze_then_report(tmp_path):
    """Chạy trọn chuỗi evaluate, error_analysis rồi report_tables và kiểm tra các file đầu ra."""
    build_dataset(tmp_path)
    metrics = tmp_path / "metrics.csv"
    # bước 1: tính metric từng ảnh
    assert evaluate.main(["--pred_dir", str(tmp_path / "pred"),
                          "--gt_dir", str(tmp_path / "gt"), "--out", str(metrics)]) == 0
    out_dir = tmp_path / "ea"
    # bước 2: phân tích lỗi, lấy 3 ví dụ cho mỗi nhóm
    summary = error_analysis.analyze(metrics, tmp_path / "images", tmp_path / "gt",
                                     tmp_path / "pred", out_dir, k=3)
    assert summary["n_images"] == 9
    # ảnh i = 8 lệch nhiều nhất nên Dice thấp nhất, phải đứng đầu danh sách tệ nhất
    assert summary["worst_ids"][0] == "08"
    for name in ("examples_worst.png", "examples_median.png", "examples_best.png",
                 "error_by_condition.csv", "per_image_conditions.csv", "worst_cases.csv"):
        assert (out_dir / name).is_file()
    with (out_dir / "error_by_condition.csv").open() as f:
        table = list(csv.DictReader(f))
    assert len(table) == 12  # 4 điều kiện x 3 nhóm
    # mỗi điều kiện chia đủ 9 ảnh vào 3 nhóm nên cộng lại phải ra 9
    assert sum(int(r["n"]) for r in table if r["condition"] == "brightness") == 9
    # pred dịch xuống nên có cả pixel tô thừa lẫn bỏ sót; mỗi ảnh chỉ thuộc một trong hai loại
    assert summary["over_segmentation_images"] + summary["under_segmentation_images"] <= 9

    # bước 3: bảng báo cáo, dùng cùng một CSV cho hai run A và B cho đơn giản
    assert report_tables.main(["--run", f"A={metrics}", "--run", f"B={metrics}",
                              "--out_dir", str(tmp_path / "rep")]) == 0
    text = (tmp_path / "rep" / "results_tables.md").read_text(encoding="utf-8")
    assert "| A |" in text and "| B |" in text
    assert (tmp_path / "rep" / "dice_boxplot.png").is_file()


def test_micro_stats_match_directly_aggregated_counts(tmp_path):
    """micro_stats suy TP từ Dice trong CSV, kết quả phải khớp với việc cộng thẳng TP/FP/FN từ mask."""
    build_dataset(tmp_path)
    metrics = tmp_path / "m.csv"
    evaluate.main(["--pred_dir", str(tmp_path / "pred"), "--gt_dir", str(tmp_path / "gt"),
                   "--out", str(metrics)])
    micro = report_tables.micro_stats(report_tables.load_per_image(metrics))
    # cách tính đối chứng: đọc lại từng mask và cộng dồn số đếm
    tp = fp = fn = 0
    for i in range(9):
        gt = np.asarray(Image.open(tmp_path / "gt" / f"{i:02d}.png"))
        pred = np.asarray(Image.open(tmp_path / "pred" / f"{i:02d}.png"))
        a, b, c, _ = confusion_counts(pred, gt)
        tp, fp, fn = tp + int(a[0]), fp + int(b[0]), fn + int(c[0])
    assert micro["dice"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert micro["iou"] == pytest.approx(tp / (tp + fp + fn))


def _uyen_json(path, thr=None, detector=None):
    """Ghi file JSON kết quả detection giả theo định dạng của Uyên (mỗi file một detector)."""
    group = {"n_gt": 10, "tp": 8, "fp": 2, "fn": 2, "precision": 0.8, "recall": 0.8,
             "f1": 0.8, "ap": 0.75}
    data = {g: dict(group) for g in ("all", "small", "medium", "large")}
    # score_thr và detector là hai khóa tùy chọn, bản detect_metrics cũ không ghi hai khóa này
    if thr is not None:
        data["score_thr"] = thr
    if detector:
        data["detector"] = detector
    path.write_text(__import__("json").dumps(data))


def test_detection_table_reads_uyen_per_detector_files(tmp_path):
    """Bảng detection đọc được file của Uyên, lấy ngưỡng từ file hoặc từ phần @ngưỡng trong tham số."""
    from src.evaluation.report_tables import detection_markdown
    _uyen_json(tmp_path / "r.json", 0.5, "retinaface")
    _uyen_json(tmp_path / "h.json")  # file cũ không có ngưỡng -> truyền @ngưỡng
    text = detection_markdown([str(tmp_path / "r.json"), f"haar={tmp_path / 'h.json'}@0.693"])
    assert "retinaface (score ≥ 0.5)" in text and "haar (score ≥ 0.693)" in text
    # mỗi detector có đúng một dòng nhóm "all" nên hai detector là hai dòng
    assert text.count("| all |") == 2

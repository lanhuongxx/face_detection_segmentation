"""Test phần đánh giá detection (detect_metrics, wider_data) bằng các ví dụ tính tay."""

import numpy as np
import pytest

from src.detection.detect_metrics import (
    average_precision, box_iou, evaluate_detections, match_image, prf,
)
from src.detection.wider_data import image_key, load_ground_truth


def test_iou_by_hand():
    """IoU của hai box có giao một nửa: giao 50, hợp 150 nên IoU = 1/3."""
    # hai box 10x10 chồng nhau 5x10: giao 50, hợp 100 + 100 - 50 = 150
    iou = box_iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])
    assert iou[0, 0] == pytest.approx(1 / 3)


def test_ap_example_from_plan():
    """Ví dụ AP trong evaluation_plan mục 4.4, kết quả đúng phải là 0.6875."""
    # 4 GT, prediction theo thứ tự TP, TP, FP, TP, FP -> AP = 0.6875 (evaluation_plan 4.4)
    status = np.array([1, 1, 0, 1, 0])
    # tp, fp là số TP và FP tích lũy sau mỗi prediction, từ đó ra precision và recall
    tp, fp = np.cumsum(status == 1), np.cumsum(status == 0)
    ap = average_precision(tp / (tp + fp), tp / 4)
    assert ap == pytest.approx(0.6875)


def test_each_gt_matched_once():
    """Hai prediction trùng một GT thì chỉ box đầu là TP, box thứ hai là FP."""
    gt = np.array([[0, 0, 10, 10.0]])
    boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
    _, status = match_image(boxes, [0.9, 0.8], gt, np.array([False]))
    assert list(status) == [1, 0]  # box thứ hai trùng GT đã ghép nên là FP


def test_invalid_gt_is_ignored():
    """Prediction trùng mặt invalid thì bị bỏ qua, không tính là FP."""
    gt = np.array([[0, 0, 10, 10.0]])
    _, status = match_image([[0, 0, 10, 10]], [0.9], gt, np.array([True]))
    assert list(status) == [-1]  # không phải FP


def test_empty_convention():
    """Quy ước khi mẫu số bằng 0: không có gì cả (0, 0, 0) thì cả ba bằng 1, các trường hợp khác bằng 0."""
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)
    # có GT nhưng không bắt được gì
    assert prf(0, 0, 3) == (0.0, 0.0, 0.0)
    # không có GT nhưng đoán ra mặt (toàn FP)
    assert prf(0, 2, 0) == (0.0, 0.0, 0.0)


def write_gt(tmp_path):
    """Ghi file nhãn WIDER giả có hai ảnh: ảnh a có 1 mặt hợp lệ và 1 mặt invalid, ảnh b không có mặt."""
    f = tmp_path / "gt.txt"
    # cột thứ 8 của mỗi dòng nhãn là cờ invalid: mặt thứ hai đặt bằng 1.
    # ảnh b không có mặt nên có thêm dòng giữ chỗ toàn số 0
    f.write_text("0--A/a.jpg\n2\n10 10 20 20 0 0 0 0 0 0\n100 100 50 50 0 0 0 1 0 0\n"
                 "0--A/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n", encoding="utf-8")
    return f


def test_parse_gt_with_empty_image(tmp_path):
    """Đọc file nhãn: đổi (x, y, w, h) sang (x1, y1, x2, y2), đánh dấu invalid, xử lý ảnh không có mặt."""
    gt = load_ground_truth(write_gt(tmp_path))
    assert set(gt) == {"0--A/a.jpg", "0--A/b.jpg"}
    boxes, invalid = gt["0--A/a.jpg"]
    assert boxes[0].tolist() == [10, 10, 30, 30]  # x2 = x + w
    assert invalid.tolist() == [False, True]
    assert len(gt["0--A/b.jpg"][0]) == 0


def test_evaluate_end_to_end(tmp_path):
    """Chạy cả quy trình đánh giá trên dữ liệu nhỏ, kiểm tra số TP, FP, FN, precision và recall."""
    gt = load_ground_truth(write_gt(tmp_path))
    # khóa của predictions là đường dẫn đầy đủ, evaluate_detections tự cắt về "<event>/<tên file>"
    preds = {
        "WIDER_val/images/0--A/a.jpg": [
            {"bbox": [10, 10, 30, 30], "score": 0.9},      # TP
            {"bbox": [100, 100, 150, 150], "score": 0.9},  # trùng mặt invalid, bỏ qua
            {"bbox": [300, 300, 320, 320], "score": 0.8},  # FP
        ],
        "WIDER_val/images/0--A/b.jpg": [],
    }
    r = evaluate_detections(preds, gt)["all"]
    # chỉ còn 1 GT hợp lệ (mặt invalid không tính vào n_gt)
    assert (r["n_gt"], r["tp"], r["fp"], r["fn"]) == (1, 1, 1, 0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def test_missing_prediction_is_error(tmp_path):
    """Thiếu prediction cho một ảnh trong GT thì phải báo lỗi, không được âm thầm bỏ qua."""
    gt = load_ground_truth(write_gt(tmp_path))
    with pytest.raises(ValueError):
        evaluate_detections({"x/0--A/a.jpg": []}, gt)


def test_size_group_ignores_other_sizes(tmp_path):
    """Khi xét một nhóm kích thước, GT và prediction của nhóm khác bị bỏ qua, không tính FP oan."""
    gt = load_ground_truth(write_gt(tmp_path))
    preds = {"0--A/a.jpg": [{"bbox": [10, 10, 30, 30], "score": 0.9}], "0--A/b.jpg": []}
    res = evaluate_detections(preds, gt)  # mặt cao 20 px -> nhóm small
    assert res["small"]["n_gt"] == 1 and res["medium"]["n_gt"] == 0
    assert res["medium"]["fp"] == 0  # box nhóm khác không bị tính FP


def test_image_key():
    """image_key đổi đường dẫn kiểu Windows (dấu \\) về dạng '<event>/<tên file>'."""
    assert image_key("WIDER_val\\images\\0--A\\a.jpg") == "0--A/a.jpg"

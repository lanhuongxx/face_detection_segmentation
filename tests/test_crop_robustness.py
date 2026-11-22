"""Test logic các cách cắt trong crop_robustness, dùng engine giả (không cần code hay weights của Hương)."""

import numpy as np

from src.evaluation.crop_robustness import (
    box_iou, evaluate_image, jitter_box, mask_bbox, summarize,
)
from src.evaluation.pipeline_adapter import MaskOutput


def scene(size=100, lo=30, hi=70):
    """Ảnh đen size x size và GT là một hình vuông True từ lo đến hi."""
    image = np.zeros((size, size, 3), np.uint8)
    gt = np.zeros((size, size), bool)
    gt[lo:hi, lo:hi] = True
    return image, gt


class FakeEngine:
    """Engine giả: mask dự đoán chính là hộp được đưa vào, còn detector trả hộp cho trước (hoặc không thấy mặt)."""

    def __init__(self, detected_box=None):
        self.detected_box = detected_box

    @staticmethod
    def _paint(shape, box):
        """Tô kín vùng bbox lên mask, max(..., 0) để box lọt ra ngoài ảnh vẫn không bị lỗi chỉ số âm."""
        mask = np.zeros(shape, bool)
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
        return mask

    def mask_from_boxes(self, image, boxes):
        return MaskOutput(self._paint(image.shape[:2], boxes[0]), tuple(boxes[0]), 1, False)

    def mask_from_detector(self, image):
        # detected_box = None nghĩa là detector không thấy mặt, trả mask rỗng và n_faces = 0
        if self.detected_box is None:
            return MaskOutput(np.zeros(image.shape[:2], bool), None, 0, False)
        return self.mask_from_boxes(image, [self.detected_box])


def test_jitter_and_iou_known_values():
    """Kiểm tra jitter_box và box_iou với vài ví dụ tự tính tay."""
    # dịch 0.5 chiều rộng sang phải: (0,0,10,10) thành (5,0,15,10)
    assert jitter_box((0, 0, 10, 10), 0.5, 0, 1.0) == (5, 0, 15, 10)
    # thu nhỏ còn một nửa quanh tâm: cạnh từ 10 xuống 5 nên còn (2.5, 2.5, 7.5, 7.5)
    assert jitter_box((0, 0, 10, 10), 0, 0, 0.5) == (2.5, 2.5, 7.5, 7.5)
    # giao là 5x10 = 50, hợp là 100 + 100 - 50 = 150 nên IoU = 1/3
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == 1 / 3
    # hai box không chồng nhau thì IoU bằng 0
    assert box_iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_missed_detection_counts_as_zero_dice_and_miss():
    """Detector không thấy mặt thì pred_crop phải tính là miss và Dice = 0, còn gt_crop vẫn hoàn hảo."""
    image, gt = scene()
    rows = {r["setting"]: r for r in evaluate_image(image, gt, FakeEngine(None))}
    assert rows["pred_crop"]["detected"] == 0 and rows["pred_crop"]["dice"] == 0.0
    assert rows["gt_crop"]["dice"] == 1.0


def test_summary_reports_drop_miss_rate_and_extra_columns():
    """summarize phải tính đúng miss_rate, mức Dice giảm so với gt_crop và có đủ các cột phụ."""
    image, gt = scene()
    # hai ảnh: ảnh 0 detector bắt đúng mặt, ảnh 1 detector không thấy mặt
    rows = [{"image_id": str(i), **r}
            for i, eng in enumerate((FakeEngine((30, 30, 70, 70)), FakeEngine(None)))
            for r in evaluate_image(image, gt, eng)]
    s = {r["setting"]: r for r in summarize(rows)}
    # 1 trên 2 ảnh bị miss nên miss_rate = 0.5; pred_crop có Dice 1 và 0 nên trung bình 0.5,
    # còn gt_crop luôn là 1 nên mức giảm là 1 - 0.5 = 0.5
    assert s["pred_crop"]["n"] == 2 and s["pred_crop"]["miss_rate"] == 0.5
    assert s["pred_crop"]["dice_drop_vs_gt"] == 0.5
    assert "fallback_rate" in s["gt_crop"] and "n_faces_mean" in s["gt_crop"]

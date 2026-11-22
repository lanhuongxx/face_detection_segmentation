"""Test engine nối với FacePipeline của Hương, dùng detector và segmenter giả (không cần weights)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("src.pipeline.inference")  # chỉ chạy khi repo có code của Hương

from src.evaluation.crop_robustness import evaluate_image  # noqa: E402
from src.evaluation.pipeline_adapter import PipelineEngine  # noqa: E402
from src.evaluation.seg_metrics import dice_score  # noqa: E402
from src.pipeline.schemas import Detection  # noqa: E402


class FakeSegmenter:
    """Segmenter giả: trả hình elip nằm giữa crop, giống một U-Net chạy đúng."""

    def __init__(self, empty=False):
        self.empty = empty

    def predict(self, crops):
        n, s = crops.shape[0], crops.shape[1]
        out = np.zeros((n, s, s), np.float32)
        # empty=True thì trả mask rỗng, dùng để mô phỏng trường hợp U-Net không tìm được mặt
        if not self.empty:
            for i in range(n):
                cv2.ellipse(out[i], (s // 2, s // 2), (int(s * 0.28), int(s * 0.34)),
                            0, 0, 360, 1.0, -1)
        return out


class FakeDetector:
    """Detector giả: trả các (bbox, score) cho trước, chỉ giữ box có score đạt ngưỡng."""

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image, conf_threshold=0.5):
        return [Detection(b, s) for b, s in self.boxes if s >= conf_threshold]


def scene(h=300, w=400, cx=200, cy=150, a=50, b=60):
    """Ảnh xám cỡ h x w có một mặt hình elip, trả (ảnh, mask GT, bbox GT)."""
    image = np.full((h, w, 3), 80, np.uint8)
    gt = np.zeros((h, w), np.uint8)
    cv2.ellipse(gt, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    return image, gt > 0, (cx - a, cy - b, cx + a, cy + b)


def engine(**kw):
    # config_path=None để không cần file cấu hình, dùng giá trị mặc định của pipeline
    return PipelineEngine(config_path=None, **kw)


def test_mask_from_boxes_has_image_shape_and_lands_on_box():
    """Mask trả về phải cùng cỡ ảnh gốc và nằm đúng chỗ bbox được đưa vào."""
    image, gt, box = scene()
    out = engine(segmenter=FakeSegmenter()).mask_from_boxes(image, [box])
    assert out.mask.shape == image.shape[:2] and out.n_faces == 1 and not out.fallback
    # tâm của mask phải gần tâm mặt thật (200, 150), cho phép lệch vài pixel do resize
    ys, xs = np.nonzero(out.mask)
    assert abs(xs.mean() - 200) < 6 and abs(ys.mean() - 150) < 6
    assert float(dice_score(out.mask, gt)) > 0.9


def test_oversized_image_is_rejected_instead_of_silently_misaligned():
    """Ảnh quá lớn bị pipeline resize thì phải báo lỗi, vì mask sẽ lệch cỡ với GT nếu cứ tiếp tục."""
    image = np.zeros((40, 5000, 3), np.uint8)
    with pytest.raises(ValueError, match="resize"):
        engine(segmenter=FakeSegmenter()).mask_from_boxes(image, [(100, 5, 200, 35)])

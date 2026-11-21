"""Test engine nối với FacePipeline của Hương, dùng detector/segmenter giả (không cần weights)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("src.pipeline.inference")  # chỉ chạy khi repo có code của Hương

from src.evaluation.crop_robustness import evaluate_image  # noqa: E402
from src.evaluation.pipeline_adapter import PipelineEngine  # noqa: E402
from src.evaluation.seg_metrics import dice_score  # noqa: E402
from src.pipeline.schemas import Detection  # noqa: E402


class FakeSegmenter:
    """Elip nằm giữa crop, như U-Net chạy đúng."""

    def __init__(self, empty=False):
        self.empty = empty

    def predict(self, crops):
        n, s = crops.shape[0], crops.shape[1]
        out = np.zeros((n, s, s), np.float32)
        if not self.empty:
            for i in range(n):
                cv2.ellipse(out[i], (s // 2, s // 2), (int(s * 0.28), int(s * 0.34)),
                            0, 0, 360, 1.0, -1)
        return out


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image, conf_threshold=0.5):
        return [Detection(b, s) for b, s in self.boxes if s >= conf_threshold]


def scene(h=300, w=400, cx=200, cy=150, a=50, b=60):
    image = np.full((h, w, 3), 80, np.uint8)
    gt = np.zeros((h, w), np.uint8)
    cv2.ellipse(gt, (cx, cy), (a, b), 0, 0, 360, 255, -1)
    return image, gt > 0, (cx - a, cy - b, cx + a, cy + b)


def engine(**kw):
    return PipelineEngine(config_path=None, **kw)


def test_mask_from_boxes_has_image_shape_and_lands_on_box():
    image, gt, box = scene()
    out = engine(segmenter=FakeSegmenter()).mask_from_boxes(image, [box])
    assert out.mask.shape == image.shape[:2] and out.n_faces == 1 and not out.fallback
    ys, xs = np.nonzero(out.mask)
    assert abs(xs.mean() - 200) < 6 and abs(ys.mean() - 150) < 6
    assert float(dice_score(out.mask, gt)) > 0.9


def test_oversized_image_is_rejected_instead_of_silently_misaligned():
    image = np.zeros((40, 5000, 3), np.uint8)
    with pytest.raises(ValueError, match="resize"):
        engine(segmenter=FakeSegmenter()).mask_from_boxes(image, [(100, 5, 200, 35)])

"""Unit tests for API contract types and validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import (
    FaceDetection,
    clamp_bbox,
    coerce_detection,
    validate_prob_mask,
    validate_rgb_image,
)


def test_validate_rgb_image_normalizes_float() -> None:
    img_f = np.full((32, 48, 3), 0.5, dtype=np.float32)
    out = validate_rgb_image(img_f)
    assert out.dtype == np.uint8
    assert out.shape == (32, 48, 3)
    assert int(out[0, 0, 0]) == 127


def test_validate_rgb_image_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        validate_rgb_image(np.zeros((32, 32), dtype=np.uint8))


def test_clamp_bbox_bounds_and_degenerate() -> None:
    assert clamp_bbox((-10.4, -2.1, 105.2, 85.9), (80, 100)) == (0, 0, 100, 80)
    assert clamp_bbox((50, 50, 40, 60), (80, 100)) is None
    assert clamp_bbox((120, 10, 140, 30), (80, 100)) is None


def test_coerce_detection_formats() -> None:
    d1 = coerce_detection({"facial_area": [10, 20, 50, 70], "score": 0.92}, (100, 100))
    assert isinstance(d1, FaceDetection)
    assert d1.bbox == (10, 20, 50, 70)
    assert d1.score == pytest.approx(0.92)

    d2 = coerce_detection((5, 5, 25, 35, 0.8), (100, 100))
    assert d2 == FaceDetection(bbox=(5, 5, 25, 35), score=0.8)


def test_validate_prob_mask_uint8_scaling() -> None:
    m = np.full((16, 16), 255, dtype=np.uint8)
    out = validate_prob_mask(m, expected_size=(16, 16))
    assert out.dtype == np.float32
    assert float(out.max()) == pytest.approx(1.0)

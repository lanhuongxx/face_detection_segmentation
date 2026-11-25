"""Unit tests for PrivacyFilter modes and image saving."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.contracts import FaceDetection
from src.inference import FacePrivacyPipeline, PipelineConfig
from src.privacy_filter import PrivacyFilter, PrivacyFilterConfig


def _make_textured_image(h: int = 160, w: int = 200) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_pixelate_modifies_only_masked_region() -> None:
    img = _make_textured_image()
    mask = np.zeros((160, 200), dtype=np.uint8)
    mask[40:100, 60:130] = 1

    pf = PrivacyFilter()
    out = pf.apply_with_mask(
        img,
        mask,
        config=PrivacyFilterConfig(mode="pixelate", pixel_block_size=12),
    )

    assert not np.array_equal(out[40:100, 60:130], img[40:100, 60:130])
    # Background pixels outside the mask must remain strictly untouched.
    assert np.array_equal(out[:35, :], img[:35, :])
    assert np.array_equal(out[105:, :], img[105:, :])


def test_blur_modifies_face_and_keeps_far_background_exact() -> None:
    img = _make_textured_image()
    det = FaceDetection(bbox=(60, 40, 120, 100), score=0.99)
    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(crop_margin_ratio=0.0, unet_input_size=(64, 64)),
        detector=lambda _img: [det],
        segmentor=lambda crop: np.ones(crop.shape[:2], dtype=np.float32),
    )
    output = pipeline.run(img)

    pf = PrivacyFilter()
    blurred = pf.apply(
        img,
        output,
        config=PrivacyFilterConfig(mode="blur", blur_strength=25, feather_radius=1),
    )

    assert not np.array_equal(blurred[50:90, 70:110], img[50:90, 70:110])
    assert np.array_equal(blurred[:20, :20], img[:20, :20])


def test_mask_and_mask_only_modes_and_save_image(tmp_path: Path) -> None:
    img = _make_textured_image()
    det1 = FaceDetection(bbox=(20, 20, 60, 60), score=0.95)
    det2 = FaceDetection(bbox=(100, 70, 150, 120), score=0.90)
    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(crop_margin_ratio=0.0, unet_input_size=(32, 32)),
        detector=lambda _img: [det1, det2],
        segmentor=lambda crop: np.ones(crop.shape[:2], dtype=np.float32),
    )
    output = pipeline.run(img)

    pf = PrivacyFilter()
    overlay = pf.apply(
        img,
        output,
        config=PrivacyFilterConfig(mode="mask", draw_bbox=True, mask_alpha=0.5),
    )
    assert overlay.shape == img.shape

    mask_only = pf.apply(
        img,
        output,
        config=PrivacyFilterConfig(mode="mask_only"),
    )
    assert np.array_equal(mask_only[5, 5], np.array([0, 0, 0], dtype=np.uint8))
    assert np.any(mask_only[35, 35] > 0)
    assert np.any(mask_only[90, 120] > 0)

    saved_path = pf.save_image(overlay, tmp_path / "nested" / "result.png")
    assert saved_path.exists()
    reloaded = cv2.imread(str(saved_path))
    assert reloaded is not None
    assert reloaded.shape == img.shape

    encoded = pf.encode_image(overlay, ext=".jpg")
    assert len(encoded) > 100

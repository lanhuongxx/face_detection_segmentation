"""Unit and integration tests for detection -> crop -> segmentation -> mask map-back."""

from __future__ import annotations

import numpy as np

from src.contracts import FaceDetection
from src.inference import (
    FacePrivacyPipeline,
    PipelineConfig,
    RetinaFaceDetectorAdapter,
)


def _filled_crop_segmentor(crop: np.ndarray) -> np.ndarray:
    """Deterministic segmentor returning a centered disc inside each crop."""
    h, w = crop.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    mask = (((yy - cy) / (h * 0.35)) ** 2 + ((xx - cx) / (w * 0.35)) ** 2) <= 1.0
    return mask.astype(np.float32)


def test_pipeline_zero_faces_returns_empty_masks() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(unet_input_size=(64, 64)),
        detector=lambda _img: [],
        segmentor=_filled_crop_segmentor,
    )
    out = pipeline.run(image)
    assert out.num_faces == 0
    assert out.faces == []
    assert out.combined_mask.shape == (120, 160)
    assert int(out.combined_mask.sum()) == 0


def test_pipeline_multiple_faces_exact_mask_map_back() -> None:
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    # Three distinct faces, including one near the bottom-right image border.
    detections = [
        FaceDetection(bbox=(20, 30, 80, 90), score=0.98),
        FaceDetection(bbox=(160, 40, 220, 100), score=0.95),
        FaceDetection(bbox=(270, 190, 318, 238), score=0.91),
    ]

    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(
            crop_margin_ratio=0.0,
            unet_input_size=(64, 64),
            mask_threshold=0.5,
        ),
        detector=lambda _img: detections,
        segmentor=_filled_crop_segmentor,
    )

    out = pipeline.run(image)
    assert out.num_faces == 3

    for face, det in zip(out.faces, detections):
        x1, y1, x2, y2 = det.bbox
        assert face.crop_metadata.crop_box == det.bbox
        assert face.full_mask.shape == (240, 320)

        # Mask must be non-empty inside the face box center and strictly 0 outside crop_box.
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        assert face.full_mask[center_y, center_x] == 1

        outside = face.full_mask.copy()
        outside[y1:y2, x1:x2] = 0
        assert int(outside.sum()) == 0

    # Combined mask must equal the exact pixel-wise union of all individual face masks.
    expected_union = np.clip(
        sum(f.full_mask.astype(np.int32) for f in out.faces),
        0,
        1,
    ).astype(np.uint8)
    assert np.array_equal(out.combined_mask, expected_union)


def test_pipeline_overlapping_faces_preserve_both_masks() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [
        FaceDetection(bbox=(40, 40, 110, 110), score=0.99),
        FaceDetection(bbox=(80, 70, 150, 140), score=0.94),
    ]
    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(crop_margin_ratio=0.0, unet_input_size=(64, 64)),
        detector=lambda _img: detections,
        segmentor=lambda crop: np.ones(crop.shape[:2], dtype=np.float32),
    )

    out = pipeline.run(image)
    assert out.num_faces == 2
    assert int(out.combined_mask[45, 45]) == 1
    assert int(out.combined_mask[135, 145]) == 1
    assert int(out.combined_mask[10, 10]) == 0


def test_detector_adapter_handles_retinaface_dict_format() -> None:
    raw_retinaface = {
        "face_1": {
            "score": 0.99,
            "facial_area": [10, 20, 60, 80],
            "landmarks": {"left_eye": [25.0, 40.0], "right_eye": [45.0, 40.0]},
        },
        "face_2": {
            "score": 0.25,  # Below threshold -> filtered out
            "facial_area": [100, 100, 140, 140],
        },
    }
    adapter = RetinaFaceDetectorAdapter(
        conf_threshold=0.5,
        custom_detector=lambda _img: raw_retinaface,
    )
    dets = adapter.detect(np.zeros((200, 200, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0].bbox == (10, 20, 60, 80)
    assert dets[0].landmarks is not None
    assert "left_eye" in dets[0].landmarks


def test_default_auto_pipeline_initializes_and_runs_on_blank_image() -> None:
    pipeline = FacePrivacyPipeline()
    out = pipeline.run(np.zeros((160, 160, 3), dtype=np.uint8))
    assert out.num_faces == 0
    assert out.combined_mask.shape == (160, 160)
    assert pipeline.detector.backend_used in ("team_retinaface", "retinaface", "haar_fallback", "skin_fallback")
    assert pipeline.segmentor.backend_used in ("unet_checkpoint", "heuristic_fallback")


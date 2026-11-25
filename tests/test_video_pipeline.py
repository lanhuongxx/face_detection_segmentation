"""Integration tests for short video reading/writing and FPS preservation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.contracts import FaceDetection
from src.inference import FacePrivacyPipeline, PipelineConfig
from src.privacy_filter import PrivacyFilter, PrivacyFilterConfig
from src.video_pipeline import VideoConfig, VideoPrivacyProcessor, resolve_source_fps


def _write_synthetic_video(path: Path, fps: float, num_frames: int, size_wh: tuple[int, int]) -> None:
    w, h = size_wh
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened()
    try:
        for i in range(num_frames):
            frame = np.full((h, w, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (30 + i * 2, 30), (90 + i * 2, 90), (200, 180, 160), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_resolve_source_fps_fallback() -> None:
    assert resolve_source_fps(23.976, fallback_fps=30.0) == pytest.approx(23.976)
    assert resolve_source_fps(0.0, fallback_fps=25.0) == pytest.approx(25.0)
    assert resolve_source_fps(float("nan"), fallback_fps=30.0) == pytest.approx(30.0)


@pytest.mark.parametrize("source_fps", [15.0, 24.0])
def test_video_pipeline_preserves_original_fps_and_frame_count(
    tmp_path: Path,
    source_fps: float,
) -> None:
    in_video = tmp_path / f"sample_{int(source_fps)}.mp4"
    out_video = tmp_path / f"out_{int(source_fps)}.mp4"
    num_frames = 8
    _write_synthetic_video(in_video, fps=source_fps, num_frames=num_frames, size_wh=(160, 120))

    pipeline = FacePrivacyPipeline(
        config=PipelineConfig(crop_margin_ratio=0.0, unet_input_size=(32, 32)),
        detector=lambda _img: [FaceDetection(bbox=(30, 30, 90, 90), score=0.96)],
        segmentor=lambda crop: np.ones(crop.shape[:2], dtype=np.float32),
    )
    processor = VideoPrivacyProcessor(
        pipeline=pipeline,
        privacy_filter=PrivacyFilter(PrivacyFilterConfig(mode="blur", blur_strength=15)),
    )

    progress_events: list[tuple[int, int]] = []
    result = processor.process_video(
        in_video,
        out_video,
        video_config=VideoConfig(frame_stride=2),
        progress_callback=lambda done, total: progress_events.append((done, total)),
    )

    assert out_video.exists()
    assert result.written_frames == num_frames
    assert result.inferred_frames == 4  # frame_stride=2 on 8 frames -> 4 inferences, 8 frames written
    assert result.output_fps == pytest.approx(source_fps, rel=1e-2)
    assert len(progress_events) == num_frames

    cap = cv2.VideoCapture(str(out_video))
    try:
        assert cap.isOpened()
        written_fps = float(cap.get(cv2.CAP_PROP_FPS))
        written_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        written_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert written_fps == pytest.approx(source_fps, rel=1e-2)
        assert (written_w, written_h) == (160, 120)
    finally:
        cap.release()

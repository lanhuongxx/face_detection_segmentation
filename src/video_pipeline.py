"""Short video privacy processing pipeline preserving source FPS and dimensions."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from src.contracts import PipelineOutput
from src.inference import FacePrivacyPipeline
from src.privacy_filter import PrivacyFilter, PrivacyFilterConfig


@dataclass
class VideoConfig:
    """Configuration for reading and writing processed videos."""

    codec: str = "mp4v"
    frame_stride: int = 1
    max_frames: int | None = None
    fallback_fps: float = 30.0


@dataclass
class VideoProcessResult:
    """Summary metadata returned after processing a video."""

    input_path: Path
    output_path: Path
    input_fps: float
    output_fps: float
    width: int
    height: int
    total_input_frames: int
    written_frames: int
    inferred_frames: int
    total_faces_detected: int
    avg_inference_ms: float
    elapsed_sec: float


def resolve_source_fps(raw_fps: float, fallback_fps: float = 30.0) -> float:
    """Return valid source FPS if finite and positive, otherwise `fallback_fps`."""
    if math.isfinite(raw_fps) and raw_fps > 0.0:
        return float(raw_fps)
    return float(fallback_fps)


def _open_video_writer(
    output_path: Path,
    fps: float,
    frame_size: tuple[int, int],
    preferred_codec: str,
) -> cv2.VideoWriter:
    codecs_to_try = [preferred_codec]
    for fallback in ("mp4v", "XVID", "MJPG"):
        if fallback not in codecs_to_try:
            codecs_to_try.append(fallback)

    for codec in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        f"Could not open VideoWriter for {output_path} with codecs {codecs_to_try}."
    )


class VideoPrivacyProcessor:
    """Processes short videos frame-by-frame while preserving original FPS."""

    def __init__(
        self,
        pipeline: FacePrivacyPipeline | None = None,
        privacy_filter: PrivacyFilter | None = None,
    ) -> None:
        self.pipeline = pipeline or FacePrivacyPipeline()
        self.privacy_filter = privacy_filter or PrivacyFilter()

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        filter_config: PrivacyFilterConfig | None = None,
        video_config: VideoConfig | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> VideoProcessResult:
        """Read `input_path`, apply face mask/blur per frame, and write `output_path` at source FPS."""
        in_path = Path(input_path)
        out_path = Path(output_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {in_path}")

        vcfg = video_config or VideoConfig()
        if vcfg.frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {vcfg.frame_stride}.")
        if vcfg.max_frames is not None and vcfg.max_frames < 1:
            raise ValueError(f"max_frames must be >= 1 when set, got {vcfg.max_frames}.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {in_path}")

        writer: cv2.VideoWriter | None = None
        t_start = time.perf_counter()

        try:
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
            output_fps = resolve_source_fps(raw_fps, fallback_fps=vcfg.fallback_fps)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid video dimensions {(width, height)} in {in_path}.")

            writer = _open_video_writer(
                out_path,
                fps=output_fps,
                frame_size=(width, height),
                preferred_codec=vcfg.codec,
            )

            written_frames = 0
            inferred_frames = 0
            total_faces = 0
            total_infer_ms = 0.0
            last_output: PipelineOutput | None = None

            while True:
                if vcfg.max_frames is not None and written_frames >= vcfg.max_frames:
                    break

                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if written_frames % vcfg.frame_stride == 0 or last_output is None:
                    last_output = self.pipeline.run(frame_rgb)
                    inferred_frames += 1
                    total_infer_ms += float(last_output.timings_ms.get("total_ms", 0.0))

                total_faces += last_output.num_faces
                rendered_rgb = self.privacy_filter.apply(
                    frame_rgb,
                    last_output,
                    config=filter_config,
                )
                rendered_bgr = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)
                writer.write(rendered_bgr)
                written_frames += 1

                if progress_callback is not None:
                    progress_callback(written_frames, max(reported_total, written_frames))

            elapsed_sec = time.perf_counter() - t_start
            avg_infer_ms = total_infer_ms / inferred_frames if inferred_frames > 0 else 0.0

            return VideoProcessResult(
                input_path=in_path,
                output_path=out_path,
                input_fps=raw_fps if raw_fps > 0 else output_fps,
                output_fps=output_fps,
                width=width,
                height=height,
                total_input_frames=max(reported_total, written_frames),
                written_frames=written_frames,
                inferred_frames=inferred_frames,
                total_faces_detected=total_faces,
                avg_inference_ms=avg_infer_ms,
                elapsed_sec=elapsed_sec,
            )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

"""Nối evaluation với pipeline thật của Hương (``src.pipeline.inference.FacePipeline``).

Vì sao cần: pipeline thật cắt khung VUÔNG quanh bbox (cạnh = max(w,h)*(1+2*margin)), pad
đen phần vượt biên, resize 256x256, chạy U-Net, resize xác suất về khung rồi
``clean_mask`` (giữ mảnh lớn nhất, lấp lỗ), kiểm tra chất lượng mask và có thể thay bằng
elip dự phòng. Tự viết lại các bước đó trong evaluation sẽ lệch dần khỏi pipeline; ở đây
chạy chính ``FacePipeline.analyze`` với detector hoặc bbox được chỉ định, nên số đo là số
của pipeline thật.

Chỉ import ``src.pipeline`` khi tạo engine, nên các module evaluation khác không phụ thuộc
vào code của Hương.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

Box = tuple[float, float, float, float]


@dataclass
class MaskOutput:
    """Kết quả một lần chạy: mask bool cỡ ảnh gốc và thông tin chẩn đoán."""

    mask: np.ndarray
    box: Box | None          # bbox mặt điểm cao nhất mà pipeline dùng; None nếu không có mặt
    n_faces: int
    fallback: bool           # True nếu có mặt phải dùng elip dự phòng (mask hỏng/mặt quá nhỏ)


class PipelineEngine:
    """Chạy FacePipeline của Hương theo hai cách: bbox cho trước hoặc detector thật.

    ``segmenter`` và ``detector`` có thể truyền vào (đối tượng giả khi test). Mặc định
    pipeline tự nạp RetinaFace và U-Net theo ``configs/pipeline.yaml``.
    """

    def __init__(
        self,
        config_path: str | None = "configs/pipeline.yaml",
        weights: str | None = None,
        device: str | None = None,
        segmenter: Any = None,
        detector: Any = None,
    ) -> None:
        from src.pipeline.inference import FacePipeline
        from src.utils.config import load_config

        cfg = load_config(config_path)
        if weights:
            cfg["segmentation"]["weights"] = weights
        if device:
            cfg["device"] = device
        self.cfg = cfg
        self._boxes: list[tuple[int, int, int, int]] = []
        self._fixed = _FixedBoxDetector(self)
        self._real = FacePipeline(cfg, detector=detector, segmenter=segmenter)
        # Hai pipeline dùng chung một segmenter để chỉ nạp U-Net một lần.
        self._by_box = FacePipeline(cfg, detector=self._fixed, segmenter=segmenter)
        self._shared = segmenter

    def _segmenter(self):
        if self._shared is None:
            self._shared = self._real._get_segmenter()
            self._by_box._segmenter = self._shared
        return self._shared

    @staticmethod
    def _to_output(result: Any, image_shape: Sequence[int]) -> MaskOutput:
        if result.image.shape[:2] != tuple(image_shape[:2]):
            raise ValueError(
                "pipeline đã resize ảnh (preprocess.max_side); mask khác cỡ ảnh gốc, "
                "không so được với GT. Tăng max_side hoặc dùng ảnh nhỏ hơn."
            )
        faces = result.faces
        top = max(faces, key=lambda f: f.score) if faces else None
        return MaskOutput(
            mask=result.instance_map > 0,
            box=tuple(float(v) for v in top.bbox) if top else None,
            n_faces=len(faces),
            fallback=any(f.status == "fallback_ellipse" for f in faces),
        )

    def mask_from_boxes(self, image_rgb: np.ndarray, boxes: Sequence[Box]) -> MaskOutput:
        """U-Net + cách crop/map-back của pipeline, nhưng bbox do người gọi chỉ định."""
        self._segmenter()
        self._boxes = [tuple(int(round(v)) for v in box) for box in boxes]
        result = self._by_box.analyze(np.ascontiguousarray(image_rgb), conf_threshold=0.0)
        return self._to_output(result, image_rgb.shape)

    def mask_from_detector(self, image_rgb: np.ndarray) -> MaskOutput:
        """Toàn bộ pipeline: detector thật -> crop -> U-Net -> map-back."""
        self._segmenter()
        self._real._segmenter = self._shared
        result = self._real.analyze(np.ascontiguousarray(image_rgb))
        return self._to_output(result, image_rgb.shape)


class _FixedBoxDetector:
    """Detector giả trả về đúng các bbox mà engine đang giữ (score 1.0)."""

    def __init__(self, engine: PipelineEngine) -> None:
        self._engine = engine

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5):
        from src.pipeline.schemas import Detection

        return [Detection(box, 1.0) for box in self._engine._boxes]

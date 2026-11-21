"""Nối phần evaluation với pipeline thật của Hương (``src.pipeline.inference.FacePipeline``).

Vì sao cần file này: pipeline thật cắt khung vuông quanh bbox (cạnh = max(w, h) * (1 + 2 * margin)),
pad đen phần vượt ra ngoài ảnh, resize về 256x256, chạy U-Net, resize xác suất về lại khung,
rồi qua ``clean_mask`` (giữ mảnh lớn nhất, lấp lỗ). Sau đó còn kiểm tra chất lượng mask và có thể
thay bằng elip dự phòng. Nếu tự viết lại từng bước này trong evaluation thì sớm muộn sẽ lệch
khỏi pipeline. Vì vậy ở đây gọi thẳng ``FacePipeline.analyze`` với detector hoặc bbox mình chỉ
định, để số đo ra chính là số của pipeline thật.

``src.pipeline`` chỉ được import khi tạo engine, nên các module evaluation khác vẫn chạy
được mà không cần code của Hương.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

Box = tuple[float, float, float, float]


@dataclass
class MaskOutput:
    """Kết quả của một lần chạy: mask bool cỡ ảnh gốc kèm vài thông tin để chẩn đoán."""

    mask: np.ndarray
    box: Box | None          # bbox của mặt có điểm cao nhất mà pipeline dùng, None nếu không thấy mặt
    n_faces: int
    fallback: bool           # True nếu có mặt phải dùng elip dự phòng (mask hỏng hoặc mặt quá nhỏ)


class PipelineEngine:
    """Chạy FacePipeline của Hương theo hai cách: bbox cho trước, hoặc detector thật.

    ``segmenter`` và ``detector`` có thể truyền từ ngoài vào (lúc test thì truyền đối tượng giả).
    Nếu không truyền, pipeline tự nạp RetinaFace và U-Net theo ``configs/pipeline.yaml``.
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
        # _real dùng detector thật, còn _by_box dùng detector giả luôn trả đúng bbox mình đưa
        self._real = FacePipeline(cfg, detector=detector, segmenter=segmenter)
        # hai pipeline dùng chung một segmenter để U-Net chỉ phải nạp một lần
        self._by_box = FacePipeline(cfg, detector=self._fixed, segmenter=segmenter)
        self._shared = segmenter

    def _segmenter(self):
        """Lấy segmenter dùng chung. Nếu chưa có thì nhờ pipeline thật nạp rồi chia cho pipeline kia."""
        if self._shared is None:
            self._shared = self._real._get_segmenter()
            self._by_box._segmenter = self._shared
        return self._shared

    @staticmethod
    def _to_output(result: Any, image_shape: Sequence[int]) -> MaskOutput:
        """Đổi kết quả của pipeline thành MaskOutput, kiểm tra mask còn cùng cỡ với ảnh gốc."""
        if result.image.shape[:2] != tuple(image_shape[:2]):
            raise ValueError(
                "pipeline đã resize ảnh (preprocess.max_side); mask khác cỡ ảnh gốc, "
                "không so được với GT. Tăng max_side hoặc dùng ảnh nhỏ hơn."
            )
        faces = result.faces
        # ảnh có nhiều mặt thì lấy mặt có điểm cao nhất làm bbox đại diện
        top = max(faces, key=lambda f: f.score) if faces else None
        return MaskOutput(
            mask=result.instance_map > 0,
            box=tuple(float(v) for v in top.bbox) if top else None,
            n_faces=len(faces),
            fallback=any(f.status == "fallback_ellipse" for f in faces),
        )

    def mask_from_boxes(self, image_rgb: np.ndarray, boxes: Sequence[Box]) -> MaskOutput:
        """Chạy U-Net với cách crop và ghép ngược của pipeline, nhưng bbox do người gọi đưa vào."""
        self._segmenter()
        self._boxes = [tuple(int(round(v)) for v in box) for box in boxes]
        # conf_threshold=0.0 để detector giả (điểm 1.0) không bao giờ bị lọc mất
        result = self._by_box.analyze(np.ascontiguousarray(image_rgb), conf_threshold=0.0)
        return self._to_output(result, image_rgb.shape)

    def mask_from_detector(self, image_rgb: np.ndarray) -> MaskOutput:
        """Chạy cả pipeline: detector thật, crop, U-Net, rồi ghép ngược về ảnh gốc."""
        self._segmenter()
        self._real._segmenter = self._shared
        result = self._real.analyze(np.ascontiguousarray(image_rgb))
        return self._to_output(result, image_rgb.shape)


class _FixedBoxDetector:
    """Detector giả: luôn trả về đúng các bbox mà engine đang giữ, với điểm 1.0."""

    def __init__(self, engine: PipelineEngine) -> None:
        self._engine = engine

    def detect(self, image: np.ndarray, conf_threshold: float = 0.5):
        from src.pipeline.schemas import Detection

        return [Detection(box, 1.0) for box in self._engine._boxes]

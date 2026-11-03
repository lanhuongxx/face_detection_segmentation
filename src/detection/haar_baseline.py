"""Baseline phát hiện mặt bằng Haar Cascade của OpenCV."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class HaarFaceDetector:
    """Phát hiện mặt frontal; score chỉ dùng xếp hạng, không phải xác suất."""

    _CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        equalize_hist: bool = False,
        cascade_path: str | Path | None = None,
    ) -> None:
        """Tạo classifier và kiểm tra tham số trước khi detect."""
        if not math.isfinite(scale_factor) or scale_factor <= 1.0:
            raise ValueError("scale_factor phải là số hữu hạn lớn hơn 1.")
        if (
            isinstance(min_neighbors, bool)
            or not isinstance(min_neighbors, int)
            or min_neighbors < 0
        ):
            raise ValueError("min_neighbors phải là số nguyên không âm.")
        if not isinstance(min_size, (list, tuple)) or len(min_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in min_size
        ):
            raise ValueError("min_size phải gồm hai số nguyên dương (width, height).")
        if not isinstance(equalize_hist, bool):
            raise ValueError("equalize_hist phải là bool.")

        if cascade_path is None:
            cascade_path = (
                Path(cv2.data.haarcascades) / self._CASCADE_FILENAME
            )
        self.scale_factor = float(scale_factor)
        self.min_neighbors = min_neighbors
        self.min_size = tuple(min_size)
        self.equalize_hist = equalize_hist
        self.cascade_path = Path(cascade_path)
        self._classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self._classifier.empty():
            raise FileNotFoundError(
                f"Không tải được Haar cascade: {self.cascade_path}"
            )

    @staticmethod
    def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
        """Kiểm tra ảnh uint8 và chuyển ảnh xám/BGR/BGRA sang grayscale."""
        if not isinstance(image_bgr, np.ndarray):
            raise ValueError("Ảnh đầu vào phải là numpy.ndarray.")
        if image_bgr.dtype != np.uint8:
            raise ValueError("Ảnh đầu vào phải có dtype uint8.")
        if image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào không được rỗng.")

        if image_bgr.ndim == 2:
            gray = image_bgr
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 1:
            gray = image_bgr[:, :, 0]
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Ảnh phải có shape HxW, HxWx1, HxWx3 hoặc HxWx4.")

        if gray.shape[0] == 0 or gray.shape[1] == 0:
            raise ValueError("Ảnh đầu vào không được rỗng.")
        return np.ascontiguousarray(gray)

    @staticmethod
    def _weight_to_score(level_weight: float) -> float:
        """Chuẩn hóa levelWeight bằng sigmoid ổn định về [0, 1]."""
        if not math.isfinite(level_weight):
            raise RuntimeError("Haar cascade trả levelWeight không hữu hạn.")
        if level_weight >= 0:
            return 1.0 / (1.0 + math.exp(-level_weight))
        exp_weight = math.exp(level_weight)
        return exp_weight / (1.0 + exp_weight)

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Trả bbox pixel trong ảnh gốc, score xếp hạng và landmarks=None."""
        gray = self._to_gray(image_bgr)
        if self.equalize_hist:
            gray = cv2.equalizeHist(gray)

        rectangles, _, level_weights = (
            self._classifier.detectMultiScale3(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_size,
                outputRejectLevels=True,
            )
        )
        rectangles = np.asarray(rectangles)
        weights = np.asarray(level_weights).reshape(-1)
        if rectangles.size == 0:
            return []
        if rectangles.ndim != 2 or rectangles.shape[1] != 4:
            raise RuntimeError(
                "Haar cascade trả bbox sai định dạng; kỳ vọng N x 4."
            )
        if rectangles.shape[0] != weights.size:
            raise RuntimeError(
                "Số bbox và levelWeights từ Haar cascade không khớp."
            )

        detections: list[dict[str, Any]] = []
        for (x, y, width, height), weight in zip(rectangles, weights):
            detections.append(
                {
                    "bbox": [
                        float(x),
                        float(y),
                        float(x + width),
                        float(y + height),
                    ],
                    "score": self._weight_to_score(float(weight)),
                    "landmarks": None,
                }
            )
        detections.sort(key=lambda face: face["score"], reverse=True)
        return detections

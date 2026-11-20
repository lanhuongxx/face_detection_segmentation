"""Kiểu dữ liệu dùng chung cho mọi module (xem mục 3.7 của file thiết kế)."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

BBox = tuple  # (x1, y1, x2, y2), số nguyên, tọa độ ảnh đang xử lý


@dataclass
class Detection:
    """Kết quả của detector cho một khuôn mặt."""
    bbox: BBox
    score: float
    landmarks: Optional[np.ndarray] = None      # (5, 2) hoặc None


@dataclass
class Face:
    """Một khuôn mặt sau khi qua toàn bộ pipeline."""
    id: int                                     # bắt đầu từ 1, sắp theo x1 tăng dần
    bbox: BBox
    score: float
    landmarks: Optional[np.ndarray] = None
    crop_box: Optional[BBox] = None             # khung vuông đưa vào U-Net (có thể vượt biên ảnh); None nếu không chạy U-Net
    mask_box: Optional[BBox] = None             # vị trí đặt mask_local vào ảnh (có thể vượt biên ảnh)
    mask_local: Optional[np.ndarray] = None     # (h, w) uint8 {0,1}, kích thước đúng bằng mask_box
    thumbnail: Optional[np.ndarray] = None      # ảnh crop nhỏ cho thẻ khuôn mặt
    status: str = "ok"                          # ok | fallback_ellipse
    reason: Optional[str] = None                # too_small | bad_mask
    blurred: bool = True
    # Chừa chỗ cho hướng phát triển (nhận diện danh tính). Hiện luôn là None.
    identity: Optional[str] = None
    identity_score: Optional[float] = None

    @property
    def size_px(self):
        return (self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass
class PipelineResult:
    image: np.ndarray                           # ảnh RGB đang xử lý (sau khi giới hạn kích thước)
    faces: list
    instance_map: np.ndarray                    # (H, W) uint16: 0 = nền, k = mặt có id k
    overlay_boxes: np.ndarray
    overlay_masks: np.ndarray
    result_image: np.ndarray                    # ảnh đã che (cập nhật sau apply_privacy)
    timing_ms: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

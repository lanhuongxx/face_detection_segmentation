"""Module 5: che các khuôn mặt được chọn (làm mờ / pixel hóa / tô màu).
Mặt KHÔNG được chọn giữ nguyên từng pixel."""
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.pipeline.mask_mapping import ellipse_mask, paste_mask


@dataclass
class PrivacyOptions:
    mode: str = "blur"            # blur | pixelate | solid
    strength: int = 6             # 1-10
    region: str = "mask"          # mask | bbox | ellipse
    dilate_px: int = 4            # nở mask thêm vài pixel để an toàn ở viền
    feather_px: int = 3           # làm mềm viền khi trộn với ảnh gốc
    solid_color: tuple = (30, 30, 30)


def face_region(face, region):
    """Trả về (mask uint8 cục bộ, khung đặt (x1,y1,x2,y2)) theo kiểu vùng che."""
    x1, y1, x2, y2 = face.bbox
    if region == "bbox":
        return np.ones((y2 - y1, x2 - x1), np.uint8), (x1, y1, x2, y2)
    if region == "ellipse":
        return ellipse_mask(face.bbox)
    return face.mask_local, face.mask_box            # "mask" (mặt nhỏ/mask hỏng đã được gán elip từ trước)


def covered_fraction(shape, faces, selected_ids, region):
    """Tỉ lệ diện tích ảnh bị che (chưa tính dilate) - dùng cho tab So sánh."""
    canvas = np.zeros(shape[:2], np.uint8)
    for face in faces:
        if face.id in selected_ids:
            local, box = face_region(face, region)
            if local is not None:
                paste_mask(canvas, local, box[0], box[1])
    return float(canvas.mean())


def _effect(roi, face_size, opts):
    """Tạo ảnh đã được áp hiệu ứng cho vùng roi. Cường độ tỉ lệ theo kích thước mặt
    (mặt nhỏ và mặt lớn đều bị che mạnh như nhau)."""
    strength = float(np.clip(opts.strength, 1, 10))
    if opts.mode == "solid":
        return np.full_like(roi, opts.solid_color)
    if opts.mode == "pixelate":
        cells = int(round(np.interp(strength, [1, 10], [14, 3])))   # số ô trên một mặt
        block = max(2, face_size // cells)
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    sigma = max(1.5, face_size * float(np.interp(strength, [1, 10], [0.04, 0.30])))
    return cv2.GaussianBlur(roi, (0, 0), sigma)


def apply_privacy(image, faces, selected_ids, opts):
    """image RGB uint8. Trả về ảnh mới đã che các mặt có id trong selected_ids."""
    out = image.copy()
    H, W = image.shape[:2]
    selected = set(selected_ids)
    pad = int(opts.dilate_px + 3 * opts.feather_px + 2)
    kernel = None
    if opts.dilate_px > 0:
        k = 2 * int(opts.dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for face in faces:
        if face.id not in selected:
            continue
        local, box = face_region(face, opts.region)
        if local is None:
            continue
        # Vùng làm việc (ROI): khung mask mở rộng một chút, cắt theo ảnh
        rx1, ry1 = max(0, box[0] - pad), max(0, box[1] - pad)
        rx2, ry2 = min(W, box[2] + pad), min(H, box[3] + pad)
        if rx2 <= rx1 or ry2 <= ry1:
            continue
        alpha = np.zeros((ry2 - ry1, rx2 - rx1), np.uint8)
        paste_mask(alpha, local, box[0] - rx1, box[1] - ry1)
        if kernel is not None:
            alpha = cv2.dilate(alpha, kernel)
        alpha = alpha.astype(np.float32)
        if opts.feather_px > 0:
            alpha = cv2.GaussianBlur(alpha, (0, 0), float(opts.feather_px))
        alpha = alpha[..., None]

        roi = out[ry1:ry2, rx1:rx2]
        size = max(face.bbox[2] - face.bbox[0], face.bbox[3] - face.bbox[1])
        effect = _effect(roi, size, opts)
        out[ry1:ry2, rx1:rx2] = np.clip(roi * (1 - alpha) + effect * alpha, 0, 255).astype(np.uint8)
    return out

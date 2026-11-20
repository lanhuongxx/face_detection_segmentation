"""Module 0: đọc ảnh, sửa hướng EXIF, về RGB, giới hạn kích thước."""
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class UnsupportedFormat(Exception):
    def __str__(self):
        return "Hệ thống hỗ trợ ảnh JPG, PNG, WEBP."


class CorruptImage(Exception):
    def __str__(self):
        return "Không đọc được ảnh (file hỏng hoặc không phải ảnh)."


class FileTooLarge(Exception):
    def __init__(self, limit_mb):
        self.limit_mb = limit_mb

    def __str__(self):
        return f"Ảnh vượt {self.limit_mb} MB. Hãy giảm kích thước rồi thử lại."


def load_image(source, max_file_mb=10):
    """source: đường dẫn, bytes hoặc file-like. Trả về (ảnh RGB uint8 (H,W,3), meta)."""
    if isinstance(source, (str, Path)):
        data = Path(source).read_bytes()
        name = Path(source).name
    elif isinstance(source, (bytes, bytearray)):
        data, name = bytes(source), None
    else:
        data, name = source.read(), getattr(source, "name", None)

    if len(data) > max_file_mb * 1024 * 1024:
        raise FileTooLarge(max_file_mb)
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt not in ALLOWED_FORMATS:
            raise UnsupportedFormat()
        img = ImageOps.exif_transpose(img).convert("RGB")   # sửa hướng xoay theo EXIF
    except UnsupportedFormat:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise CorruptImage() from error
    array = np.asarray(img)
    return array, {"filename": name, "orig_size": array.shape[:2]}


def limit_size(image, max_side):
    """Thu nhỏ (giữ tỉ lệ) nếu cạnh dài > max_side. Trả về (ảnh, scale), scale <= 1."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale

"""Đọc video, xuất PNG."""
import io

import cv2
import numpy as np
from PIL import Image


def video_info(path):
    """Trả về dict {fps, total_frames, width, height, duration_s} của file video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Không mở được video.")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def extract_frame(path, frame_idx):
    """Lấy một khung hình (RGB uint8) từ video theo chỉ số."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Không mở được video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Không đọc được khung hình {frame_idx}.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_png_bytes(image):
    """Mảng RGB uint8 -> bytes PNG (cho nút tải về)."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()

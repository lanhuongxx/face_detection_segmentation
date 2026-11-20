"""Module 2: cắt khung vuông quanh khuôn mặt, GIỐNG HỆT cách crop lúc train (notebook CELL 4)."""
import numpy as np
from PIL import Image


def square_crop_box(bbox, margin):
    """Khung vuông quanh tâm bbox, cạnh = max(w,h)*(1+2*margin). Có thể vượt biên ảnh."""
    left, top, right, bottom = bbox
    side = int(np.ceil(max(right - left, bottom - top) * (1 + 2 * margin)))
    x = int(np.floor((left + right - side) / 2))
    y = int(np.floor((top + bottom - side) / 2))
    return (x, y, x + side, y + side)


def crop_faces(image, boxes, size):
    """Cắt từng khung và resize về size x size (BILINEAR như lúc train).
    Phần vượt biên ảnh được pad ĐEN (PIL tự làm), đúng như dữ liệu train."""
    pil = Image.fromarray(image)
    return [np.asarray(pil.crop(tuple(int(v) for v in box))
                       .resize((size, size), Image.Resampling.BILINEAR)) for box in boxes]

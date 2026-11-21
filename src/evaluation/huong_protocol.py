"""Giao thức dữ liệu của Hương (notebook celebamask_unet_v3.ipynb), viết lại để Yến đo cùng chuẩn.

Gồm: chia tập, định nghĩa mask "mặt" (9 nhãn), làm sạch nhãn, cắt khung vuông 256.
Mọi hằng số khớp CELL 1-4 của notebook. Nếu Hương đổi notebook thì sửa ở đây.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

SEED = 42
MAX_IMAGES = 6000
SIZE = 256
MARGIN = 0.25
LABEL_OPEN_KERNEL = 31
# Chín nhãn mặt của Hương: KHÔNG có tai (l_ear/r_ear) và KHÔNG có ear_r (= khuyên tai).
FACE_PARTS = ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip")


def huong_split(ids: Sequence[int], max_images: int = MAX_IMAGES, seed: int = SEED
                ) -> dict[str, list[int]]:
    """Tái tạo đúng chia tập của Hương.

    Notebook: ``samples`` = các ảnh CÓ nhãn da, sắp theo id tăng dần, ``random.Random(42)
    .shuffle``, lấy ``MAX_IMAGES`` ảnh đầu, rồi holdout = n // 10;
    train = [:-2*holdout], val = [-2*holdout:-holdout], test = [-holdout:].
    ``ids`` phải là đúng danh sách id có nhãn da (không phải mặc định 0..29999 nếu có ảnh thiếu da).
    """
    samples = sorted(int(i) for i in ids)
    random.Random(seed).shuffle(samples)
    samples = samples[:max_images]
    holdout = max(1, len(samples) // 10)
    return {
        "train": samples[:-2 * holdout],
        "val": samples[-2 * holdout:-holdout],
        "test": samples[-holdout:],
    }


def ids_with_skin(mask_dir: Path) -> list[int]:
    """Các id có file ``<id>_skin.png`` (điều kiện lọc của Hương)."""
    found = set()
    for path in Path(mask_dir).rglob("*_skin.png"):
        head = path.stem.partition("_")[0]
        if head.isdecimal():
            found.add(int(head))
    return sorted(found)


def remove_thin_parts(mask: np.ndarray, kernel_size: int = LABEL_OPEN_KERNEL) -> np.ndarray:
    """Giống hệt ``remove_thin_parts`` của Hương: opening, giữ mảnh lớn nhất."""
    m = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    if not opened.any():
        opened = m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n > 1:
        opened = (labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    return opened.astype(bool)


def _load_part(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        raw = source.convert("L") if not source.mode.startswith("I") else \
            Image.fromarray(np.clip(np.rint(np.asarray(source) / 257.0), 0, 255).astype(np.uint8))
        return np.asarray(raw.resize(size, Image.Resampling.NEAREST)) > 0


def clean_face_mask(mask_dir: Path, image_id: int, size: tuple[int, int]) -> np.ndarray:
    """Mask mặt cỡ ảnh gốc ``size=(W,H)``, đúng target lúc train của Hương.

    Hợp 9 nhãn mặt (nearest), trừ tóc (nếu kết quả còn pixel), rồi ``remove_thin_parts``.
    CelebAMask-HQ lưu nhãn tên ``%05d_<part>.png`` (có thể nằm trong thư mục con).
    """
    mask_dir = Path(mask_dir)

    def find(part: str) -> Path | None:
        name = f"{image_id:05d}_{part}.png"
        for candidate in (mask_dir / name, mask_dir / str(image_id // 2000) / name):
            if candidate.is_file():
                return candidate
        hits = list(mask_dir.rglob(name))
        return hits[0] if hits else None

    mask = np.zeros((size[1], size[0]), dtype=bool)
    for part in FACE_PARTS:
        path = find(part)
        if path is not None:
            mask |= _load_part(path, size)
    hair_path = find("hair")
    if hair_path is not None:
        without_hair = mask & ~_load_part(hair_path, size)
        if without_hair.any():
            mask = without_hair
    return remove_thin_parts(mask)


def square_box(box: Sequence[float], margin: float = MARGIN) -> tuple[int, int, int, int]:
    """Khung vuông như ``src.pipeline.crop.square_crop_box`` (cạnh = max(w,h)*(1+2m))."""
    left, top, right, bottom = box
    side = int(np.ceil(max(right - left, bottom - top) * (1 + 2 * margin)))
    x = int(np.floor((left + right - side) / 2))
    y = int(np.floor((top + bottom - side) / 2))
    return x, y, x + side, y + side


def crop_mask(mask: np.ndarray, box: Sequence[float], size: int = SIZE,
              margin: float = MARGIN) -> np.ndarray:
    """Cắt mask theo khung vuông (pad 0 ngoài biên) rồi resize NEAREST về size x size."""
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    crop = image.crop(square_box(box, margin)).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(crop) > 0

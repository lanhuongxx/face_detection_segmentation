"""Cách chia dữ liệu và làm sạch mask của Hương (notebook celebamask_unet_v3.ipynb).

Viết lại ở đây để Yến đo trên đúng tập và đúng loại mask mà Hương đã train. Gồm bốn việc:
chia train/val/test, chọn 9 nhãn thuộc "mặt", làm sạch mask, và cắt khung vuông 256.
Các hằng số bên dưới lấy theo CELL 1 đến 4 của notebook, nên nếu Hương đổi notebook
thì phải sửa lại ở file này cho khớp.
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
# 9 nhãn mặt của Hương. Lưu ý là không có tai (l_ear, r_ear) và cũng không có ear_r
# (ear_r trong CelebAMask-HQ là khuyên tai, không phải tai)
FACE_PARTS = ("skin", "nose", "l_eye", "r_eye", "l_brow", "r_brow", "mouth", "u_lip", "l_lip")


def huong_split(ids: Sequence[int], max_images: int = MAX_IMAGES, seed: int = SEED
                ) -> dict[str, list[int]]:
    """Chia tập train/val/test giống hệt notebook của Hương.

    Các bước trong notebook: lấy những ảnh có nhãn da, sắp theo id tăng dần, xáo bằng
    ``random.Random(42)``, giữ lại ``max_images`` ảnh đầu, rồi lấy holdout = n // 10.
    Train là phần đầu, val là holdout ảnh kế cuối, test là holdout ảnh cuối.
    ``ids`` phải đúng là danh sách id có nhãn da. Nếu có ảnh thiếu da thì không được
    dùng mặc định 0..29999, vì như vậy thứ tự xáo sẽ lệch so với Hương.
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
    """Lấy id của những ảnh có file ``<id>_skin.png`` (Hương chỉ dùng các ảnh này)."""
    found = set()
    for path in Path(mask_dir).rglob("*_skin.png"):
        # tên file dạng 00012_skin.png, phần trước dấu "_" đầu tiên là id
        head = path.stem.partition("_")[0]
        if head.isdecimal():
            found.add(int(head))
    return sorted(found)


def remove_thin_parts(mask: np.ndarray, kernel_size: int = LABEL_OPEN_KERNEL) -> np.ndarray:
    """Làm sạch mask giống hàm ``remove_thin_parts`` của Hương.

    Mở (opening) bằng kernel elip để cắt các phần mảnh, sau đó chỉ giữ mảnh liên thông lớn nhất.
    """
    m = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    # nếu opening xóa sạch mask (mặt nhỏ quá) thì quay về mask ban đầu
    if not opened.any():
        opened = m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if n > 1:
        # nhãn 0 là nền nên bỏ qua; stats[1:] là các mảnh còn lại, lấy mảnh có diện tích lớn nhất
        opened = (labels == 1 + stats[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
    return opened.astype(bool)


def _load_part(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Đọc một file nhãn thành mask bool, resize về ``size`` (W, H) bằng nearest."""
    with Image.open(path) as source:
        # file 16-bit (mode bắt đầu bằng "I") thì chia 257 để về 0..255 như bên evaluate.py,
        # file thường thì đổi sang xám
        raw = source.convert("L") if not source.mode.startswith("I") else \
            Image.fromarray(np.clip(np.rint(np.asarray(source) / 257.0), 0, 255).astype(np.uint8))
        return np.asarray(raw.resize(size, Image.Resampling.NEAREST)) > 0


def clean_face_mask(mask_dir: Path, image_id: int, size: tuple[int, int]) -> np.ndarray:
    """Tạo mask mặt cỡ ảnh gốc ``size=(W, H)``, đúng loại target mà Hương dùng để train.

    Cách làm: hợp 9 nhãn mặt lại (resize nearest), trừ phần tóc nếu trừ xong vẫn còn pixel,
    rồi cho qua ``remove_thin_parts``. File nhãn của CelebAMask-HQ tên là
    ``%05d_<part>.png`` và có thể nằm trong thư mục con.
    """
    mask_dir = Path(mask_dir)

    def find(part: str) -> Path | None:
        name = f"{image_id:05d}_{part}.png"
        # thử hai chỗ hay gặp trước (ngay trong thư mục, hoặc thư mục con đánh số
        # theo id // 2000), không có mới quét cả cây thư mục vì rglob khá chậm
        for candidate in (mask_dir / name, mask_dir / str(image_id // 2000) / name):
            if candidate.is_file():
                return candidate
        hits = list(mask_dir.rglob(name))
        return hits[0] if hits else None

    mask = np.zeros((size[1], size[0]), dtype=bool)
    for part in FACE_PARTS:
        path = find(part)
        # ảnh nào không có nhãn bộ phận đó (ví dụ không thấy một bên lông mày) thì bỏ qua
        if path is not None:
            mask |= _load_part(path, size)
    hair_path = find("hair")
    if hair_path is not None:
        without_hair = mask & ~_load_part(hair_path, size)
        # chỉ trừ tóc khi trừ xong vẫn còn pixel, tránh làm mask rỗng
        if without_hair.any():
            mask = without_hair
    return remove_thin_parts(mask)


def square_box(box: Sequence[float], margin: float = MARGIN) -> tuple[int, int, int, int]:
    """Tạo khung vuông quanh bbox, giống ``src.pipeline.crop.square_crop_box``.

    Cạnh của khung = cạnh dài nhất của bbox * (1 + 2 * margin), tâm giữ nguyên.
    """
    left, top, right, bottom = box
    side = int(np.ceil(max(right - left, bottom - top) * (1 + 2 * margin)))
    x = int(np.floor((left + right - side) / 2))
    y = int(np.floor((top + bottom - side) / 2))
    return x, y, x + side, y + side


def crop_mask(mask: np.ndarray, box: Sequence[float], size: int = SIZE,
              margin: float = MARGIN) -> np.ndarray:
    """Cắt mask theo khung vuông rồi resize nearest về size x size.

    Phần khung nằm ngoài ảnh được pad bằng 0 (nền), giống lúc pipeline cắt ảnh thật.
    """
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    crop = image.crop(square_box(box, margin)).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(crop) > 0

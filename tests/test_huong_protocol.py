"""Test cách chia dữ liệu và làm sạch mask của Hương (huong_protocol)."""

import numpy as np
from PIL import Image

from src.evaluation.huong_protocol import (clean_face_mask, crop_mask, huong_split, ids_with_skin,
                                           remove_thin_parts, square_box)


def test_split_sizes_and_disjoint_like_notebook():
    """Chia 30000 id phải ra 4800/600/600 như notebook, ba tập không trùng nhau và lần nào cũng như nhau."""
    s = huong_split(range(30000))
    assert [len(s[k]) for k in ("train", "val", "test")] == [4800, 600, 600]
    # train/val không được lẫn vào test, nếu không thì số đo trên test không còn độc lập
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    assert s == huong_split(range(30000))          # tái lập được


def _write(path, arr):
    """Lưu mask bool thành ảnh PNG 0/255, tự tạo thư mục cha nếu chưa có."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


def test_clean_face_mask_subtracts_hair_and_ignores_ears(tmp_path):
    """Mask sạch phải trừ phần tóc ra và không tính tai, đồng thời ids_with_skin tìm đúng id."""
    z = np.zeros((64, 64), bool)
    skin, hair, ear = z.copy(), z.copy(), z.copy()
    skin[8:56, 8:56] = True
    hair[8:20, 8:56] = True   # tóc phủ lên phần trán, nằm chồng trong vùng da
    ear[30:34, 0:6] = True    # tai nằm ngoài vùng da
    _write(tmp_path / "00007_skin.png", skin)
    _write(tmp_path / "00007_hair.png", hair)
    _write(tmp_path / "00007_l_ear.png", ear)
    clean = clean_face_mask(tmp_path, 7, (64, 64))
    assert not clean[32, 2] and not clean[10, 30]     # tai không tính, tóc bị trừ
    assert clean[40, 32]                              # phần da còn lại vẫn giữ
    assert ids_with_skin(tmp_path) == [7]

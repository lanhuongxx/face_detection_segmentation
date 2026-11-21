import numpy as np
from PIL import Image

from src.evaluation.huong_protocol import (clean_face_mask, crop_mask, huong_split, ids_with_skin,
                                           remove_thin_parts, square_box)


def test_split_sizes_and_disjoint_like_notebook():
    s = huong_split(range(30000))
    assert [len(s[k]) for k in ("train", "val", "test")] == [4800, 600, 600]
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["val"]) & set(s["test"]))
    assert s == huong_split(range(30000))          # tái lập được


def _write(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8) * 255).save(path)


def test_clean_face_mask_subtracts_hair_and_ignores_ears(tmp_path):
    z = np.zeros((64, 64), bool)
    skin, hair, ear = z.copy(), z.copy(), z.copy()
    skin[8:56, 8:56] = True
    hair[8:20, 8:56] = True
    ear[30:34, 0:6] = True
    _write(tmp_path / "00007_skin.png", skin)
    _write(tmp_path / "00007_hair.png", hair)
    _write(tmp_path / "00007_l_ear.png", ear)
    clean = clean_face_mask(tmp_path, 7, (64, 64))
    assert not clean[32, 2] and not clean[10, 30]     # tai không tính, tóc bị trừ
    assert clean[40, 32]
    assert ids_with_skin(tmp_path) == [7]

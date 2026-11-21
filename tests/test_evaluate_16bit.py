"""PNG xác suất 16-bit không được bị cắt về 255 khi đọc mask."""

import numpy as np
from PIL import Image

from src.evaluation.evaluate import _load_mask
from src.evaluation.seg_metrics import dice_score


def _gt():
    gt = np.zeros((8, 8), np.uint8)
    gt[:4] = 255
    return gt


def test_16bit_probability_png_is_scaled_not_saturated(tmp_path):
    gt = _gt()
    prob = np.where(gt > 0, 0.9, 0.1)
    Image.fromarray((prob * 65535).astype(np.uint16)).save(tmp_path / "p.png")
    loaded = _load_mask(tmp_path / "p.png")
    assert float(dice_score(loaded, gt)) == 1.0
    assert loaded.max() > 200 and loaded.min() < 50


def test_16bit_binary_png_full_range(tmp_path):
    gt = _gt()
    Image.fromarray((gt > 0).astype(np.uint16) * 65535).save(tmp_path / "b.png")
    assert float(dice_score(_load_mask(tmp_path / "b.png"), gt)) == 1.0


def test_8bit_png_unchanged(tmp_path):
    gt = _gt()
    Image.fromarray(gt).save(tmp_path / "g.png")
    assert (_load_mask(tmp_path / "g.png") == gt).all()

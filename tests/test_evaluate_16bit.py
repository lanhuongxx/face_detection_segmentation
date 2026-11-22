"""PNG xác suất 16-bit không được bị cắt về 255 khi đọc mask.

Lỗi này từng làm Dice sai mà không báo gì: Pillow convert("L") với ảnh 16-bit sẽ cắt mọi
giá trị lớn về 255, nên mọi pixel đều thành foreground.
"""

import numpy as np
from PIL import Image

from src.evaluation.evaluate import _load_mask
from src.evaluation.seg_metrics import dice_score


def _gt():
    """GT giả 8x8: nửa trên (4 hàng đầu) là foreground (255), nửa dưới là nền."""
    gt = np.zeros((8, 8), np.uint8)
    gt[:4] = 255
    return gt


def test_16bit_probability_png_is_scaled_not_saturated(tmp_path):
    """Bản đồ xác suất 16-bit (0.9 ở vùng mặt, 0.1 ở nền) phải được đưa về thang 0..255, không bị cắt."""
    gt = _gt()
    prob = np.where(gt > 0, 0.9, 0.1)
    # lưu xác suất thành PNG 16-bit: nhân với 65535 (dải đầy đủ của uint16)
    Image.fromarray((prob * 65535).astype(np.uint16)).save(tmp_path / "p.png")
    loaded = _load_mask(tmp_path / "p.png")
    assert float(dice_score(loaded, gt)) == 1.0
    # nếu bị cắt về 255 thì min cũng là 255; đọc đúng thì vùng nền phải còn giá trị thấp
    assert loaded.max() > 200 and loaded.min() < 50


def test_16bit_binary_png_full_range(tmp_path):
    """Mask nhị phân lưu 16-bit với giá trị 0 và 65535 vẫn phải ra Dice = 1."""
    gt = _gt()
    Image.fromarray((gt > 0).astype(np.uint16) * 65535).save(tmp_path / "b.png")
    assert float(dice_score(_load_mask(tmp_path / "b.png"), gt)) == 1.0


def test_8bit_png_unchanged(tmp_path):
    """PNG 8-bit bình thường thì đọc ra y nguyên, không bị đổi giá trị."""
    gt = _gt()
    Image.fromarray(gt).save(tmp_path / "g.png")
    assert (_load_mask(tmp_path / "g.png") == gt).all()


import io

import numpy as np
import pytest
from PIL import Image

from src.pipeline.preprocess_input import (
    load_image,
    limit_size,
    CorruptImage,
    UnsupportedFormat,
    FileTooLarge,
)


def tao_anh_bytes(format="PNG", mode="RGB"):
    """Tạo ảnh thử nghiệm ngay trong bộ nhớ."""
    color = (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)
    image = Image.new(mode, (40, 30), color)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()


def test_doc_anh_jpeg():
    """TC25: Đọc JPEG và chuyển thành RGB."""
    data = tao_anh_bytes(format="JPEG")

    image, meta = load_image(data)

    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8
    assert meta["orig_size"] == (30, 40)


def test_anh_png_co_alpha():
    """TC26: PNG có alpha được chuyển về RGB."""
    data = tao_anh_bytes(format="PNG", mode="RGBA")

    image, meta = load_image(data)

    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8
    assert meta["orig_size"] == (30, 40)


def test_anh_bi_hong():
    """TC27: Dữ liệu không phải ảnh phải bị từ chối."""
    data = b"Day khong phai la file anh hop le"

    with pytest.raises(CorruptImage):
        load_image(data)


def test_dinh_dang_khong_ho_tro():
    """TC28: GIF không nằm trong định dạng được hỗ trợ."""
    data = tao_anh_bytes(format="GIF")

    with pytest.raises(UnsupportedFormat):
        load_image(data)


def test_file_vuot_dung_luong():
    """TC29: File vượt giới hạn phải bị từ chối."""
    data = tao_anh_bytes(format="PNG")

    # Đặt giới hạn rất nhỏ cho dữ liệu thử nghiệm.
    with pytest.raises(FileTooLarge):
        load_image(data, max_file_mb=0.000001)


def test_gioi_han_do_phan_giai():
    """TC30: Ảnh lớn được thu nhỏ, ảnh nhỏ giữ nguyên."""
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)

    resized, scale = limit_size(image, max_side=500)

    assert resized.shape == (250, 500, 3)
    assert abs(scale - 0.25) < 1e-6

    small = np.zeros((100, 200, 3), dtype=np.uint8)
    unchanged, small_scale = limit_size(small, max_side=500)

    assert unchanged.shape == small.shape
    assert small_scale == 1.0
    assert np.array_equal(unchanged, small)

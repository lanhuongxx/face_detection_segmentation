
import numpy as np

from src.pipeline.schemas import Face
from src.pipeline.privacy_filter import (
    PrivacyOptions,
    face_region,
    covered_fraction,
    apply_privacy,
)


def tao_face():
    """Tạo khuôn mặt giả để kiểm thử mà không cần U-Net."""
    return Face(
        id=1,
        bbox=(5, 5, 15, 15),
        score=0.99,
        mask_local=np.ones((10, 10), dtype=np.uint8),
        mask_box=(5, 5, 15, 15),
    )


def test_privacy_khong_chon_mat():
    """Không chọn khuôn mặt thì ảnh giữ nguyên."""
    image = np.full((20, 20, 3), 150, dtype=np.uint8)
    face = tao_face()

    opts = PrivacyOptions(
        mode="solid",
        region="mask",
        dilate_px=0,
        feather_px=0,
    )

    result = apply_privacy(image, [face], [], opts)

    assert np.array_equal(result, image)


def test_privacy_che_dung_vung():
    """Chỉ che khuôn mặt được chọn."""
    image = np.full((20, 20, 3), 150, dtype=np.uint8)
    face = tao_face()

    opts = PrivacyOptions(
        mode="solid",
        region="mask",
        solid_color=(30, 30, 30),
        dilate_px=0,
        feather_px=0,
    )

    result = apply_privacy(image, [face], [1], opts)

    assert np.all(result[5:15, 5:15] == 30)
    assert np.all(result[0:5, :, :] == 150)
    assert np.all(result[15:, :, :] == 150)


def test_privacy_mask_rong():
    """Mask rỗng không làm thay đổi ảnh."""
    image = np.full((20, 20, 3), 150, dtype=np.uint8)
    face = tao_face()
    face.mask_local = np.zeros((10, 10), dtype=np.uint8)

    opts = PrivacyOptions(
        mode="solid",
        region="mask",
        dilate_px=0,
        feather_px=0,
    )

    result = apply_privacy(image, [face], [1], opts)

    assert np.array_equal(result, image)


def test_privacy_ty_le_che():
    """Kiểm tra tỷ lệ diện tích được che."""
    face = tao_face()

    fraction = covered_fraction(
        (20, 20, 3),
        [face],
        [1],
        "bbox",
    )

    # Vùng che 10x10 trên ảnh 20x20 = 25%
    assert abs(fraction - 0.25) < 1e-6

def tao_anh_nhieu_chi_tiet():
    """Ảnh giả lập có nhiều chi tiết để kiểm tra blur và pixelate."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)


def tao_face_lon():
    return Face(
        id=1,
        bbox=(10, 10, 30, 30),
        score=0.99,
        mask_local=np.ones((20, 20), dtype=np.uint8),
        mask_box=(10, 10, 30, 30),
    )


def test_blur_lam_thay_doi_khuon_mat():
    """Blur thay đổi điểm ảnh bên trong vùng mặt."""
    image = tao_anh_nhieu_chi_tiet()
    face = tao_face_lon()
    opts = PrivacyOptions(
        mode="blur", region="mask",
        strength=6, dilate_px=0, feather_px=0
    )

    result = apply_privacy(image, [face], [1], opts)

    assert not np.array_equal(result[12:28, 12:28],
                              image[12:28, 12:28])


def test_blur_giu_nguyen_nen():
    """Blur không làm thay đổi vùng nền ngoài mask."""
    image = tao_anh_nhieu_chi_tiet()
    face = tao_face_lon()
    opts = PrivacyOptions(
        mode="blur", region="mask",
        strength=6, dilate_px=0, feather_px=0
    )

    result = apply_privacy(image, [face], [1], opts)

    assert np.array_equal(result[:8], image[:8])
    assert np.array_equal(result[32:], image[32:])


def test_pixelate_lam_thay_doi_khuon_mat():
    """Pixelate làm thay đổi vùng khuôn mặt."""
    image = tao_anh_nhieu_chi_tiet()
    face = tao_face_lon()
    opts = PrivacyOptions(
        mode="pixelate", region="mask",
        strength=8, dilate_px=0, feather_px=0
    )

    result = apply_privacy(image, [face], [1], opts)

    assert not np.array_equal(result[12:28, 12:28],
                              image[12:28, 12:28])


def test_pixelate_giu_nguyen_nen():
    """Pixelate không làm thay đổi nền ngoài mask."""
    image = tao_anh_nhieu_chi_tiet()
    face = tao_face_lon()
    opts = PrivacyOptions(
        mode="pixelate", region="mask",
        strength=8, dilate_px=0, feather_px=0
    )

    result = apply_privacy(image, [face], [1], opts)

    assert np.array_equal(result[:8], image[:8])
    assert np.array_equal(result[32:], image[32:])

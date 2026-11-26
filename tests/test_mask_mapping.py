
import numpy as np
from types import SimpleNamespace

from src.pipeline.mask_mapping import (
    paste_mask,
    prob_to_local_mask,
    build_instance_map,
)


def test_mask_vuot_bien():
    """Mask vượt biên trái/trên phải được cắt đúng."""
    output = np.zeros((10, 10), dtype=np.uint8)
    mask = np.ones((5, 5), dtype=np.uint8)

    paste_mask(output, mask, -2, -2)

    assert output.sum() == 9


def test_mask_nam_ngoai_anh():
    """Mask hoàn toàn ngoài ảnh không làm thay đổi ảnh."""
    output = np.zeros((10, 10), dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.uint8)

    paste_mask(output, mask, 20, 20)

    assert output.sum() == 0


def test_mask_resize():
    """Mask được resize về đúng kích thước crop."""
    prob = np.ones((256, 256), dtype=np.float32)

    mask = prob_to_local_mask(
        prob,
        (10, 20, 110, 170),
        0.5
    )

    assert mask.shape == (150, 100)
    assert mask.sum() == 15000


def test_mask_chong_lan():
    """Vùng chồng lấn ưu tiên mặt có confidence cao."""
    face1 = SimpleNamespace(
        id=1,
        score=0.6,
        mask_local=np.ones((5, 5), dtype=np.uint8),
        mask_box=(0, 0, 5, 5)
    )

    face2 = SimpleNamespace(
        id=2,
        score=0.9,
        mask_local=np.ones((5, 5), dtype=np.uint8),
        mask_box=(2, 2, 7, 7)
    )

    result = build_instance_map((10, 10), [face1, face2])

    assert result[0, 0] == 1
    assert result[3, 3] == 2
    assert result[9, 9] == 0

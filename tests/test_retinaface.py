"""Test RetinaFaceDetector bằng backend giả, không cần tải weights thật."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest
import torch

from src.detection.retinaface import RetinaFaceDetector, bbox_to_int, main


class FakeModel:
    """Model giả: ghi lại tham số được gọi và trả kết quả do từng test tự định nghĩa."""

    def __init__(self, output_factory: Any) -> None:
        self.output_factory = output_factory
        # các thuộc tính này để test kiểm tra detector có truyền đúng ảnh và ngưỡng xuống model không
        self.last_image: np.ndarray | None = None
        self.last_conf: float | None = None
        self.last_nms: float | None = None
        self.last_use_origin_size: bool | None = None

    def detect_faces(
        self,
        image: np.ndarray,
        conf_threshold: float,
        nms_threshold: float,
        use_origin_size: bool,
    ) -> np.ndarray:
        """Lưu lại các tham số gọi, rồi trả detections do output_factory tạo ra từ ảnh."""
        self.last_image = image
        self.last_conf = conf_threshold
        self.last_nms = nms_threshold
        self.last_use_origin_size = use_origin_size
        return self.output_factory(image)


def make_detector(
    model: FakeModel, max_size: int | None = None
) -> RetinaFaceDetector:
    """Tạo detector gắn model giả. Bỏ qua __init__ để khỏi tải weights, nhờ vậy test chạy nhanh."""
    detector = RetinaFaceDetector.__new__(RetinaFaceDetector)
    detector._model = model
    detector.conf_thres = 0.5
    detector.nms_thres = 0.4
    detector.max_size = max_size
    detector.model_name = "resnet50"
    detector.device = "cpu"
    detector._torch = torch
    return detector


def detection(
    bbox: tuple[float, float, float, float],
    score: float = 0.9,
) -> list[float]:
    """Tạo một hàng output giả gồm 15 số: bbox (4), score (1) và 5 landmarks (10)."""
    x1, y1, x2, y2 = bbox
    # đặt các điểm landmark theo tỉ lệ trong bbox (mắt trái, mắt phải, mũi, hai khóe miệng)
    # để khi ảnh bị resize thì vị trí của chúng vẫn dễ kiểm tra
    landmarks = [
        x1 + (x2 - x1) * 0.3,
        y1 + (y2 - y1) * 0.35,
        x1 + (x2 - x1) * 0.7,
        y1 + (y2 - y1) * 0.35,
        x1 + (x2 - x1) * 0.5,
        y1 + (y2 - y1) * 0.55,
        x1 + (x2 - x1) * 0.35,
        y1 + (y2 - y1) * 0.75,
        x1 + (x2 - x1) * 0.65,
        y1 + (y2 - y1) * 0.75,
    ]
    return [x1, y1, x2, y2, score, *landmarks]


def test_white_image_returns_empty_list() -> None:
    """Model không tìm ra mặt nào thì detect trả danh sách rỗng, không báo lỗi."""
    detector = make_detector(FakeModel(lambda _image: np.empty((0, 15))))
    faces = detector.detect(np.full((80, 100, 3), 255, dtype=np.uint8))
    assert faces == []


def test_grayscale_input_is_converted_to_bgr() -> None:
    """Ảnh xám phải thành ảnh 3 kênh giống nhau trước khi đưa xuống model."""
    model = FakeModel(lambda _image: np.empty((0, 15)))
    detector = make_detector(model)
    detector.detect(np.full((40, 50), 127, dtype=np.uint8))
    assert model.last_image is not None
    assert model.last_image.shape == (40, 50, 3)
    assert np.all(model.last_image == 127)


def test_float_grayscale_input_is_normalized_to_uint8() -> None:
    """Ảnh float trong [0, 1] phải được đổi sang uint8 [0, 255] trước khi chạy model."""
    model = FakeModel(lambda _image: np.empty((0, 15)))
    make_detector(model).detect(np.full((40, 50), 0.5, dtype=np.float64))
    assert model.last_image is not None
    assert model.last_image.dtype == np.uint8
    # 0.5 * 255 = 127.5 và astype cắt phần thập phân nên ra 127
    assert np.all(model.last_image == 127)


def test_rgba_input_drops_alpha_without_swapping_bgr() -> None:
    """Ảnh 4 kênh chỉ bỏ kênh alpha, ba kênh màu vẫn giữ nguyên thứ tự BGR."""
    rgba = np.zeros((40, 50, 4), dtype=np.uint8)
    rgba[:, :, 0] = 10
    rgba[:, :, 1] = 20
    rgba[:, :, 2] = 30
    rgba[:, :, 3] = 255
    model = FakeModel(lambda _image: np.empty((0, 15)))
    make_detector(model).detect(rgba)
    assert model.last_image is not None
    assert model.last_image[0, 0].tolist() == [10, 20, 30]


def test_single_face_contract_and_thresholds() -> None:
    """Một mặt phải ra đúng dạng {bbox, score, landmarks}, và ngưỡng đã đặt phải được truyền xuống model."""
    model = FakeModel(
        lambda _image: np.asarray(
            [detection((10, 12, 40, 45), score=0.84)], dtype=np.float32
        )
    )
    detector = make_detector(model)
    # đổi ngưỡng khác mặc định để chắc chắn giá trị kiểm tra là giá trị vừa đặt
    detector.conf_thres = 0.6
    detector.nms_thres = 0.3
    faces = detector.detect(np.zeros((80, 100, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert faces[0]["bbox"] == [10.0, 12.0, 40.0, 45.0]
    assert 0.0 <= faces[0]["score"] <= 1.0
    assert len(faces[0]["landmarks"]) == 5
    assert model.last_conf == 0.6
    assert model.last_nms == 0.3
    assert model.last_use_origin_size is True


def test_multiple_faces_and_batch_preserve_order() -> None:
    """Nhiều mặt trong một ảnh được xếp theo score giảm dần, còn detect_batch giữ đúng thứ tự ảnh."""
    # cố ý đưa mặt có score thấp (0.75) lên trước để kiểm tra việc sắp xếp lại
    rows = np.asarray(
        [
            detection((60, 10, 90, 45), 0.75),
            detection((5, 6, 25, 30), 0.95),
        ],
        dtype=np.float32,
    )
    detector = make_detector(FakeModel(lambda _image: rows))
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    faces = detector.detect(image)
    batch = detector.detect_batch([image, image])
    assert len(faces) == 2
    assert len(batch) == 2
    # hai ảnh giống nhau thì kết quả hai phần tử của batch phải giống nhau
    assert batch[0] == batch[1]
    assert np.allclose([face["score"] for face in faces], [0.95, 0.75])


def test_bbox_and_landmarks_are_clipped_and_score_is_bounded() -> None:
    """Tọa độ lọt ra ngoài ảnh được kẹp lại, và score luôn nằm trong [0, 1]."""
    # bbox lấn ra ngoài ảnh 100x100 ở cả bốn phía, score = 1.2 (vượt quá 1)
    row = detection((-5, -4, 120, 110), score=1.2)
    # landmarks cũng cố ý đặt lệch ra ngoài ảnh
    row[5:] = [-2, -3, 110, -3, 50, 50, -1, 120, 105, 120]
    detector = make_detector(FakeModel(lambda _image: np.asarray([row])))
    faces = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    face = faces[0]
    assert face["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert face["score"] == 1.0
    assert all(0 <= x <= 99 and 0 <= y <= 99 for x, y in face["landmarks"])
    x1, y1, x2, y2 = face["bbox"]
    assert 0 <= x1 < x2 <= 100
    assert 0 <= y1 < y2 <= 100


def test_bbox_to_int_rounds_and_clamps_exclusive_edges() -> None:
    """Góc trên trái làm tròn xuống, góc dưới phải làm tròn lên, rồi kẹp trong khung ảnh."""
    assert bbox_to_int((1.2, 2.8, 8.1, 9.01), (10, 10, 3)) == (1, 2, 9, 10)
    assert bbox_to_int((-2.0, -1.0, 12.0, 15.0), (10, 10, 3)) == (
        0,
        0,
        10,
        10,
    )


def test_bbox_to_int_returns_none_for_empty_crop() -> None:
    """Bbox bị suy biến (chiều rộng bằng 0) hoặc nằm hẳn ngoài ảnh thì không cắt được, trả None."""
    assert bbox_to_int((2.0, 3.0, 2.0, 8.0), (10, 10, 3)) is None
    assert bbox_to_int((12.0, 2.0, 14.0, 4.0), (10, 10, 3)) is None


def test_detection_with_empty_integer_crop_is_discarded() -> None:
    """detect phải bỏ những bbox không cắt được ảnh sau khi làm tròn và kẹp vào khung."""
    row = detection((12.0, 2.0, 12.0, 4.0))
    detector = make_detector(FakeModel(lambda _image: np.asarray([row])))
    assert detector.detect(np.zeros((40, 40, 3), dtype=np.uint8)) == []


def test_clipped_bbox_has_positive_area_and_landmarks_inside_image() -> None:
    """Sau khi kẹp, bbox phải còn diện tích dương và các landmark nằm trong ảnh."""
    row = detection((-2, -3, 100, 100), score=0.8)
    row[5:] = [-3, -2, 105, -2, 50, 50, -1, 101, 105, 101]
    faces = make_detector(
        FakeModel(lambda _image: np.asarray([row], dtype=np.float32))
    ).detect(np.zeros((40, 60, 3), dtype=np.uint8))

    assert len(faces) == 1
    x1, y1, x2, y2 = faces[0]["bbox"]
    assert 0 <= x1 < x2 <= 60
    assert 0 <= y1 < y2 <= 40
    assert all(0 <= x <= 59 and 0 <= y <= 39 for x, y in faces[0]["landmarks"])


def test_draw_returns_annotated_copy() -> None:
    """draw trả về một bản sao đã vẽ, còn ảnh gốc phải giữ nguyên."""
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    faces = [
        {
            "bbox": [10.0, 10.0, 40.0, 45.0],
            "score": 0.9,
            "landmarks": [[15.0, 20.0]] * 5,
        }
    ]
    rendered = RetinaFaceDetector.draw(image, faces)
    assert rendered.shape == image.shape
    # bản vẽ phải khác ảnh đen ban đầu, còn ảnh gốc vẫn toàn pixel 0
    assert not np.array_equal(rendered, image)
    assert np.count_nonzero(image) == 0


def test_large_and_small_scales_map_to_original_coordinates() -> None:
    """Ảnh lớn bị thu nhỏ khi chạy model, nhưng bbox quy về ảnh gốc vẫn phải cùng tỉ lệ."""
    def proportional_face(image: np.ndarray) -> np.ndarray:
        # model giả luôn trả một mặt chiếm từ 20% đến 80% ảnh mà nó nhận được
        height, width = image.shape[:2]
        return np.asarray(
            [detection((0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height))]
        )

    # max_size=100 nên ảnh 200x200 sẽ bị thu về 100x100 trước khi vào model
    detector = make_detector(FakeModel(proportional_face), max_size=100)
    small = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))[0]["bbox"]
    large = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))[0]["bbox"]
    # chia cho kích thước ảnh gốc thì hai bbox phải trùng tỉ lệ
    assert np.allclose(np.asarray(small) / 100, np.asarray(large) / 200)


def test_tiny_image_is_upscaled_before_inference() -> None:
    """Ảnh thumbnail quá nhỏ được phóng to để cạnh ngắn đạt mức tối thiểu của backbone."""
    model = FakeModel(lambda _image: np.empty((0, 15)))
    make_detector(model).detect(np.zeros((2, 3, 3), dtype=np.uint8))
    assert model.last_image is not None
    assert min(model.last_image.shape[:2]) >= RetinaFaceDetector._MIN_INPUT_SIDE


@pytest.mark.parametrize(
    "image",
    [
        None,
        [[0, 1], [2, 3]],
        np.empty((0, 10, 3), dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((1, 2, 3, 4, 5), dtype=np.uint8),
        np.zeros((10, 10, 2), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.bool_),
        np.zeros((10, 10, 3), dtype=np.complex64),
        np.asarray([[[object(), object(), object()]]], dtype=object),
        np.full((10, 10, 3), np.nan, dtype=np.float32),
        np.full((10, 10, 3), np.inf, dtype=np.float32),
    ],
    ids=[
        "none",
        "python-list",
        "empty-array",
        "one-dimensional",
        "five-dimensional",
        "two-channels",
        "bool-dtype",
        "complex-dtype",
        "object-dtype",
        "nan",
        "infinity",
    ],
)
def test_invalid_image_raises_value_error(image: Any) -> None:
    """Đầu vào sai loại, sai shape hoặc có giá trị lạ (NaN, Inf) đều phải báo ValueError."""
    detector = make_detector(FakeModel(lambda _image: np.empty((0, 15))))
    # match="Ảnh" vì mọi thông báo lỗi kiểm tra đầu vào đều bắt đầu bằng chữ "Ảnh"
    with pytest.raises(ValueError, match="Ảnh"):
        detector.detect(image)


def test_uint16_full_scale_is_normalized_to_uint8() -> None:
    """Ảnh uint16 có giá trị 65535 phải thành pixel 255 (chia 257)."""
    model = FakeModel(lambda _image: np.empty((0, 15)))
    make_detector(model).detect(np.full((10, 12, 3), 65535, dtype=np.uint16))
    assert model.last_image is not None
    assert model.last_image.dtype == np.uint8
    assert np.all(model.last_image == 255)


def test_one_pixel_image_is_accepted_and_upscaled() -> None:
    """Ảnh 1x1 vẫn là đầu vào hợp lệ và được phóng lên 32x32 trước khi chạy model."""
    model = FakeModel(lambda _image: np.empty((0, 15)))
    faces = make_detector(model).detect(np.zeros((1, 1, 3), dtype=np.uint8))
    assert faces == []
    assert model.last_image is not None
    assert model.last_image.shape == (32, 32, 3)


def test_missing_demo_image_reports_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Chạy demo với ảnh không tồn tại thì phải báo FileNotFoundError kèm đường dẫn file."""
    missing = tmp_path / "missing.png"
    monkeypatch.setattr(sys, "argv", ["retinaface"])
    with pytest.raises(FileNotFoundError, match="missing.png"):
        main(["--image", str(missing), "--out", str(tmp_path / "out.png")])

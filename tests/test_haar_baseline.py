"""Kiểm thử Haar baseline và schema dùng chung của CLI detection."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.detection.haar_baseline import HaarFaceDetector
from src.detection import run_detect


class FakeCascade:
    """Cascade giả để kiểm tra chuyển đổi output mà không phụ thuộc model."""

    def __init__(self) -> None:
        self.last_image: np.ndarray | None = None
        self.kwargs: dict[str, Any] = {}

    def empty(self) -> bool:
        return False

    def detectMultiScale3(
        self,
        image: np.ndarray,
        *,
        scaleFactor: float,
        minNeighbors: int,
        minSize: tuple[int, int],
        outputRejectLevels: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.last_image = image
        self.kwargs = {
            "scaleFactor": scaleFactor,
            "minNeighbors": minNeighbors,
            "minSize": minSize,
            "outputRejectLevels": outputRejectLevels,
        }
        return (
            np.asarray([[2, 3, 12, 14]], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            np.asarray([-2.0], dtype=np.float64),
        )


class TwoBoxCascade(FakeCascade):
    def detectMultiScale3(
        self,
        image: np.ndarray,
        *,
        scaleFactor: float,
        minNeighbors: int,
        minSize: tuple[int, int],
        outputRejectLevels: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.last_image = image
        return (
            np.asarray([[2, 3, 12, 14], [10, 11, 15, 16]], dtype=np.int32),
            np.asarray([1, 1], dtype=np.int32),
            np.asarray([-2.0, 2.0], dtype=np.float64),
        )


def make_detector(classifier: FakeCascade) -> HaarFaceDetector:
    detector = HaarFaceDetector.__new__(HaarFaceDetector)
    detector.scale_factor = 1.2
    detector.min_neighbors = 4
    detector.min_size = (20, 24)
    detector.equalize_hist = False
    detector._classifier = classifier
    return detector


def test_white_image_returns_empty_list() -> None:
    detector = HaarFaceDetector()
    assert detector.detect(np.full((100, 100, 3), 255, dtype=np.uint8)) == []


@pytest.mark.parametrize(
    "image",
    [
        None,
        [],
        np.array([], dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((1, 1, 1, 1, 1), dtype=np.uint8),
        np.zeros((10, 10, 2), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.float32),
        np.zeros((10, 10), dtype=np.bool_),
    ],
)
def test_invalid_image_input_is_rejected(image: Any) -> None:
    detector = make_detector(FakeCascade())
    with pytest.raises(ValueError):
        detector.detect(image)


@pytest.mark.parametrize("min_neighbors", [-1, 1.5, True])
def test_invalid_min_neighbors_is_rejected(min_neighbors: Any) -> None:
    with pytest.raises(ValueError, match="min_neighbors"):
        HaarFaceDetector(min_neighbors=min_neighbors)


def test_detect_returns_shared_face_contract_in_original_pixels() -> None:
    classifier = FakeCascade()
    detector = make_detector(classifier)
    faces = detector.detect(np.zeros((40, 50, 3), dtype=np.uint8))

    assert len(faces) == 1
    assert set(faces[0]) == {"bbox", "score", "landmarks"}
    assert faces[0]["bbox"] == [2.0, 3.0, 14.0, 17.0]
    assert faces[0]["score"] == pytest.approx(1.0 / (1.0 + np.exp(2.0)))
    assert faces[0]["landmarks"] is None
    assert classifier.kwargs == {
        "scaleFactor": 1.2,
        "minNeighbors": 4,
        "minSize": (20, 24),
        "outputRejectLevels": True,
    }


def test_detect_sorts_faces_by_descending_score() -> None:
    detector = make_detector(TwoBoxCascade())
    faces = detector.detect(np.zeros((40, 50, 3), dtype=np.uint8))

    assert [face["bbox"] for face in faces] == [
        [10.0, 11.0, 25.0, 27.0],
        [2.0, 3.0, 14.0, 17.0],
    ]
    assert [face["score"] for face in faces] == sorted(
        [face["score"] for face in faces], reverse=True
    )


def test_sigmoid_score_is_monotonic_and_bounded() -> None:
    scores = [HaarFaceDetector._weight_to_score(value) for value in (-5, 0, 5)]
    assert scores == sorted(scores)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_equalize_hist_is_applied_before_cascade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classifier = FakeCascade()
    detector = make_detector(classifier)
    detector.equalize_hist = True
    calls: list[np.ndarray] = []

    def equalize(image: np.ndarray) -> np.ndarray:
        calls.append(image)
        return image

    monkeypatch.setattr(cv2, "equalizeHist", equalize)
    detector.detect(np.zeros((40, 50, 3), dtype=np.uint8))
    assert len(calls) == 1
    assert classifier.last_image is calls[0]
    assert classifier.last_image is not None
    assert classifier.last_image.ndim == 2


def test_frontal_face_image_is_detected_when_fixture_is_available() -> None:
    image_path = os.environ.get("UY2_FACE_IMAGE")
    if not image_path:
        pytest.skip("Đặt UY2_FACE_IMAGE tới ảnh mặt chính diện để chạy test thật.")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert image is not None, f"Không đọc được ảnh fixture: {image_path}"
    assert HaarFaceDetector().detect(image)


def test_run_haar_uses_manifest_and_writes_empty_predictions(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "blank.png"
    assert cv2.imwrite(
        str(image_path), np.full((100, 100, 3), 255, dtype=np.uint8)
    )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "split"])
        writer.writeheader()
        writer.writerow({"image_path": "blank.png", "split": "test"})

    output_path = tmp_path / "predictions.json"
    predictions = run_detect.run_detection(
        detector_name="haar",
        manifest_path=manifest,
        output_path=output_path,
        split="test",
    )

    assert predictions == {"blank.png": []}
    assert json.loads(output_path.read_text(encoding="utf-8")) == predictions


@pytest.mark.parametrize("detector_name", ["haar", "retinaface"])
def test_run_detect_writes_same_prediction_schema(
    detector_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "face.png"
    assert cv2.imwrite(str(image_path), np.zeros((40, 50, 3), dtype=np.uint8))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "split"])
        writer.writeheader()
        writer.writerow({"image_path": "face.png", "split": "test"})

    class FakeDetector:
        def detect(self, _image: np.ndarray) -> list[dict[str, Any]]:
            return [
                {
                    "bbox": [1.0, 2.0, 20.0, 22.0],
                    "score": 0.75,
                    "landmarks": None,
                }
            ]

    monkeypatch.setattr(
        run_detect,
        "create_detector",
        lambda _name, _config: FakeDetector(),
    )
    output_path = tmp_path / f"{detector_name}.json"
    result = run_detect.run_detection(
        detector_name=detector_name,
        manifest_path=manifest,
        output_path=output_path,
    )
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert result == saved
    assert list(saved) == ["face.png"]
    assert saved["face.png"] == [
        {"bbox": [1.0, 2.0, 20.0, 22.0], "score": 0.75}
    ]

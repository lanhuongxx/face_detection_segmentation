
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

BBox = tuple[int, int, int, int]
ImageSize = tuple[int, int]  # (height, width)
LandmarkDict = dict[str, tuple[float, float]]
PrivacyMode = Literal[
    "blur",
    "pixelate",
    "mask",
    "mask_only",
    "blur_with_bbox",
    "original",
]


def validate_rgb_image(image: np.ndarray) -> np.ndarray:
    """Validate that `image` is a 3-channel uint8 RGB image of shape (H, W, 3)."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray image, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3) in RGB order, got {image.shape}."
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"Image has empty spatial dimension: {image.shape}.")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_val = float(np.max(image)) if image.size > 0 else 1.0
            if max_val <= 1.0:
                image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def clamp_bbox(bbox: tuple[float | int, ...], image_size: ImageSize) -> BBox | None:
    """Clamp `(x1, y1, x2, y2)` to `[0, W] x [0, H]`. Returns None if degenerate."""
    height, width = image_size
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got {len(bbox)}.")

    raw_x1, raw_y1, raw_x2, raw_y2 = (float(v) for v in bbox)
    x1 = max(0, min(width, int(np.floor(raw_x1))))
    y1 = max(0, min(height, int(np.floor(raw_y1))))
    x2 = max(0, min(width, int(np.ceil(raw_x2))))
    y2 = max(0, min(height, int(np.ceil(raw_y2))))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def validate_prob_mask(mask: np.ndarray, expected_size: ImageSize | None = None) -> np.ndarray:
    """Normalize a 2D or (H, W, 1) mask into contiguous float32 in [0.0, 1.0]."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray mask, got {type(mask).__name__}.")
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0, :, :]
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask (H, W), got shape {mask.shape}.")
    if expected_size is not None and mask.shape != expected_size:
        raise ValueError(
            f"Mask shape {mask.shape} does not match expected size {expected_size}."
        )
    mask_f32 = mask.astype(np.float32, copy=False)
    if np.issubdtype(mask.dtype, np.uint8) and float(np.max(mask_f32)) > 1.0:
        mask_f32 = mask_f32 / 255.0
    return np.ascontiguousarray(np.clip(mask_f32, 0.0, 1.0))


@dataclass(frozen=True)
class FaceDetection:
    """Single detected face in original image coordinates."""

    bbox: BBox
    score: float = 1.0
    landmarks: LandmarkDict | None = None

    def __post_init__(self) -> None:
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 ints, got {self.bbox}.")
        x1, y1, x2, y2 = (int(v) for v in self.bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox coordinates (x2<=x1 or y2<=y1): {self.bbox}.")
        object.__setattr__(self, "bbox", (x1, y1, x2, y2))
        object.__setattr__(self, "score", float(max(0.0, min(1.0, self.score))))


@dataclass(frozen=True)
class CropMetadata:
    """Coordinate mapping metadata between original image and resized U-Net crop."""

    image_size: ImageSize          # (orig_H, orig_W)
    orig_bbox: BBox                # (x1, y1, x2, y2)
    crop_box: BBox                 # (cx1, cy1, cx2, cy2) after margin + clamp
    crop_size: ImageSize           # (cy2 - cy1, cx2 - cx1)
    target_size: ImageSize         # (target_H, target_W)
    margin_ratio: float = 0.15


@dataclass
class FaceCrop:
    """Cropped and resized face patch paired with coordinate mapping metadata."""

    image: np.ndarray              # uint8, (target_H, target_W, 3), RGB
    metadata: CropMetadata
    detection: FaceDetection


@dataclass
class FaceSegmentationResult:
    """Segmentation output for one face in both crop space and original image space."""

    detection: FaceDetection
    crop_metadata: CropMetadata
    crop_prob_mask: np.ndarray     # float32, (target_H, target_W) in [0, 1]
    crop_mask: np.ndarray          # uint8, (target_H, target_W) in {0, 1}
    full_prob_mask: np.ndarray     # float32, (orig_H, orig_W) in [0, 1]
    full_mask: np.ndarray          # uint8, (orig_H, orig_W) in {0, 1}


@dataclass
class PipelineOutput:
    """End-to-end detection + segmentation output for one RGB image."""

    image_size: ImageSize
    faces: list[FaceSegmentationResult] = field(default_factory=list)
    combined_prob_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    combined_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.uint8))
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def num_faces(self) -> int:
        return len(self.faces)


@runtime_checkable
class FaceDetectorProtocol(Protocol):
    """Contract for face detection modules (e.g. RetinaFace, Haar Cascade)."""

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces in an RGB uint8 image `(H, W, 3)` and return original-coordinate detections."""
        ...


@runtime_checkable
class FaceSegmentorProtocol(Protocol):
    """Contract for face segmentation modules (e.g. U-Net)."""

    def predict(self, crop_image: np.ndarray) -> np.ndarray:
        """Predict soft probability mask `(H, W)` in `[0, 1]` for an RGB uint8 face crop."""
        ...


def coerce_detection(raw: Any, image_size: ImageSize) -> FaceDetection | None:
    """Convert common detector return formats into a validated `FaceDetection`."""
    if isinstance(raw, FaceDetection):
        clamped = clamp_bbox(raw.bbox, image_size)
        if clamped is None:
            return None
        return FaceDetection(bbox=clamped, score=raw.score, landmarks=raw.landmarks)

    if isinstance(raw, dict):
        box_raw = raw.get("bbox") or raw.get("box") or raw.get("facial_area")
        if box_raw is None:
            return None
        clamped = clamp_bbox(tuple(box_raw), image_size)
        if clamped is None:
            return None
        score = float(raw.get("score", raw.get("confidence", 1.0)))
        landmarks_raw = raw.get("landmarks")
        landmarks: LandmarkDict | None = None
        if isinstance(landmarks_raw, dict):
            landmarks = {
                str(k): (float(v[0]), float(v[1]))
                for k, v in landmarks_raw.items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
        return FaceDetection(bbox=clamped, score=score, landmarks=landmarks)

    if isinstance(raw, (list, tuple)) and len(raw) in (4, 5):
        clamped = clamp_bbox(tuple(raw[:4]), image_size)
        if clamped is None:
            return None
        score = float(raw[4]) if len(raw) == 5 else 1.0
        return FaceDetection(bbox=clamped, score=score)

    raise TypeError(f"Unsupported detection object type: {type(raw).__name__}.")

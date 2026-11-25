"""End-to-end face detection -> crop -> segmentation -> mask map-back pipeline."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from src.contracts import (
    BBox,
    CropMetadata,
    FaceCrop,
    FaceDetection,
    FaceDetectorProtocol,
    FaceSegmentationResult,
    FaceSegmentorProtocol,
    ImageSize,
    PipelineOutput,
    coerce_detection,
    validate_prob_mask,
    validate_rgb_image,
)


def _resize_hw(array: np.ndarray, size_hw: ImageSize, interpolation: int) -> np.ndarray:
    """Resize a 2D or 3D array to `(height, width)` using OpenCV."""
    target_h, target_w = int(size_hw[0]), int(size_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Target size must be positive, got {size_hw}.")
    if array.shape[:2] == (target_h, target_w):
        return array.copy()
    return cv2.resize(array, (target_w, target_h), interpolation=interpolation)


def compute_crop_box(
    bbox: BBox,
    image_size: ImageSize,
    margin_ratio: float = 0.15,
) -> BBox:
    """Expand `bbox` by `margin_ratio` and clamp strictly inside `(orig_H, orig_W)`."""
    if margin_ratio < 0.0:
        raise ValueError(f"margin_ratio must be >= 0, got {margin_ratio}.")
    orig_h, orig_w = image_size
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    dx = int(round(box_w * margin_ratio))
    dy = int(round(box_h * margin_ratio))

    cx1 = max(0, x1 - dx)
    cy1 = max(0, y1 - dy)
    cx2 = min(orig_w, x2 + dx)
    cy2 = min(orig_h, y2 + dy)

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"Degenerate crop box {(cx1, cy1, cx2, cy2)} for image {image_size}.")
    return (cx1, cy1, cx2, cy2)


def crop_face(
    image: np.ndarray,
    detection: FaceDetection,
    target_size: ImageSize = (256, 256),
    margin_ratio: float = 0.15,
) -> FaceCrop:
    """Crop a single detected face from an RGB image and record exact map-back metadata."""
    rgb = validate_rgb_image(image)
    orig_h, orig_w = rgb.shape[:2]
    crop_box = compute_crop_box(detection.bbox, (orig_h, orig_w), margin_ratio=margin_ratio)
    cx1, cy1, cx2, cy2 = crop_box
    patch = rgb[cy1:cy2, cx1:cx2]
    crop_h, crop_w = patch.shape[:2]

    resized = _resize_hw(patch, target_size, interpolation=cv2.INTER_LINEAR)
    metadata = CropMetadata(
        image_size=(orig_h, orig_w),
        orig_bbox=detection.bbox,
        crop_box=crop_box,
        crop_size=(crop_h, crop_w),
        target_size=(int(target_size[0]), int(target_size[1])),
        margin_ratio=float(margin_ratio),
    )
    return FaceCrop(image=np.ascontiguousarray(resized), metadata=metadata, detection=detection)


def crop_faces(
    image: np.ndarray,
    detections: Sequence[FaceDetection],
    target_size: ImageSize = (256, 256),
    margin_ratio: float = 0.15,
) -> list[FaceCrop]:
    """Crop all detected faces in original image coordinate order."""
    rgb = validate_rgb_image(image)
    return [
        crop_face(rgb, det, target_size=target_size, margin_ratio=margin_ratio)
        for det in detections
    ]


def map_mask_to_original(
    crop_prob_mask: np.ndarray,
    crop_metadata: CropMetadata,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize a crop mask back to `crop_size` and place it at `crop_box` on `(orig_H, orig_W)`."""
    prob_patch = validate_prob_mask(crop_prob_mask, expected_size=crop_metadata.target_size)
    orig_h, orig_w = crop_metadata.image_size
    cx1, cy1, cx2, cy2 = crop_metadata.crop_box
    crop_h, crop_w = crop_metadata.crop_size

    if (cy2 - cy1, cx2 - cx1) != (crop_h, crop_w):
        raise ValueError(
            f"CropMetadata mismatch: crop_box {(cx1, cy1, cx2, cy2)} vs crop_size {(crop_h, crop_w)}."
        )

    resized_patch = _resize_hw(prob_patch, (crop_h, crop_w), interpolation=cv2.INTER_LINEAR)
    resized_patch = np.clip(resized_patch, 0.0, 1.0).astype(np.float32, copy=False)

    full_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
    full_prob[cy1:cy2, cx1:cx2] = resized_patch
    full_mask = (full_prob >= threshold).astype(np.uint8)
    return full_prob, full_mask


def merge_face_masks(
    image_size: ImageSize,
    face_results: Sequence[FaceSegmentationResult],
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge per-face full-image masks via pixel-wise maximum so multi-face overlaps stay intact."""
    orig_h, orig_w = image_size
    combined_prob = np.zeros((orig_h, orig_w), dtype=np.float32)
    for result in face_results:
        np.maximum(combined_prob, result.full_prob_mask, out=combined_prob)
    combined_mask = (combined_prob >= threshold).astype(np.uint8)
    return combined_prob, combined_mask


class RetinaFaceDetectorAdapter(FaceDetectorProtocol):
    """Adapter unifying custom RetinaFace modules, `retina-face` PyPI, and Haar fallback."""

    def __init__(
        self,
        backend: str = "auto",
        conf_threshold: float = 0.5,
        custom_detector: FaceDetectorProtocol | Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self.requested_backend = backend
        self.conf_threshold = float(conf_threshold)
        self.custom_detector = custom_detector
        self.backend_used = "uninitialized"
        self._haar_cascade: cv2.CascadeClassifier | None = None
        self._retinaface_api: Any = None
        self._team_detector: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.custom_detector is not None:
            self.backend_used = "custom"
            return

        if self.requested_backend not in ("auto", "retinaface", "haar"):
            raise ValueError(f"Unsupported detector backend: {self.requested_backend!r}.")

        if self.requested_backend in ("auto", "retinaface"):
            if self._try_load_team_retinaface():
                self.backend_used = "team_retinaface"
                return
            if self._try_load_pypi_retinaface():
                self.backend_used = "retinaface"
                return
            if self.requested_backend == "retinaface":
                raise ImportError(
                    "RetinaFace backend requested, but neither `src.detection.retinaface` "
                    "nor the `retina-face` package is available."
                )

        if hasattr(cv2, "CascadeClassifier"):
            cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self._haar_cascade = cascade
                self.backend_used = "haar_fallback"
                return
            if self.requested_backend == "haar":
                raise RuntimeError(f"Failed to load OpenCV Haar Cascade at {cascade_path}.")
        elif self.requested_backend == "haar":
            raise RuntimeError(
                f"Haar Cascade requested, but cv2 {cv2.__version__} has no CascadeClassifier. "
                "Install `opencv-python<5.0.0` or use `retinaface`."
            )

        self.backend_used = "skin_fallback"

    def _try_load_team_retinaface(self) -> bool:
        try:
            mod = importlib.import_module("src.detection.retinaface")
        except ImportError:
            return False

        for attr in ("RetinaFaceDetector", "detect_faces", "detect"):
            if hasattr(mod, attr):
                obj = getattr(mod, attr)
                self._team_detector = obj() if isinstance(obj, type) else obj
                return True
        return False

    def _try_load_pypi_retinaface(self) -> bool:
        try:
            mod = importlib.import_module("retinaface")
        except ImportError:
            return False
        if hasattr(mod, "RetinaFace"):
            self._retinaface_api = mod.RetinaFace
            return True
        return False

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        rgb = validate_rgb_image(image)
        orig_h, orig_w = rgb.shape[:2]

        if self.custom_detector is not None:
            raw = (
                self.custom_detector.detect(rgb)
                if hasattr(self.custom_detector, "detect")
                else self.custom_detector(rgb)
            )
            return self._normalize_detections(raw, (orig_h, orig_w))

        if self.backend_used == "team_retinaface" and self._team_detector is not None:
            raw = (
                self._team_detector.detect(rgb)
                if hasattr(self._team_detector, "detect")
                else self._team_detector(rgb)
            )
            return self._normalize_detections(raw, (orig_h, orig_w))

        if self.backend_used == "retinaface" and self._retinaface_api is not None:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            raw = self._retinaface_api.detect_faces(bgr, threshold=self.conf_threshold)
            return self._normalize_detections(raw, (orig_h, orig_w))

        if self.backend_used == "haar_fallback":
            return self._detect_haar(rgb)
        return self._detect_skin_regions(rgb)

    def _detect_skin_regions(self, rgb: np.ndarray) -> list[FaceDetection]:
        """Lightweight YCrCb skin-region fallback detector when Haar/RetinaFace are unavailable."""
        orig_h, orig_w = rgb.shape[:2]
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        mask = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = max(400, int(0.005 * orig_h * orig_w))
        detections: list[FaceDetection] = []
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if int(area) < min_area or w <= 0 or h <= 0:
                continue
            aspect = h / float(w)
            if 0.6 <= aspect <= 2.2:
                det = coerce_detection((int(x), int(y), int(x + w), int(y + h), 0.65), (orig_h, orig_w))
                if det is not None and det.score >= self.conf_threshold:
                    detections.append(det)
        detections.sort(key=lambda d: (d.bbox[0], d.bbox[1]))
        return detections

    def _detect_haar(self, rgb: np.ndarray) -> list[FaceDetection]:
        if self._haar_cascade is None:
            return []
        orig_h, orig_w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self._haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(boxes) == 0:
            return []
        detections: list[FaceDetection] = []
        for x, y, w, h in boxes:
            det = coerce_detection((int(x), int(y), int(x + w), int(y + h), 0.95), (orig_h, orig_w))
            if det is not None and det.score >= self.conf_threshold:
                detections.append(det)
        detections.sort(key=lambda d: (d.bbox[0], d.bbox[1]))
        return detections

    def _normalize_detections(self, raw: Any, image_size: ImageSize) -> list[FaceDetection]:
        if raw is None:
            return []
        items: list[Any]
        if isinstance(raw, dict):
            # Handles `{"face_1": {"facial_area": [...], "score": ...}}` format.
            if "bbox" in raw or "box" in raw or "facial_area" in raw:
                items = [raw]
            else:
                items = list(raw.values())
        elif isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            raise TypeError(f"Unsupported detector return type: {type(raw).__name__}.")

        results: list[FaceDetection] = []
        for item in items:
            det = coerce_detection(item, image_size)
            if det is not None and det.score >= self.conf_threshold:
                results.append(det)
        results.sort(key=lambda d: d.score, reverse=True)
        return results


class UNetSegmentorAdapter(FaceSegmentorProtocol):
    """Adapter unifying custom U-Net modules, PyTorch checkpoints, and a skin-ellipse fallback."""

    def __init__(
        self,
        backend: str = "auto",
        input_size: ImageSize = (256, 256),
        checkpoint_path: str | Path | None = None,
        custom_segmentor: FaceSegmentorProtocol | Callable[[np.ndarray], np.ndarray] | None = None,
        device: str = "cpu",
    ) -> None:
        self.requested_backend = backend
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.custom_segmentor = custom_segmentor
        self.device = device
        self.backend_used = "uninitialized"
        self._torch: Any = None
        self._model: Any = None
        self._init_backend()

    def _init_backend(self) -> None:
        if self.custom_segmentor is not None:
            self.backend_used = "custom"
            return

        if self.requested_backend not in ("auto", "unet", "heuristic"):
            raise ValueError(f"Unsupported segmentor backend: {self.requested_backend!r}.")

        if self.requested_backend in ("auto", "unet") and self.checkpoint_path is not None:
            self._load_unet_checkpoint(self.checkpoint_path)
            self.backend_used = "unet_checkpoint"
            return

        if self.requested_backend == "unet":
            raise FileNotFoundError(
                "U-Net backend requested, but `checkpoint_path` was not provided."
            )

        self.backend_used = "heuristic_fallback"

    def _load_unet_checkpoint(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"U-Net checkpoint not found: {path}")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to load a U-Net checkpoint. Install `torch` first."
            ) from exc

        self._torch = torch
        loaded = torch.load(str(path), map_location=self.device)
        if hasattr(loaded, "eval") and hasattr(loaded, "forward"):
            model = loaded
        else:
            unet_mod = importlib.import_module("src.segmentation.unet")
            model_cls = getattr(unet_mod, "UNet")
            model = model_cls()
            state_dict = (
                loaded.get("model_state_dict")
                or loaded.get("state_dict")
                or loaded
            )
            model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self._model = model

    def predict(self, crop_image: np.ndarray) -> np.ndarray:
        rgb = validate_rgb_image(crop_image)
        crop_h, crop_w = rgb.shape[:2]

        if self.custom_segmentor is not None:
            raw = (
                self.custom_segmentor.predict(rgb)
                if hasattr(self.custom_segmentor, "predict")
                else self.custom_segmentor(rgb)
            )
            mask = validate_prob_mask(np.asarray(raw))
            if mask.shape != (crop_h, crop_w):
                mask = _resize_hw(mask, (crop_h, crop_w), interpolation=cv2.INTER_LINEAR)
            return np.clip(mask, 0.0, 1.0).astype(np.float32, copy=False)

        if self.backend_used == "unet_checkpoint" and self._model is not None:
            return self._predict_torch(rgb, (crop_h, crop_w))

        return self._predict_heuristic(rgb)

    def _predict_torch(self, rgb: np.ndarray, out_size: ImageSize) -> np.ndarray:
        torch = self._torch
        resized = _resize_hw(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred = self._model(tensor)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if float(pred.min()) < 0.0 or float(pred.max()) > 1.0:
                pred = torch.sigmoid(pred)
            prob = pred.squeeze().detach().cpu().numpy().astype(np.float32)

        if prob.ndim != 2:
            prob = validate_prob_mask(prob)
        if prob.shape != out_size:
            prob = _resize_hw(prob, out_size, interpolation=cv2.INTER_LINEAR)
        return np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _predict_heuristic(rgb: np.ndarray) -> np.ndarray:
        """Fallback soft face mask combining an elliptical facial prior and YCrCb skin cues."""
        h, w = rgb.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h * 0.5, w * 0.5
        ry, rx = max(1.0, h * 0.42), max(1.0, w * 0.34)
        ellipse_dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        ellipse_prior = np.clip(1.35 - ellipse_dist, 0.0, 1.0).astype(np.float32)

        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        skin = cv2.inRange(
            ycrcb,
            np.array([0, 130, 75], dtype=np.uint8),
            np.array([255, 178, 135], dtype=np.uint8),
        ).astype(np.float32) / 255.0

        # Blend skin evidence inside the ellipse so non-skin background stays low.
        core_ellipse = (ellipse_dist <= 0.72).astype(np.float32)
        prob = np.clip(0.55 * ellipse_prior + 0.45 * (ellipse_prior * np.maximum(skin, core_ellipse)), 0.0, 1.0)
        k = max(3, (min(h, w) // 16) | 1)
        prob = cv2.GaussianBlur(prob, (k, k), 0)
        return np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)


@dataclass
class PipelineConfig:
    """Configuration for `FacePrivacyPipeline`."""

    detector_backend: str = "auto"
    segmentor_backend: str = "auto"
    conf_threshold: float = 0.5
    mask_threshold: float = 0.5
    crop_margin_ratio: float = 0.15
    unet_input_size: ImageSize = (256, 256)
    unet_checkpoint: str | Path | None = None
    device: str = "cpu"


class FacePrivacyPipeline:
    """Orchestrates Detection -> Crop -> Segmentation -> Mask Map-Back."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: FaceDetectorProtocol | Callable[[np.ndarray], Any] | None = None,
        segmentor: FaceSegmentorProtocol | Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = RetinaFaceDetectorAdapter(
            backend=self.config.detector_backend,
            conf_threshold=self.config.conf_threshold,
            custom_detector=detector,
        )
        self.segmentor = UNetSegmentorAdapter(
            backend=self.config.segmentor_backend,
            input_size=self.config.unet_input_size,
            checkpoint_path=self.config.unet_checkpoint,
            custom_segmentor=segmentor,
            device=self.config.device,
        )

    def run(self, image: np.ndarray) -> PipelineOutput:
        """Run detection, crop, segmentation, and exact coordinate map-back on an RGB image."""
        t0 = time.perf_counter()
        rgb = validate_rgb_image(image)
        orig_h, orig_w = rgb.shape[:2]
        image_size = (orig_h, orig_w)

        t_det_start = time.perf_counter()
        detections = self.detector.detect(rgb)
        detect_ms = (time.perf_counter() - t_det_start) * 1000.0

        if not detections:
            empty_prob = np.zeros(image_size, dtype=np.float32)
            empty_bin = np.zeros(image_size, dtype=np.uint8)
            total_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineOutput(
                image_size=image_size,
                faces=[],
                combined_prob_mask=empty_prob,
                combined_mask=empty_bin,
                timings_ms={
                    "detect_ms": detect_ms,
                    "crop_ms": 0.0,
                    "segment_ms": 0.0,
                    "map_back_ms": 0.0,
                    "total_ms": total_ms,
                },
            )

        t_crop_start = time.perf_counter()
        crops = crop_faces(
            rgb,
            detections,
            target_size=self.config.unet_input_size,
            margin_ratio=self.config.crop_margin_ratio,
        )
        crop_ms = (time.perf_counter() - t_crop_start) * 1000.0

        t_seg_start = time.perf_counter()
        crop_prob_masks = [self.segmentor.predict(crop.image) for crop in crops]
        segment_ms = (time.perf_counter() - t_seg_start) * 1000.0

        t_map_start = time.perf_counter()
        face_results: list[FaceSegmentationResult] = []
        for crop, crop_prob in zip(crops, crop_prob_masks):
            crop_prob_valid = validate_prob_mask(crop_prob, expected_size=crop.metadata.target_size)
            crop_bin = (crop_prob_valid >= self.config.mask_threshold).astype(np.uint8)
            full_prob, full_mask = map_mask_to_original(
                crop_prob_valid,
                crop.metadata,
                threshold=self.config.mask_threshold,
            )
            face_results.append(
                FaceSegmentationResult(
                    detection=crop.detection,
                    crop_metadata=crop.metadata,
                    crop_prob_mask=crop_prob_valid,
                    crop_mask=crop_bin,
                    full_prob_mask=full_prob,
                    full_mask=full_mask,
                )
            )

        combined_prob, combined_mask = merge_face_masks(
            image_size,
            face_results,
            threshold=self.config.mask_threshold,
        )
        map_back_ms = (time.perf_counter() - t_map_start) * 1000.0
        total_ms = (time.perf_counter() - t0) * 1000.0

        return PipelineOutput(
            image_size=image_size,
            faces=face_results,
            combined_prob_mask=combined_prob,
            combined_mask=combined_mask,
            timings_ms={
                "detect_ms": detect_ms,
                "crop_ms": crop_ms,
                "segment_ms": segment_ms,
                "map_back_ms": map_back_ms,
                "total_ms": total_ms,
            },
        )

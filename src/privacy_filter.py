"""Mask-guided face privacy filtering, visualization modes, and image saving."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.contracts import (
    FaceDetection,
    PipelineOutput,
    PrivacyMode,
    validate_prob_mask,
    validate_rgb_image,
)

_MULTI_FACE_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 200, 255),
    (80, 220, 120),
    (255, 165, 60),
    (220, 90, 220),
    (255, 95, 110),
    (120, 160, 255),
)


@dataclass
class PrivacyFilterConfig:
    """Visual and anonymization settings for `PrivacyFilter`."""

    mode: PrivacyMode = "blur"
    blur_strength: int = 35
    feather_radius: int = 9
    pixel_block_size: int = 16
    mask_color: tuple[int, int, int] = (0, 200, 255)
    mask_alpha: float = 0.45
    draw_bbox: bool = False
    draw_labels: bool = True
    bbox_color: tuple[int, int, int] = (50, 230, 110)
    bbox_thickness: int = 2
    draw_contours: bool = True


def _make_odd(value: int, minimum: int = 1) -> int:
    val = max(minimum, int(value))
    return val if val % 2 == 1 else val + 1


def _build_soft_alpha(binary_mask: np.ndarray, feather_radius: int) -> np.ndarray:
    """Create a `[0, 1]` blending matte from a binary mask, optionally feathered."""
    alpha = (binary_mask > 0).astype(np.float32)
    if feather_radius > 1 and np.any(alpha > 0):
        k = _make_odd(feather_radius, minimum=3)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def _blur_image(rgb: np.ndarray, blur_strength: int) -> np.ndarray:
    k = _make_odd(blur_strength, minimum=3)
    return cv2.GaussianBlur(rgb, (k, k), 0)


def _pixelate_image(rgb: np.ndarray, block_size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    step = max(2, int(block_size))
    small_w = max(1, w // step)
    small_h = max(1, h // step)
    down = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)


def _draw_detections(
    rgb: np.ndarray,
    detections: list[FaceDetection],
    color: tuple[int, int, int],
    thickness: int,
    draw_labels: bool,
) -> np.ndarray:
    canvas = rgb.copy()
    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, max(1, thickness))
        if draw_labels:
            label = f"Face {idx}: {det.score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(canvas, (x1, top), (min(canvas.shape[1], x1 + tw + 6), y1), color, -1)
            cv2.putText(
                canvas,
                label,
                (x1 + 3, max(th + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )
    return canvas


class PrivacyFilter:
    """Applies mask-based face blurring, pixelation, or mask visualization."""

    def __init__(self, default_config: PrivacyFilterConfig | None = None) -> None:
        self.default_config = default_config or PrivacyFilterConfig()

    def apply(
        self,
        image: np.ndarray,
        pipeline_output: PipelineOutput,
        config: PrivacyFilterConfig | None = None,
    ) -> np.ndarray:
        """Render the chosen privacy/visualization mode using `PipelineOutput`."""
        cfg = config or self.default_config
        rgb = validate_rgb_image(image)
        detections = [face.detection for face in pipeline_output.faces]
        face_masks = [face.full_mask for face in pipeline_output.faces]
        return self.apply_with_mask(
            rgb,
            pipeline_output.combined_mask,
            detections=detections,
            per_face_masks=face_masks,
            config=cfg,
        )

    def apply_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        detections: list[FaceDetection] | None = None,
        per_face_masks: list[np.ndarray] | None = None,
        config: PrivacyFilterConfig | None = None,
    ) -> np.ndarray:
        """Apply privacy filter or mask overlay directly from an RGB image and 2D mask."""
        cfg = config or self.default_config
        rgb = validate_rgb_image(image)
        h, w = rgb.shape[:2]
        mask_f = validate_prob_mask(mask, expected_size=(h, w))
        bin_mask = (mask_f >= 0.5).astype(np.uint8)
        dets = detections or []

        if cfg.mode == "original":
            return rgb.copy()

        if cfg.mode in ("blur", "blur_with_bbox"):
            if not np.any(bin_mask):
                out = rgb.copy()
            else:
                blurred = _blur_image(rgb, cfg.blur_strength)
                alpha = _build_soft_alpha(bin_mask, cfg.feather_radius)[..., None]
                out = (blurred.astype(np.float32) * alpha + rgb.astype(np.float32) * (1.0 - alpha))
                out = np.clip(out, 0, 255).astype(np.uint8)
            if cfg.mode == "blur_with_bbox" or cfg.draw_bbox:
                out = _draw_detections(out, dets, cfg.bbox_color, cfg.bbox_thickness, cfg.draw_labels)
            return out

        if cfg.mode == "pixelate":
            if not np.any(bin_mask):
                out = rgb.copy()
            else:
                pix = _pixelate_image(rgb, cfg.pixel_block_size)
                alpha = (bin_mask > 0).astype(np.float32)[..., None]
                out = np.where(alpha > 0.5, pix, rgb)
            if cfg.draw_bbox:
                out = _draw_detections(out, dets, cfg.bbox_color, cfg.bbox_thickness, cfg.draw_labels)
            return out

        if cfg.mode == "mask":
            out = self._render_mask_overlay(rgb, bin_mask, per_face_masks, cfg)
            if cfg.draw_bbox:
                out = _draw_detections(out, dets, cfg.bbox_color, cfg.bbox_thickness, cfg.draw_labels)
            return out

        if cfg.mode == "mask_only":
            return self._render_mask_only((h, w), bin_mask, per_face_masks, cfg)

        raise ValueError(f"Unsupported privacy mode: {cfg.mode!r}.")

    @staticmethod
    def _render_mask_overlay(
        rgb: np.ndarray,
        bin_mask: np.ndarray,
        per_face_masks: list[np.ndarray] | None,
        cfg: PrivacyFilterConfig,
    ) -> np.ndarray:
        out = rgb.copy()
        alpha = float(np.clip(cfg.mask_alpha, 0.0, 1.0))

        masks_to_draw = per_face_masks if per_face_masks else ([bin_mask] if np.any(bin_mask) else [])
        for idx, m in enumerate(masks_to_draw):
            region = m > 0
            if not np.any(region):
                continue
            color = (
                _MULTI_FACE_PALETTE[idx % len(_MULTI_FACE_PALETTE)]
                if per_face_masks and len(per_face_masks) > 1
                else cfg.mask_color
            )
            color_arr = np.array(color, dtype=np.float32)
            out_region = out[region].astype(np.float32)
            out[region] = np.clip((1.0 - alpha) * out_region + alpha * color_arr, 0, 255).astype(np.uint8)

            if cfg.draw_contours:
                contours, _ = cv2.findContours(
                    region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(out, contours, -1, color, 2)
        return out

    @staticmethod
    def _render_mask_only(
        image_size: tuple[int, int],
        bin_mask: np.ndarray,
        per_face_masks: list[np.ndarray] | None,
        cfg: PrivacyFilterConfig,
    ) -> np.ndarray:
        h, w = image_size
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        masks_to_draw = per_face_masks if per_face_masks else ([bin_mask] if np.any(bin_mask) else [])
        for idx, m in enumerate(masks_to_draw):
            region = m > 0
            if not np.any(region):
                continue
            color = (
                _MULTI_FACE_PALETTE[idx % len(_MULTI_FACE_PALETTE)]
                if per_face_masks and len(per_face_masks) > 1
                else cfg.mask_color
            )
            canvas[region] = np.array(color, dtype=np.uint8)
        return canvas

    @staticmethod
    def save_image(image: np.ndarray, output_path: str | Path, quality: int = 95) -> Path:
        """Save an RGB uint8 image to disk (handling RGB -> BGR conversion)."""
        rgb = validate_rgb_image(image)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        ext = path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, int(np.clip(quality, 1, 100))]

        ok = cv2.imwrite(str(path), bgr, params)
        if not ok:
            raise OSError(f"Failed to write image to {path}.")
        return path

    @staticmethod
    def encode_image(image: np.ndarray, ext: str = ".png", quality: int = 95) -> bytes:
        """Encode an RGB uint8 image to bytes for web downloads."""
        rgb = validate_rgb_image(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        norm_ext = ext if ext.startswith(".") else f".{ext}"
        params: list[int] = []
        if norm_ext.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 1, 100))]
        ok, buf = cv2.imencode(norm_ext, bgr, params)
        if not ok:
            raise RuntimeError(f"Failed to encode image as {norm_ext}.")
        return buf.tobytes()

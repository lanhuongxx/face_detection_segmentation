"""Compatibility facade exporting `FacePrivacyPipeline` under `src.pipeline`."""

from src.inference import (
    FacePrivacyPipeline,
    PipelineConfig,
    RetinaFaceDetectorAdapter,
    UNetSegmentorAdapter,
    compute_crop_box,
    crop_face,
    crop_faces,
    map_mask_to_original,
    merge_face_masks,
)

__all__ = [
    "FacePrivacyPipeline",
    "PipelineConfig",
    "RetinaFaceDetectorAdapter",
    "UNetSegmentorAdapter",
    "compute_crop_box",
    "crop_face",
    "crop_faces",
    "map_mask_to_original",
    "merge_face_masks",
]

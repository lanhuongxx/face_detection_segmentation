"""Pipeline phát hiện, phân vùng và che khuôn mặt theo lựa chọn."""

import time

import numpy as np

from src.pipeline import privacy_filter, visualize
from src.pipeline.crop import crop_faces, square_crop_box
from src.pipeline.mask_mapping import (
    build_instance_map,
    ellipse_mask,
    mask_quality_ratio,
    prob_to_local_mask,
)
from src.pipeline.preprocess_input import limit_size
from src.pipeline.privacy_filter import PrivacyOptions
from src.pipeline.schemas import Face, PipelineResult
from src.utils.config import load_config, resolve_device


def _ms(start):
    return (time.perf_counter() - start) * 1000.0


def _box_iou(a, b):
    """Tính IoU của hai bbox hợp lệ theo định dạng x1, y1, x2, y2."""
    overlap_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = overlap_w * overlap_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


class FacePipeline:
    def __init__(self, cfg=None, detector=None, segmenter=None):
        """Cho phép truyền detector và segmenter giả khi kiểm thử."""
        self.cfg = cfg if cfg is not None else load_config()
        self.device = resolve_device(self.cfg)
        self._detector = detector
        self._segmenter = segmenter

    def _get_detector(self):
        if self._detector is None:
            from src.detection.retinaface import RetinaFaceDetector

            self._detector = RetinaFaceDetector(
                self.cfg["detection"]["model"], self.device
            )
        return self._detector

    def _get_segmenter(self):
        if self._segmenter is None:
            from src.segmentation.unet import FaceSegmenter

            seg = self.cfg["segmentation"]
            self._segmenter = FaceSegmenter(
                seg["weights"], self.device, seg["batch_size"]
            )
        return self._segmenter

    def _clean_detections(self, detections, shape):
        """Cắt bbox theo biên ảnh, lọc khung trùng rồi gán ID từ trái sang phải."""
        height, width = shape[:2]
        candidates = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = int(np.clip(x1, 0, width))
            x2 = int(np.clip(x2, 0, width))
            y1 = int(np.clip(y1, 0, height))
            y2 = int(np.clip(y2, 0, height))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                candidates.append((det, (x1, y1, x2, y2)))

        candidates.sort(key=lambda item: -item[0].score)
        detection_cfg = self.cfg["detection"]
        nms_iou = float(detection_cfg.get("nms_iou_threshold", 0.38))
        max_faces = int(detection_cfg["max_faces"])

        kept = []
        for det, box in candidates:
            if len(kept) >= max_faces:
                break
            if any(_box_iou(box, saved_box) >= nms_iou for _, saved_box in kept):
                continue
            kept.append((det, box))

        kept.sort(key=lambda item: item[1][0])
        return [
            Face(id=i, bbox=box, score=float(det.score), landmarks=det.landmarks)
            for i, (det, box) in enumerate(kept, start=1)
        ]

    def analyze(self, image, conf_threshold=None, source=None, mask_threshold=None):
        """Nhận ảnh RGB uint8 và trả kết quả phát hiện, phân vùng."""
        total_start = time.perf_counter()
        cfg = self.cfg
        conf = (
            cfg["detection"]["conf_threshold"]
            if conf_threshold is None
            else conf_threshold
        )
        thr = (
            cfg["segmentation"]["mask_threshold"]
            if mask_threshold is None
            else mask_threshold
        )
        timing = {}

        # 0. Tiền xử lý.
        t = time.perf_counter()
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Ảnh phải là mảng uint8 dạng (H, W, 3) RGB.")
        orig_hw = image.shape[:2]
        work, scale = limit_size(
            np.ascontiguousarray(image), cfg["preprocess"]["max_side"]
        )
        timing["preprocess"] = _ms(t)

        # 1. Phát hiện và lọc khung trùng trước khi crop.
        t = time.perf_counter()
        detections = self._get_detector().detect(work, conf)
        faces = self._clean_detections(detections, work.shape)
        timing["detect"] = _ms(t)

        # 2. Crop các mặt đủ lớn cho U-Net.
        t = time.perf_counter()
        crop_cfg = cfg["crop"]
        seg_faces, boxes = [], []
        for face in faces:
            w, h = face.size_px
            if min(w, h) < crop_cfg["min_face_px"]:
                face.status, face.reason = "fallback_ellipse", "too_small"
            else:
                face.crop_box = square_crop_box(face.bbox, crop_cfg["margin"])
                seg_faces.append(face)
                boxes.append(face.crop_box)
        crops = crop_faces(work, boxes, crop_cfg["size"]) if boxes else []
        timing["crop"] = _ms(t)

        # 3. Phân vùng.
        t = time.perf_counter()
        probs = self._get_segmenter().predict(np.stack(crops)) if crops else []
        timing["segment"] = _ms(t)

        # 4. Đưa mask về tọa độ ảnh và xử lý mask không đạt.
        t = time.perf_counter()
        min_ratio = cfg["segmentation"]["min_mask_ratio"]
        for face, prob in zip(seg_faces, probs):
            local = prob_to_local_mask(prob, face.crop_box, thr)
            if local.any() and mask_quality_ratio(
                local, face.crop_box, face.bbox
            ) >= min_ratio:
                face.mask_local, face.mask_box = local, face.crop_box
            else:
                face.status, face.reason = "fallback_ellipse", "bad_mask"

        for face in faces:
            if face.status == "fallback_ellipse":
                face.mask_local, face.mask_box = ellipse_mask(face.bbox)
            face.thumbnail = visualize.make_thumbnail(work, face.bbox)

        instance_map = build_instance_map(work.shape, faces)
        timing["map_back"] = _ms(t)

        # 5. Tạo ảnh hiển thị.
        overlay_boxes = visualize.draw_boxes(work, faces)
        overlay_masks = visualize.draw_masks(work, faces, instance_map)
        timing["total"] = _ms(total_start)

        meta = {
            "source": dict(source or {}),
            "orig_size": orig_hw,
            "work_size": work.shape[:2],
            "scale": scale,
            "device": self.device,
            "params": {
                "conf_threshold": conf,
                "margin": crop_cfg["margin"],
                "crop_size": crop_cfg["size"],
                "mask_threshold": thr,
            },
        }
        return PipelineResult(
            image=work,
            faces=faces,
            instance_map=instance_map,
            overlay_boxes=overlay_boxes,
            overlay_masks=overlay_masks,
            result_image=work.copy(),
            timing_ms=timing,
            meta=meta,
        )

    def apply_privacy(
        self, result, selected_ids, mode=None, strength=None, region=None, store=True
    ):
        """Che các mặt đã chọn và trả về ảnh kết quả."""
        privacy = self.cfg["privacy"]
        opts = PrivacyOptions(
            mode=mode or privacy["mode"],
            strength=privacy["strength"] if strength is None else strength,
            region=region or privacy["region"],
            dilate_px=privacy["dilate_px"],
            feather_px=privacy["feather_px"],
            solid_color=tuple(privacy["solid_color"]),
        )
        selected = set(selected_ids)
        t = time.perf_counter()
        out = privacy_filter.apply_privacy(result.image, result.faces, selected, opts)

        if store:
            result.result_image = out
            for face in result.faces:
                face.blurred = face.id in selected
            result.timing_ms["filter"] = _ms(t)
            result.meta["params"].update(
                {"mode": opts.mode, "strength": opts.strength, "region": opts.region}
            )
        return out

    def report(self, result):
        """Trả về báo cáo của lần phân tích dưới dạng dict."""
        return visualize.build_report(result)
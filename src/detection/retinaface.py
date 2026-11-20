"""Module 1: RetinaFace (facexlib) trả về List[Detection]."""
import cv2
import numpy as np

from src.pipeline.schemas import Detection


class RetinaFaceDetector:
    def __init__(self, model_name="retinaface_mobile0.25", device="cuda"):
        from facexlib.detection import init_detection_model   # nhập muộn để test không cần facexlib
        try:
            self.model = init_detection_model(model_name, half=False, device=device)
        except TypeError:                                      # bản facexlib cũ không có tham số device
            self.model = init_detection_model(model_name, half=False)

    def detect(self, image_rgb, conf_threshold=0.5):
        """image_rgb: (H,W,3) uint8 RGB. Trả về danh sách Detection (có thể rỗng)."""
        import torch
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)       # facexlib cần BGR
        with torch.inference_mode():
            raw = self.model.detect_faces(bgr, conf_threshold)
        detections = []
        for row in raw:
            row = np.asarray(row, dtype=np.float32)
            x1, y1, x2, y2 = (int(round(float(v))) for v in row[:4])
            landmarks = row[5:15].reshape(5, 2) if len(row) >= 15 else None
            detections.append(Detection((x1, y1, x2, y2), float(row[4]), landmarks))
        return detections

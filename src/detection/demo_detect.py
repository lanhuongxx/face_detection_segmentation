from pathlib import Path

import cv2
import torch
from facexlib.detection import init_detection_model

project = Path(__file__).resolve().parents[2]
image_path = project / "data" / "demo" / "demo_group.jpg"
output_path = project / "data" / "demo" / "demo_group_boxes.png"

image = cv2.imread(str(image_path))
if image is None:
    raise SystemExit(f"Không đọc được ảnh: {image_path}")

detector = init_detection_model("retinaface_mobile0.25", half=False)

with torch.no_grad():
    faces = detector.detect_faces(image, 0.8)

for face in faces:
    x1, y1, x2, y2 = map(int, face[:4])
    score = float(face[4])
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{score:.2f}", (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

if not cv2.imwrite(str(output_path), image):
    raise SystemExit(f"Không lưu được ảnh: {output_path}")

print(f"Số mặt phát hiện: {len(faces)}")
print(f"Ảnh kết quả: {output_path}")
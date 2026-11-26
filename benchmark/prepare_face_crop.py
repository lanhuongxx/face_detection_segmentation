
from pathlib import Path
import cv2

# Đọc ảnh demo
image_path = Path("data/demo/demo_group.jpg")
image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

# Phát hiện khuôn mặt để lấy vùng crop thử nghiệm
detector_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
detector = cv2.CascadeClassifier(detector_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(30, 30),
)

if len(faces) == 0:
    print("Khong tim thay khuon mat. Can chon anh khac.")
else:
    # Chọn khuôn mặt lớn nhất
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Thêm vùng xung quanh khuôn mặt
    margin = int(max(w, h) * 0.25)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image.shape[1], x + w + margin)
    y2 = min(image.shape[0], y + h + margin)

    crop = image[y1:y2, x1:x2]

    output_dir = Path("benchmark/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "face_crop.jpg"
    cv2.imwrite(str(output_path), crop)

    print("So khuon mat phat hien:", len(faces))
    print("Kich thuoc crop:", crop.shape)
    print("Da luu anh:", output_path)
    print("PREPARE FACE CROP: PASS")

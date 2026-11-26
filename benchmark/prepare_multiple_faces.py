
from pathlib import Path
import cv2

# Hai ảnh demo của dự án
IMAGE_PATHS = [
    Path("data/demo/demo_group.jpg"),
    Path("data/demo/images2.jpg"),
]

OUTPUT_DIR = Path("benchmark/outputs/multiple_faces")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Khởi tạo Haar Cascade
detector_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)
detector = cv2.CascadeClassifier(detector_path)

if detector.empty():
    raise RuntimeError("Khong the khoi tao Haar Cascade")

total_faces = 0

for image_path in IMAGE_PATHS:
    image = cv2.imread(str(image_path))

    if image is None:
        print("Khong doc duoc:", image_path)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
    )

    print(f"\nAnh: {image_path.name}")
    print("So khuon mat phat hien:", len(faces))

    # Lấy tất cả các khuôn mặt
    faces = sorted(faces, key=lambda f: (int(f[1]), int(f[0])))

    for i, (x, y, w, h) in enumerate(faces, start=1):
        # Tạo crop vuông, có vùng mở rộng
        cx = x + w / 2
        cy = y + h / 2
        side = int(max(w, h) * 1.5)

        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(image.shape[1], int(cx + side / 2))
        y2 = min(image.shape[0], int(cy + side / 2))

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        output_path = (
            OUTPUT_DIR / f"{image_path.stem}_face_{i}.png"
        )

        success = cv2.imwrite(str(output_path), crop)

        if success:
            total_faces += 1
            print(
                f"  Face {i}: "
                f"{crop.shape[1]}x{crop.shape[0]} "
                f"-> {output_path.name}"
            )

print("\nTong so khuon mat da luu:", total_faces)
print("Thu muc:", OUTPUT_DIR)
print("=== PREPARE MULTIPLE FACES FINISHED ===")

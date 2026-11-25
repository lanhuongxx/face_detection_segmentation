
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.segmentation.unet import FaceSegmenter

# Cấu hình thực nghiệm
SIZES = [128, 192, 256]
THRESHOLD = 0.5

input_path = Path("benchmark/outputs/face_crop.jpg")
output_dir = Path("benchmark/outputs/resize_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

# Đọc ảnh RGB
image_bgr = cv2.imread(str(input_path))
if image_bgr is None:
    raise FileNotFoundError(input_path)

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]

# Nạp mô hình U-Net đã huấn luyện
segmenter = FaceSegmenter(
    weights="models/unet_best.pt",
    device="cuda",
    batch_size=1,
)

masks = {}

for size in SIZES:
    # Resize ảnh trước khi dự đoán
    resized = cv2.resize(
        image_rgb,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )

    batch = np.array(resized[None, ...], copy=True)

    # Dự đoán mask xác suất
    prob = segmenter.predict(batch)[0]

    # Đưa mask xác suất về kích thước ảnh gốc
    prob_full = cv2.resize(
        prob,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )

    binary = (prob_full >= THRESHOLD).astype(np.uint8)
    masks[size] = binary

    # Tạo ảnh overlay màu xanh để quan sát
    overlay = image_bgr.copy()
    color = np.zeros_like(overlay)
    color[:, :] = (0, 255, 0)

    region = binary.astype(bool)
    overlay[region] = (
        0.6 * overlay[region] + 0.4 * color[region]
    ).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / f"mask_{size}.png"),
        binary * 255,
    )
    cv2.imwrite(
        str(output_dir / f"overlay_{size}.png"),
        overlay,
    )

    print(
        f"Size {size}: "
        f"mask_pixels={int(binary.sum())}, "
        f"coverage={binary.mean() * 100:.2f}%"
    )


def dice_score(a, b):
    intersection = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return (
        1.0 if total == 0
        else (2.0 * intersection) / total
    )


def iou_score(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return (
        1.0 if union == 0
        else intersection / union
    )


# Dùng 256 làm mốc tham chiếu
reference = masks[256]

print("\n=== SO SANH VOI MASK 256 ===")

for size in [128, 192]:
    dice = dice_score(masks[size], reference)
    iou = iou_score(masks[size], reference)

    print(
        f"{size} vs 256: "
        f"Dice={dice:.4f}, IoU={iou:.4f}"
    )

# Ghép 3 ảnh để so sánh trực quan
panels = []

for size in SIZES:
    panel = cv2.imread(
        str(output_dir / f"overlay_{size}.png")
    )
    cv2.putText(
        panel, f"{size}x{size}",
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 1,
    )
    panels.append(panel)

comparison = np.hstack(panels)

cv2.imwrite(
    str(output_dir / "comparison.png"),
    comparison,
)

print("Da luu ket qua tai:", output_dir)
print("=== COMPARE RESIZE FINISHED ===")

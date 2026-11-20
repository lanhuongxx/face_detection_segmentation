"""RetinaFace and U-Net demo for group photos."""
import argparse
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
import torch
from facexlib.detection import init_detection_model
from PIL import Image, ImageOps

from segmentation.unet import UNet

PROJECT = Path(__file__).resolve().parents[1]
SIZE = 256


def clean_mask(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


@torch.inference_mode()
def segment(original, detector, model, device, output_path):
    width, height = original.size
    bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    combined = np.zeros((height, width), dtype=bool)
    faces = detector.detect_faces(bgr, 0.8)
    for face_id, face in enumerate(faces, start=1):
        x1, y1, x2, y2 = map(float, face[:4])
        side = ceil(max(x2 - x1, y2 - y1) * 1.5)
        if side <= 0:
            continue
        left = floor((x1 + x2 - side) / 2)
        top = floor((y1 + y2 - side) / 2)
        right, bottom = left + side, top + side
        cl, ct = max(0, left), max(0, top)
        cr, cb = min(width, right), min(height, bottom)
        if cl >= cr or ct >= cb:
            continue

        crop = original.crop((left, top, right, bottom))
        crop = crop.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        pixels = np.asarray(crop, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)

        # Logit >= 0 tương ứng xác suất >= 0.5.
        probability = model(tensor)[0, 0].sigmoid()
        raw_mask = (probability >= 0.55).cpu().numpy().astype(np.uint8)
        mask = Image.fromarray(clean_mask(raw_mask) * 255)
        green = Image.new("RGB", crop.size, (0, 255, 90))
        overlay = Image.composite(Image.blend(crop, green, 0.45), crop, mask)
        overlay.save(output_path.with_name(f"{output_path.stem}_face_{face_id}_crop.png"))

        mapped = np.asarray(mask.resize((side, side), Image.Resampling.NEAREST)) > 0
        combined[ct:cb, cl:cr] |= mapped[ct-top:cb-top, cl-left:cr-left]

    tint = Image.new("RGB", original.size, (0, 255, 90))
    merged_mask = Image.fromarray(combined.astype(np.uint8) * 255)
    result = Image.composite(Image.blend(original, tint, 0.45), original, merged_mask)
    result.save(output_path)
    return len(faces)


def main():
    parser = argparse.ArgumentParser(description="RetinaFace + U-Net face segmentation")
    parser.add_argument("--image", type=Path, default=Path("data/demo/images2.jpg"))
    parser.add_argument("--checkpoint", type=Path, help="C:\\Users\\NAO\\face_detection_segmentation\\models\\unet_runs\\20261120_032415_232010\\unet_best.pt")
    args = parser.parse_args()
    image_path = PROJECT / args.image
    checkpoint = PROJECT / args.checkpoint if args.checkpoint else None
    if checkpoint is None:
        candidates = sorted((PROJECT / "models/unet_runs").glob("20261120_032415*/unet_best.pt"))
        if len(candidates) != 1:
            parser.error("Hãy chỉ định file trọng số bằng --checkpoint.")
        checkpoint = candidates[0]
    for path in (image_path, checkpoint):
        if not path.is_file():
            parser.error(f"Không tìm thấy file: {path}")

    with Image.open(image_path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    detector = init_detection_model("retinaface_mobile0.25", half=False)
    output_path = image_path.with_name(f"{image_path.stem}_masks_t055.png")
    count = segment(original, detector, model, device, output_path)
    print(f"Checkpoint: {checkpoint}")
    print(f"Số mặt phát hiện: {count}")
    print(f"Ảnh kết quả: {output_path}")


if __name__ == "__main__":
    main()

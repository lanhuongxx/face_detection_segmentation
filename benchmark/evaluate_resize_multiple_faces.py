
from pathlib import Path
import csv
import time

import cv2
import numpy as np
import torch

from src.segmentation.unet import FaceSegmenter

INPUT_DIR = Path("benchmark/outputs/multiple_faces")
OUTPUT_DIR = Path("benchmark/outputs/multiple_faces_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [192, 256]
THRESHOLD = 0.5
WARMUP = 5
REPEATS = 20

files = sorted(INPUT_DIR.glob("*.png"))

if not files:
    raise FileNotFoundError("Khong tim thay anh crop!")

segmenter = FaceSegmenter(
    weights="models/unet_best.pt",
    device="cuda",
    batch_size=1,
)


def measure_predict(batch):
    # Chay khoi dong de on dinh phep do
    for _ in range(WARMUP):
        segmenter.predict(batch)

    times = []

    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        output = segmenter.predict(batch)

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return output[0], float(np.mean(times))


def scores(a, b):
    a = a.astype(bool)
    b = b.astype(bool)

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    total = a.sum() + b.sum()

    if union == 0:
        return np.nan, np.nan

    return 2 * intersection / total, intersection / union


rows = []
montage_rows = []

print("=== EVALUATE RESIZE ON MULTIPLE FACES ===")
print("GPU:", torch.cuda.get_device_name(0))
print("Number of faces:", len(files))

for path in files:
    image = cv2.imread(str(path))

    if image is None:
        print("SKIP:", path.name)
        continue

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = {}
    latencies = {}
    overlays = {}

    for size in SIZES:
        resized = cv2.resize(rgb, (size, size))
        batch = np.array(resized[None, ...], copy=True)

        prob, latency = measure_predict(batch)

        prob_original = cv2.resize(
            prob, (w, h), interpolation=cv2.INTER_LINEAR
        )

        mask = (prob_original >= THRESHOLD).astype(np.uint8)

        masks[size] = mask
        latencies[size] = latency

        overlay = image.copy()
        selected = mask.astype(bool)
        green = np.array([0, 255, 0], dtype=np.float32)

        overlay[selected] = (
            0.6 * overlay[selected].astype(np.float32)
            + 0.4 * green
        ).astype(np.uint8)

        overlays[size] = overlay

    dice, iou = scores(masks[192], masks[256])

    coverage_192 = float(masks[192].mean() * 100)
    coverage_256 = float(masks[256].mean() * 100)

    rows.append({
        "image": path.name,
        "latency_192_ms": round(latencies[192], 3),
        "latency_256_ms": round(latencies[256], 3),
        "dice_vs_256": round(float(dice), 4),
        "iou_vs_256": round(float(iou), 4),
        "coverage_192_pct": round(coverage_192, 2),
        "coverage_256_pct": round(coverage_256, 2),
    })

    print(
        f"{path.name}: "
        f"Dice={dice:.4f}, IoU={iou:.4f}, "
        f"192={latencies[192]:.2f}ms, "
        f"256={latencies[256]:.2f}ms"
    )

    panels = []

    for label, panel in [
        ("Original", image),
        ("192x192", overlays[192]),
        ("256x256", overlays[256]),
    ]:
        panel = cv2.resize(panel, (256, 256))
        cv2.putText(
            panel, label, (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 0, 255), 2
        )
        panels.append(panel)

    montage_rows.append(np.hstack(panels))

csv_path = OUTPUT_DIR / "resize_results.csv"

with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

valid = [
    r for r in rows
    if np.isfinite(r["dice_vs_256"])
]

print("\n=== SUMMARY ===")
print("Processed:", len(rows))
print("Valid mask comparisons:", len(valid))

if valid:
    for metric in ["dice_vs_256", "iou_vs_256"]:
        values = [r[metric] for r in valid]
        print(f"Mean {metric}: {np.mean(values):.4f}")
        print(f"Min {metric}: {np.min(values):.4f}")

for size in SIZES:
    mean_ms = np.mean(
        [r[f"latency_{size}_ms"] for r in rows]
    )
    print(f"Mean latency {size}: {mean_ms:.3f} ms")

if montage_rows:
    comparison = np.vstack(montage_rows)
    cv2.imwrite(
        str(OUTPUT_DIR / "all_faces_comparison.png"),
        comparison,
    )

print("CSV:", csv_path)
print("=== EVALUATION FINISHED ===")

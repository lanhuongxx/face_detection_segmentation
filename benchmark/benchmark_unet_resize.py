
import time
import numpy as np
import torch
from PIL import Image

from src.segmentation.unet import FaceSegmenter

DEVICE = "cuda"
SIZES = [128, 192, 256]
WARMUP = 5
REPEATS = 20

print("=== BENCHMARK U-NET RESIZE FP32 ===")
print("GPU:", torch.cuda.get_device_name(0))

segmenter = FaceSegmenter(
    weights="models/unet_best.pt",
    device=DEVICE,
    batch_size=1,
)

# Một ảnh gốc dùng chung cho tất cả kích thước
rng = np.random.default_rng(42)
original = rng.integers(
    0, 256, (256, 256, 3), dtype=np.uint8
)

for size in SIZES:
    print(f"\n--- SIZE {size} x {size} ---")

    resized = np.asarray(
        Image.fromarray(original).resize(
            (size, size),
            Image.Resampling.BILINEAR
        )
    )
    images = np.array(resized[None, ...], copy=True)

    # Warmup
    for _ in range(WARMUP):
        segmenter.predict(images)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []

    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        output = segmenter.predict(images)

        torch.cuda.synchronize()
        times.append(
            (time.perf_counter() - start) * 1000
        )

    avg = float(np.mean(times))

    print("Output shape:", output.shape)
    print("Average latency (ms):", round(avg, 3))
    print(
        "P50 latency (ms):",
        round(float(np.percentile(times, 50)), 3)
    )
    print(
        "P95 latency (ms):",
        round(float(np.percentile(times, 95)), 3)
    )
    print("Throughput (images/s):", round(1000 / avg, 3))
    print(
        "Peak VRAM allocated (MiB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            3
        )
    )

print("\n=== BENCHMARK RESIZE FINISHED ===")

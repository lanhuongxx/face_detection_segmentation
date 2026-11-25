
import time
import numpy as np
import torch

from src.segmentation.unet import FaceSegmenter

# Cấu hình thực nghiệm
DEVICE = "cuda"
BATCH_SIZE = 1
IMAGE_SIZE = 256
WARMUP = 5
REPEATS = 20

print("=== BENCHMARK U-NET FP32 ===")
print("GPU:", torch.cuda.get_device_name(0))

# Nạp checkpoint đã huấn luyện
segmenter = FaceSegmenter(
    weights="models/unet_best.pt",
    device=DEVICE,
    batch_size=BATCH_SIZE,
)

# Dữ liệu giả lập, dùng để đo tốc độ
rng = np.random.default_rng(42)
images = rng.integers(
    0, 256,
    size=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3),
    dtype=np.uint8,
)

# Khởi động trước khi đo
for _ in range(WARMUP):
    segmenter.predict(images)

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []

# Đo thời gian inference
for _ in range(REPEATS):
    torch.cuda.synchronize()
    start = time.perf_counter()

    output = segmenter.predict(images)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    times.append(elapsed_ms)

times = np.asarray(times)

avg_ms = float(np.mean(times))
p50_ms = float(np.percentile(times, 50))
p95_ms = float(np.percentile(times, 95))
fps = BATCH_SIZE * 1000 / avg_ms

peak_vram = (
    torch.cuda.max_memory_allocated() / (1024 ** 2)
)

print("Model:", segmenter.model_name)
print("Output shape:", output.shape)
print("Average latency (ms):", round(avg_ms, 3))
print("P50 latency (ms):", round(p50_ms, 3))
print("P95 latency (ms):", round(p95_ms, 3))
print("Throughput (images/s):", round(fps, 3))
print("Peak VRAM allocated (MiB):", round(peak_vram, 3))
print("=== BENCHMARK FINISHED ===")

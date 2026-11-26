
import time
import numpy as np
import torch

from src.segmentation.unet import FaceSegmenter

DEVICE = "cuda"
WARMUP = 5
REPEATS = 20

print("=== BENCHMARK U-NET FP16 ===")
print("GPU:", torch.cuda.get_device_name(0))

segmenter = FaceSegmenter(
    weights="models/unet_best.pt",
    device=DEVICE,
    batch_size=1,
)

# Dung cung seed va du lieu voi benchmark FP32
rng = np.random.default_rng(42)
images = rng.integers(
    0, 256,
    size=(1, 256, 256, 3),
    dtype=np.uint8,
)

# Ket qua goc FP32 de so sanh
torch.cuda.synchronize()
output_fp32 = segmenter.predict(images)

# Autocast FP16, khong sua model goc
def predict_fp16():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        return segmenter.predict(images)

try:
    for _ in range(WARMUP):
        predict_fp16()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()

        output_fp16 = predict_fp16()

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.asarray(times)
    avg = float(times.mean())

    print("Model:", segmenter.model_name)
    print("Precision: FP16 autocast")
    print("Output shape:", output_fp16.shape)
    print("Average latency (ms):", round(avg, 3))
    print("P50 latency (ms):", round(np.percentile(times, 50), 3))
    print("P95 latency (ms):", round(np.percentile(times, 95), 3))
    print("Throughput (images/s):", round(1000 / avg, 3))
    print(
        "Peak VRAM allocated (MiB):",
        round(torch.cuda.max_memory_allocated() / 1024**2, 3)
    )

    diff = np.abs(output_fp32 - output_fp16)
    print("Mean absolute difference:", float(diff.mean()))
    print("Max absolute difference:", float(diff.max()))
    print("All finite:", bool(np.isfinite(output_fp16).all()))

    print("=== BENCHMARK FP16 FINISHED ===")

except (RuntimeError, NotImplementedError) as error:
    print("FP16 KHONG TUONG THICH:", str(error))

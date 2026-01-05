#!/usr/bin/env python3

import sys
import subprocess
import time

# -----------------------------
# AUTO-INSTALL DEPENDENCIES
# -----------------------------
def ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

ensure("torch")
ensure("numpy")

import torch

# -----------------------------
# GPU DETECTION
# -----------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

gpu_name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024**3)

if "1080" in gpu_name:
    dtype = torch.float32
    print("GTX 1080 Ti detected → FP32 (no tensor cores)")
elif "3060" in gpu_name:
    dtype = torch.float16
    print("RTX 3060 Ti detected → FP16 + tensor cores")
else:
    dtype = torch.float32
    print("Unknown GPU → FP32 fallback")

device = "cuda"

print(f"GPU: {gpu_name}")
print(f"VRAM: {vram_gb:.1f} GB")
print(f"dtype: {dtype}")

# -----------------------------
# BENCH CONFIG (STRESSFUL)
# -----------------------------
batch_size = 4          # increase if VRAM allows
steps = 30              # diffusion steps
blocks = 12             # UNet depth
channels = 768         # SDXL-class width

resolutions = [
    (512, 512),
    (1024, 1024)
]

# -----------------------------
# HEAVY DIFFUSION BLOCK
# -----------------------------
def heavy_block(x, w1, w2, w3):
    x = torch.matmul(x, w1)
    x = torch.nn.functional.silu(x)
    x = torch.matmul(x, w2)
    x = torch.nn.functional.silu(x)
    x = torch.matmul(x, w3)
    return x

# -----------------------------
# BENCHMARK
# -----------------------------
def run_benchmark(w, h):
    print(f"\n--- {w}x{h} ---")

    tokens = (w // 8) * (h // 8)

    x = torch.randn(
        batch_size,
        tokens,
        channels,
        device=device,
        dtype=dtype,
    )

    weights = [
        (
            torch.randn(channels, channels * 4, device=device, dtype=dtype),
            torch.randn(channels * 4, channels * 4, device=device, dtype=dtype),
            torch.randn(channels * 4, channels, device=device, dtype=dtype),
        )
        for _ in range(blocks)
    ]

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(steps):
        for w1, w2, w3 in weights:
            x = heavy_block(x, w1, w2, w3)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    mp = (w * h) / 1_000_000
    print(f"Total time: {elapsed:.2f} s")
    print(f"Time / step: {elapsed / steps:.3f} s")
    print(f"Time / MP: {elapsed / mp:.2f} s")

# -----------------------------
# RUN
# -----------------------------
for w, h in resolutions:
    run_benchmark(w, h)

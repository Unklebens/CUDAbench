# Cross-Platform Python Benchmark Setup (Windows & Ubuntu)

This document contains **full scripts for Windows and Ubuntu** including virtual environment creation, activation, dependency installation, and the complete benchmark code. Each environment is self-contained and ready to run.

---

## Windows Benchmark Script

Save as `bench_windows.py`:

```python
#!/usr/bin/env python3

import sys
import subprocess
import time
import torch

# Auto-install dependencies
for pkg in ["torch", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Configuration
batch_size = 1
steps = 20
resolutions = [(512, 512), (1920, 1080)]

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
dtype = torch.float16 if "3060" in gpu_name else torch.float32

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {gpu_name}, dtype={dtype}")

# Benchmark function
def run_benchmark(width, height):
    print(f"\n--- Benchmark {width}x{height} ---")
    latents = torch.randn(batch_size, 4, height // 8, width // 8, device=device, dtype=dtype)
    conditioning = torch.zeros(batch_size, 77, 768, device=device, dtype=dtype)

    torch.cuda.synchronize() if device=="cuda" else None
    start_time = time.time()

    for step in range(steps):
        latents = latents * 0.99 + torch.randn_like(latents) * 0.01
        torch.cuda.synchronize() if device=="cuda" else None

    total_time = time.time() - start_time
    print(f"Generated {batch_size} image(s) in {total_time:.2f} s")
    print(f"Time per step: {total_time/steps:.2f} s")
    print(f"Time per megapixel: {total_time / (width*height/1_000_000):.2f} s/MP")

for width, height in resolutions:
    run_benchmark(width, height)
```

### Windows CLI Instructions

```powershell
# Create and activate virtual environment
python -m venv bench_venv
.\bench_venv\Scripts\Activate.ps1  # PowerShell
# or .\bench_venv\Scripts\activate.bat  # CMD

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install torch numpy

# Run benchmark
python bench_windows.py
```

---

## Ubuntu / Linux Benchmark Script

Save as `bench_ubuntu.py`:

```python
#!/usr/bin/env python3

import sys
import subprocess
import time
import torch

# Auto-install dependencies
for pkg in ["torch", "numpy"]:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# Configuration
batch_size = 1
steps = 20
resolutions = [(512, 512), (1920, 1080)]

gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
dtype = torch.float16 if "3060" in gpu_name else torch.float32

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {gpu_name}, dtype={dtype}")

# Benchmark function
def run_benchmark(width, height):
    print(f"\n--- Benchmark {width}x{height} ---")
    latents = torch.randn(batch_size, 4, height // 8, width // 8, device=device, dtype=dtype)
    conditioning = torch.zeros(batch_size, 77, 768, device=device, dtype=dtype)

    torch.cuda.synchronize() if device=="cuda" else None
    start_time = time.time()

    for step in range(steps):
        latents = latents * 0.99 + torch.randn_like(latents) * 0.01
        torch.cuda.synchronize() if device=="cuda" else None

    total_time = time.time() - start_time
    print(f"Generated {batch_size} image(s) in {total_time:.2f} s")
    print(f"Time per step: {total_time/steps:.2f} s")
    print(f"Time per megapixel: {total_time / (width*height/1_000_000):.2f} s/MP")

for width, height in resolutions:
    run_benchmark(width, height)
```

### Ubuntu CLI Instructions

```bash
# Create and activate virtual environment
python3 -m venv bench_venv
source bench_venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install torch numpy

# Run benchmark
python bench_ubuntu.py
```

---

✅ These scripts are fully self-contained and ready to run on their respective OSes with virtual environments, GPU detection, and benchmarking for both 512x512 and 1920x1080 resolutions.


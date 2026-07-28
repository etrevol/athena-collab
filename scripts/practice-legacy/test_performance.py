#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки GPU та CPU продуктивності
"""
import sys
import time
import numpy as np
from multiprocessing import cpu_count

print("="*70)
print("PERFORMANCE TEST FOR ATHENA VISUALIZATION")
print("="*70)

# 1. CPU Info
print("\n[1] CPU Information:")
num_cores = cpu_count()
print(f"    Total CPU cores: {num_cores}")
print(f"    Recommended --num_workers: {num_cores - 2} to {num_cores}")
print(f"    (leave 1-2 cores for system)")

# 2. GPU Test
print("\n[2] GPU Test:")
try:
    import cupy as cp
    print(f"    ✓ CuPy version: {cp.__version__}")
    
    # Get GPU info
    device = cp.cuda.Device(0)
    props = device.attributes
    print(f"    ✓ GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"    ✓ Compute capability: {device.compute_capability}")
    print(f"    ✓ Total memory: {cp.cuda.Device().mem_info[1] / 1024**3:.1f} GB")
    
    # Test GPU speed
    print("\n    Testing GPU performance...")
    size = 5000
    a_gpu = cp.random.random((size, size), dtype=cp.float32)
    b_gpu = cp.random.random((size, size), dtype=cp.float32)
    
    # Warm up
    _ = cp.mean(a_gpu)
    
    # Benchmark
    start = time.time()
    for _ in range(10):
        result = cp.mean(a_gpu * b_gpu)
        result += cp.std(b_gpu)
    cp.cuda.Stream.null.synchronize()
    gpu_time = (time.time() - start) / 10
    
    # CPU comparison
    a_cpu = cp.asnumpy(a_gpu)
    b_cpu = cp.asnumpy(b_gpu)
    start = time.time()
    for _ in range(10):
        result = np.mean(a_cpu * b_cpu)
        result += np.std(b_cpu)
    cpu_time = (time.time() - start) / 10
    
    print(f"    GPU time: {gpu_time*1000:.1f} ms")
    print(f"    CPU time: {cpu_time*1000:.1f} ms")
    print(f"    Speedup: {cpu_time/gpu_time:.1f}x faster with GPU")
    
    if cpu_time/gpu_time > 2:
        print("    ✓ GPU acceleration working well! Use --use_gpu")
    else:
        print("    ⚠ GPU not much faster, CPU might be better for small datasets")
    
except ImportError as e:
    print(f"    ✗ CuPy not installed: {e}")
    print("    Install with: pip install cupy-cuda12x")
    print("    GPU acceleration will NOT be available")
except Exception as e:
    print(f"    ✗ GPU test failed: {e}")

# 3. Memory test
print("\n[3] Memory Test:")
try:
    import psutil
    mem = psutil.virtual_memory()
    print(f"    Total RAM: {mem.total / 1024**3:.1f} GB")
    print(f"    Available: {mem.available / 1024**3:.1f} GB")
    print(f"    Used: {mem.percent}%")
except ImportError:
    print("    Install psutil for memory info: pip install psutil")

# 4. Recommendations
print("\n[4] Recommendations:")
print(f"    For CPU parallelism:")
print(f"      --num_workers {max(1, num_cores - 2)}  (safe, leaves headroom)")
print(f"      --num_workers {num_cores}              (maximum)")
print()
print(f"    For GPU acceleration:")
try:
    import cupy as cp
    print(f"      --use_gpu                           (recommended!)")
    print(f"      --use_gpu --num_workers 4           (GPU + parallel rendering)")
except:
    print(f"      Not available (CuPy not installed)")

print("\n[5] Test Commands:")
print("    # Test with CPU only:")
print("    python3 scripts/vis2d.py --mode heatmaps --frame 0")
print()
print("    # Test with GPU:")
print("    python3 scripts/vis2d.py --mode heatmaps --frame 0 --use_gpu")
print()
print("    # Test animation with parallel workers:")
print(f"    python3 scripts/vis2d.py --mode animation --fps 10 --num_workers {max(4, num_cores//2)} --subsample 5")
print()
print("    # Full power (GPU + parallel):")
print(f"    python3 scripts/vis2d.py --mode all --use_gpu --num_workers {max(4, num_cores//2)}")

print("\n" + "="*70)
print("Run this from your results/*/materials/ directory!")
print("="*70)

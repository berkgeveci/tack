"""20 -- Running the same kernel on multiple backends.

One of Tack's key features: write your kernel once, run it on CPU,
Metal, CUDA, or HIP.  This example detects available backends and
runs the same computation on each, comparing results and performance.

Usage:
  uv run python examples/20_multi_backend.py
"""

import time

import numpy as np

import tack


def detect_backends():
    """Return list of available backend names."""
    backends = ["cpu"]
    try:
        import Metal
        if Metal.MTLCreateSystemDefaultDevice() is not None:
            backends.append("metal")
    except ImportError:
        pass
    try:
        from cuda.bindings import driver
        driver.cuInit(0)
        err, dev = driver.cuDeviceGet(0)
        if err == driver.CUresult.CUDA_SUCCESS:
            backends.append("cuda")
    except (ImportError, Exception):
        pass
    try:
        from hip import hip as hipmodule
        hipmodule.hipInit(0)
        err, count = hipmodule.hipGetDeviceCount()
        if count > 0:
            backends.append("hip")
    except (ImportError, Exception):
        pass
    return backends


@tack.kernel
def saxpy(x, y, out, alpha, n):
    for i in range(n):
        out[i] = alpha * x[i] + y[i]


@tack.kernel
def dot_product(x, y, result):
    for i in range(x.shape[0]):
        tack.atomic_add(result, 0, x[i] * y[i])


BACKENDS = detect_backends()
N = 500_000
ALPHA = 2.5

np.random.seed(42)
np_x = np.random.randn(N).astype(np.float32)
np_y = np.random.randn(N).astype(np.float32)

print(f"Available backends: {BACKENDS}")
print(f"\n{'Backend':>8s} {'SAXPY (ms)':>12s} {'Dot (ms)':>10s} {'SAXPY err':>12s} {'Dot err':>12s}")
print("-" * 60)

for backend in BACKENDS:
    tack.init(arch=backend)

    x = tack.field(dtype=tack.f32, shape=(N,))
    y = tack.field(dtype=tack.f32, shape=(N,))
    out = tack.field(dtype=tack.f32, shape=(N,))
    dot_result = tack.field(dtype=tack.f32, shape=(1,))

    x.from_numpy(np_x)
    y.from_numpy(np_y)

    # Warm up
    saxpy(x, y, out, ALPHA, N)
    dot_result.fill(0.0)
    dot_product(x, y, dot_result)

    # Benchmark SAXPY
    times_saxpy = []
    for _ in range(5):
        t0 = time.perf_counter()
        saxpy(x, y, out, ALPHA, N)
        times_saxpy.append(time.perf_counter() - t0)

    # Benchmark dot product
    times_dot = []
    for _ in range(5):
        dot_result.fill(0.0)
        t0 = time.perf_counter()
        dot_product(x, y, dot_result)
        times_dot.append(time.perf_counter() - t0)

    # Verify
    saxpy_err = np.max(np.abs(out.to_numpy() - (ALPHA * np_x + np_y)))
    dot_err = abs(dot_result.to_numpy()[0] - np.dot(np_x, np_y))

    print(f"{backend:>8s} {min(times_saxpy)*1000:>12.3f} {min(times_dot)*1000:>10.3f} "
          f"{saxpy_err:>12.2e} {dot_err:>12.2e}")

print("\nSame kernels, same results, different hardware!")

"""15 -- Matrix multiplication.

Compute C = A @ B for square matrices stored as flat 1D arrays.
Uses 2D indexing with manual row-major layout.

Usage:
  uv run python examples/15_matmul.py
"""

import argparse
import time

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = _parser.parse_args().arch
tack.set_default_device(_arch)

N = 128


@tack.kernel
def matmul(a, b, c, n):
    """Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]"""
    for i in range(n):
        for j in range(128):
            s = 0.0
            for k in range(128):
                s = s + a[i * 128 + k] * b[k * 128 + j]
            c[i * 128 + j] = s


a = tack.empty((N * N,))
b = tack.empty((N * N,))
c = tack.empty((N * N,))

np.random.seed(42)
np_a = np.random.randn(N, N).astype(np.float32)
np_b = np.random.randn(N, N).astype(np.float32)
tack.copyto(a, np_a.ravel())
tack.copyto(b, np_b.ravel())

# Warm up (compile)
matmul(a, b, c, N)

# Benchmark
t0 = time.perf_counter()
matmul(a, b, c, N)
dt = time.perf_counter() - t0

result = c.numpy().reshape(N, N)
expected = np_a @ np_b
max_err = np.max(np.abs(result - expected))

gflops = 2 * N**3 / dt / 1e9

print(f"Matrix multiply: {N}x{N}")
print(f"  Time:      {dt*1000:.2f} ms")
print(f"  GFLOPS:    {gflops:.2f}")
print(f"  Max error: {max_err:.6f}")
assert max_err < 0.1, f"Error too large: {max_err}"
print("  Correctness: OK")

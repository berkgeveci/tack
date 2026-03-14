"""15 — Matrix multiplication.

Compute C = A @ B for square matrices stored as flat 1D fields.
Uses 2D indexing with manual row-major layout.

Usage:
  uv run python examples/15_matmul.py
"""

import time
import numpy as np
import pgc

pgc.init(arch=pgc.cpu)

N = 128


@pgc.kernel
def matmul(a, b, c, n):
    """Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]"""
    for i in range(n):
        for j in range(128):
            s = 0.0
            for k in range(128):
                s = s + a[i * 128 + k] * b[k * 128 + j]
            c[i * 128 + j] = s


a = pgc.field(dtype=pgc.f32, shape=(N * N,))
b = pgc.field(dtype=pgc.f32, shape=(N * N,))
c = pgc.field(dtype=pgc.f32, shape=(N * N,))

np.random.seed(42)
np_a = np.random.randn(N, N).astype(np.float32)
np_b = np.random.randn(N, N).astype(np.float32)
a.from_numpy(np_a.ravel())
b.from_numpy(np_b.ravel())

# Warm up (compile)
matmul(a, b, c, N)

# Benchmark
t0 = time.perf_counter()
matmul(a, b, c, N)
dt = time.perf_counter() - t0

result = c.to_numpy().reshape(N, N)
expected = np_a @ np_b
max_err = np.max(np.abs(result - expected))

gflops = 2 * N**3 / dt / 1e9

print(f"Matrix multiply: {N}x{N}")
print(f"  Time:      {dt*1000:.2f} ms")
print(f"  GFLOPS:    {gflops:.2f}")
print(f"  Max error: {max_err:.6f}")
assert max_err < 0.1, f"Error too large: {max_err}"
print("  Correctness: OK")

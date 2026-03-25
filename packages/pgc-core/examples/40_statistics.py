"""40 -- GPU-accelerated statistics and analysis.

Demonstrates pgc.algorithms statistics functions that run entirely
on the GPU using atomic operations — no host roundtrips.

Key APIs:
  field.mean()                      -- mean (GPU sum / size)
  algorithms.var(field)             -- population variance
  algorithms.std(field)             -- population standard deviation
  algorithms.norm(field, ord)       -- L1, L2, L-infinity norms
  algorithms.absmax(field)          -- max absolute value
  algorithms.count_nonzero(field)   -- count non-zero elements
  algorithms.dot(a, b)              -- dot product
  algorithms.histogram(field, bins) -- GPU histogram via atomics

Usage:
  uv run python packages/pgc-core/examples/40_statistics.py
  uv run python packages/pgc-core/examples/40_statistics.py --arch metal
"""

import numpy as np
import pgc
from pgc.algorithms import var, std, norm, absmax, count_nonzero, dot, histogram

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)


# ================================================================
# BASIC STATISTICS
# ================================================================

n = 100_000
np.random.seed(42)
data = pgc.field(dtype=pgc.f32, shape=(n,))
data.from_numpy(np.random.randn(n).astype(np.float32))

print(f"Data: {n:,} elements from standard normal distribution")
print(f"  sum  = {data.sum():.4f}")
print(f"  min  = {data.min():.4f}")
print(f"  max  = {data.max():.4f}")
print(f"  mean = {data.mean():.4f}")
print(f"  var  = {var(data):.4f}")
print(f"  std  = {std(data):.4f}")

# Verify against numpy
data_np = data.to_numpy()
assert abs(data.mean() - np.mean(data_np)) < 0.01
assert abs(var(data) - np.var(data_np)) < 0.01
print("  (matches numpy)")


# ================================================================
# NORMS
# ================================================================

print(f"\nNorms:")
print(f"  L1   = {norm(data, ord=1):.4f}")
print(f"  L2   = {norm(data, ord=2):.4f}")
print(f"  Linf = {norm(data, ord=float('inf')):.4f}")
print(f"  |max| = {absmax(data):.4f}")


# ================================================================
# DOT PRODUCT
# ================================================================

a = pgc.field(dtype=pgc.f32, shape=(1000,))
b = pgc.field(dtype=pgc.f32, shape=(1000,))
a.from_numpy(np.ones(1000, dtype=np.float32))
b.from_numpy(np.arange(1000, dtype=np.float32))

d = dot(a, b)
print(f"\nDot product of ones · [0..999] = {d:.0f}")
assert abs(d - 499500.0) < 1.0


# ================================================================
# COUNT NON-ZERO
# ================================================================

sparse = pgc.field(dtype=pgc.f32, shape=(10000,))
sparse_np = np.zeros(10000, dtype=np.float32)
sparse_np[::10] = 1.0  # every 10th element is 1
sparse.from_numpy(sparse_np)

nz = count_nonzero(sparse)
print(f"\nSparse field: {nz} non-zero out of {sparse.size}")
assert nz == 1000


# ================================================================
# HISTOGRAM
# ================================================================

print(f"\nHistogram of normal distribution ({n:,} samples, 20 bins):")
counts, edges = histogram(data, bins=20, range=(-4, 4))
counts_np = counts.to_numpy()

# Display ASCII histogram
max_count = int(counts_np.max())
for i in range(len(counts_np)):
    lo = edges[i]
    hi = edges[i + 1]
    bar_len = int(counts_np[i] / max_count * 40)
    bar = '#' * bar_len
    print(f"  [{lo:+5.1f}, {hi:+5.1f}) {counts_np[i]:5d} {bar}")

total_in_range = int(counts_np.sum())
print(f"  Total in range: {total_in_range:,} / {n:,}")


# ================================================================
# ANALYSIS WORKFLOW: residual norm
# ================================================================

print(f"\nResidual norm example:")
# Simulate solving Ax = b: compute residual r = b - Ax
x = pgc.field(dtype=pgc.f32, shape=(1000,))
b_field = pgc.field(dtype=pgc.f32, shape=(1000,))
residual = pgc.field(dtype=pgc.f32, shape=(1000,))

x.from_numpy(np.ones(1000, dtype=np.float32))
b_field.from_numpy(np.ones(1000, dtype=np.float32) * 1.001)

@pgc.kernel
def compute_residual(b, x, r):
    for i in range(b.shape[0]):
        r[i] = b[i] - x[i]

compute_residual(b_field, x, residual)

r_norm = norm(residual, ord=2)
r_max = absmax(residual)
print(f"  ||r||_2 = {r_norm:.6f}")
print(f"  ||r||_∞ = {r_max:.6f}")

print("\nAll statistics computed on GPU!")

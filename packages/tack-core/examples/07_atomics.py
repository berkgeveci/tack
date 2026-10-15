"""07 -- Atomic operations for thread-safe accumulation.

When multiple threads write to the same memory location, you need
atomic operations to avoid race conditions.

Available: tack.atomic_add, tack.atomic_min, tack.atomic_max

Usage:
  uv run python examples/07_atomics.py
"""

import argparse

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)

# --- Example 1: parallel sum with atomic_add ---

n = 10000
x = tack.empty((n,))
total = tack.empty((1,))

tack.copyto(x, np.ones(n, dtype=np.float32))
total.fill(0.0)


@tack.kernel
def parallel_sum(x, total):
    for i in range(x.shape[0]):
        tack.atomic_add(total, 0, x[i])


parallel_sum(x, total)
print(f"1. Parallel sum of {n} ones = {total.numpy()[0]:.0f}")
assert abs(total.numpy()[0] - n) < 1.0

# --- Example 2: find min and max in parallel ---

data = np.random.randn(n).astype(np.float32) * 100.0
tack.copyto(x, data)

min_val = tack.empty((1,))
max_val = tack.empty((1,))
tack.copyto(min_val, np.array([1e10], dtype=np.float32))
tack.copyto(max_val, np.array([-1e10], dtype=np.float32))


@tack.kernel
def find_extremes(x, min_val, max_val):
    for i in range(x.shape[0]):
        tack.atomic_min(min_val, 0, x[i])
        tack.atomic_max(max_val, 0, x[i])


find_extremes(x, min_val, max_val)
print(f"2. Data range: [{min_val.numpy()[0]:.2f}, {max_val.numpy()[0]:.2f}]")
print(f"   numpy ref:  [{data.min():.2f}, {data.max():.2f}]")

# --- Example 3: histogram ---

bins = tack.empty((10,))
bins.fill(0.0)

# Uniform data in [0, 1)
uniform = np.random.rand(n).astype(np.float32) * 0.9999
tack.copyto(x, uniform)


@tack.kernel
def histogram(x, bins):
    for i in range(x.shape[0]):
        bin_idx = int(x[i] * 10.0)
        tack.atomic_add(bins, bin_idx, 1.0)


histogram(x, bins)
counts = bins.numpy()
print(f"3. Histogram (10 bins, {n} samples):")
for i, c in enumerate(counts):
    bar = "#" * int(c / n * 100)
    print(f"   [{i*0.1:.1f}-{(i+1)*0.1:.1f}): {c:5.0f} {bar}")
print(f"   Total: {counts.sum():.0f}")

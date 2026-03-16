"""08 — Field reductions: sum, min, max.

Fields have built-in reduction methods.  On Metal, these run entirely
on the GPU using optimized threadgroup reduction kernels.  On other
backends, they fall back to efficient numpy reductions.

Usage:
  uv run python examples/08_reductions.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 100_000
x = pgc.field(dtype=pgc.f32, shape=(n,))

np.random.seed(42)
data = np.random.randn(n).astype(np.float32) * 50.0
x.from_numpy(data)

# Built-in reductions — no kernel needed
total = x.sum()
lo = x.min()
hi = x.max()

print(f"Field of {n:,} random values:")
print(f"  sum = {total:>12.2f}   (numpy: {data.sum():.2f})")
print(f"  min = {lo:>12.2f}   (numpy: {data.min():.2f})")
print(f"  max = {hi:>12.2f}   (numpy: {data.max():.2f})")

assert abs(total - data.sum()) < abs(data.sum()) * 1e-4
assert abs(lo - data.min()) < 0.01
assert abs(hi - data.max()) < 0.01
print("\nReductions match numpy!")

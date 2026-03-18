"""30 -- Scalar packing: many scalar parameters without hitting GPU limits.

GPU backends (especially Metal) have a limited number of buffer bindings
per kernel. PGC automatically packs scalar parameters into typed constant
buffers behind the scenes, so you can write kernels with as many scalar
arguments as you need.

This example passes 40+ scalar parameters to a kernel -- well beyond
Metal's 31 buffer binding limit -- and it works transparently on all
backends.

Usage:
  uv run python examples/30_scalar_packing.py
  uv run python examples/30_scalar_packing.py --arch metal
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero', 'wgpu'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 10000


# A kernel with many scalar parameters: 3 fields + 30 float scalars + 3 int scalars
# On Metal this would require 36 buffer bindings without packing (over the 31 limit).
# PGC automatically packs all 33 scalars into 2 constant buffers (one f32, one i32).
@pgc.kernel
def weighted_sum(out, a, b,
                 w0, w1, w2, w3, w4, w5, w6, w7, w8, w9,
                 w10, w11, w12, w13, w14, w15, w16, w17, w18, w19,
                 w20, w21, w22, w23, w24, w25, w26, w27, w28, w29,
                 offset, stride, count):
    """Compute out[i] = sum(weights) * a[i] + offset, strided."""
    for i in range(count):
        idx = i * stride
        total_w = (w0 + w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9
                   + w10 + w11 + w12 + w13 + w14 + w15 + w16 + w17 + w18 + w19
                   + w20 + w21 + w22 + w23 + w24 + w25 + w26 + w27 + w28 + w29)
        out[idx] = total_w * a[idx] + b[idx] + offset


# Set up fields
out = pgc.field(dtype=pgc.f32, shape=(n,))
a = pgc.field(dtype=pgc.f32, shape=(n,))
b = pgc.field(dtype=pgc.f32, shape=(n,))

a_np = np.arange(n, dtype=np.float32)
b_np = np.ones(n, dtype=np.float32) * 0.5
a.from_numpy(a_np)
b.from_numpy(b_np)

# 30 float weights + 3 int scalars = 33 scalar parameters
weights = [float(i) * 0.1 for i in range(30)]
offset = 100.0
stride = 1
count = n

# Call the kernel -- all 33 scalars are packed automatically
weighted_sum(out, a, b, *weights, offset, stride, count)

result = out.to_numpy()
total_w = sum(weights)
expected = total_w * a_np + b_np + offset
assert np.allclose(result, expected, rtol=1e-4), f"Mismatch: max err = {np.max(np.abs(result - expected))}"
print(f"Total weight: {total_w:.1f}")
print(f"First 5 results: {result[:5]}")
print(f"Expected:        {expected[:5]}")
print(f"\n33 scalar parameters packed automatically -- all correct!")

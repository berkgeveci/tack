"""06 — Device functions with @pgc.func.

Use @pgc.func to define helper functions that are inlined into kernels
at compile time.  This keeps kernels clean and lets you reuse logic.

Usage:
  uv run python examples/06_device_functions.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)


# --- Device function: inlined wherever called ---

@pgc.func
def lerp(a, b, t):
    """Linear interpolation: a*(1-t) + b*t"""
    return a + t * (b - a)


@pgc.func
def smoothstep(edge0, edge1, x):
    """Hermite smoothstep interpolation."""
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@pgc.func
def remap(value, in_lo, in_hi, out_lo, out_hi):
    """Remap a value from one range to another."""
    t = (value - in_lo) / (in_hi - in_lo)
    return lerp(out_lo, out_hi, t)  # device functions can call each other


# --- Kernel using device functions ---

n = 100
x = pgc.field(dtype=pgc.f32, shape=(n,))
out_lerp = pgc.field(dtype=pgc.f32, shape=(n,))
out_smooth = pgc.field(dtype=pgc.f32, shape=(n,))
out_remap = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.linspace(0.0, 1.0, n, dtype=np.float32))


@pgc.kernel
def apply_all(x, out_lerp, out_smooth, out_remap):
    for i in range(x.shape[0]):
        t = x[i]
        out_lerp[i] = lerp(0.0, 100.0, t)
        out_smooth[i] = smoothstep(0.2, 0.8, t)
        out_remap[i] = remap(t, 0.0, 1.0, -10.0, 10.0)


apply_all(x, out_lerp, out_smooth, out_remap)

print("lerp(0,100,t):  ", out_lerp.to_numpy()[[0, 25, 50, 75, 99]])
print("smoothstep:     ", out_smooth.to_numpy()[[0, 25, 50, 75, 99]])
print("remap(0,1,-10,10):", out_remap.to_numpy()[[0, 25, 50, 75, 99]])

# Verify lerp
assert np.allclose(out_lerp.to_numpy(), np.linspace(0, 100, n))
# Verify remap
assert np.allclose(out_remap.to_numpy(), np.linspace(-10, 10, n))
print("\nDevice functions: OK")

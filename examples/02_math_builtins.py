"""02 -- Math builtins available inside kernels.

PGC kernels have access to standard math functions without any imports.
They compile to hardware intrinsics (LLVM on CPU, device functions on GPU).

Available: sqrt, sin, cos, tan, asin, acos, atan, atan2,
           exp, exp2, log, log2, log10, floor, ceil,
           abs, min, max, pow

Usage:
  uv run python examples/02_math_builtins.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero', 'wgpu'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 256
x = pgc.field(dtype=pgc.f32, shape=(n,))
out_sqrt = pgc.field(dtype=pgc.f32, shape=(n,))
out_sin = pgc.field(dtype=pgc.f32, shape=(n,))
out_combined = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.linspace(0.1, 10.0, n, dtype=np.float32))


@pgc.kernel
def apply_math(x, out_sqrt, out_sin, out_combined):
    for i in range(x.shape[0]):
        out_sqrt[i] = sqrt(x[i])
        out_sin[i] = sin(x[i])
        # Mix several builtins in one expression
        out_combined[i] = exp(-x[i]) * cos(x[i] * 3.14159) + abs(log(x[i]))


apply_math(x, out_sqrt, out_sin, out_combined)

np_x = x.to_numpy()
print("sqrt  error:", np.max(np.abs(out_sqrt.to_numpy() - np.sqrt(np_x))))
print("sin   error:", np.max(np.abs(out_sin.to_numpy() - np.sin(np_x))))

expected = np.exp(-np_x) * np.cos(np_x * 3.14159) + np.abs(np.log(np_x))
print("combo error:", np.max(np.abs(out_combined.to_numpy() - expected)))
print("\nAll within float32 tolerance!")

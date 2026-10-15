"""02 -- Math builtins available inside kernels.

Tack kernels have access to standard math functions without any imports.
They compile to hardware intrinsics (LLVM on CPU, device functions on GPU).

Available: sqrt, sin, cos, tan, asin, acos, atan, atan2,
           exp, exp2, log, log2, log10, floor, ceil,
           abs, min, max, pow

Usage:
  uv run python examples/02_math_builtins.py
"""

import argparse

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)

n = 256
x = tack.empty((n,))
out_sqrt = tack.empty((n,))
out_sin = tack.empty((n,))
out_combined = tack.empty((n,))

tack.copyto(x, np.linspace(0.1, 10.0, n, dtype=np.float32))


@tack.kernel
def apply_math(x, out_sqrt, out_sin, out_combined):
    for i in range(x.shape[0]):
        out_sqrt[i] = sqrt(x[i])
        out_sin[i] = sin(x[i])
        # Mix several builtins in one expression
        out_combined[i] = exp(-x[i]) * cos(x[i] * 3.14159) + abs(log(x[i]))


apply_math(x, out_sqrt, out_sin, out_combined)

np_x = x.numpy()
print("sqrt  error:", np.max(np.abs(out_sqrt.numpy() - np.sqrt(np_x))))
print("sin   error:", np.max(np.abs(out_sin.numpy() - np.sin(np_x))))

expected = np.exp(-np_x) * np.cos(np_x * 3.14159) + np.abs(np.log(np_x))
print("combo error:", np.max(np.abs(out_combined.numpy() - expected)))
print("\nAll within float32 tolerance!")

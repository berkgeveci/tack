"""03 — Scalar arguments: pass Python numbers directly to kernels.

Kernels can accept plain Python int/float values alongside fields.
This is useful for passing constants like time steps, coefficients,
or array sizes without creating a field.

Usage:
  uv run python examples/03_scalar_args.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

n = 1000
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

x.from_numpy(np.arange(n, dtype=np.float32))
y.from_numpy(np.ones(n, dtype=np.float32) * 3.0)


@pgc.kernel
def saxpy(x, y, out, alpha, n):
    """out = alpha * x + y  (SAXPY: Single-precision A*X Plus Y)"""
    for i in range(n):
        out[i] = alpha * x[i] + y[i]


# Pass scalar values directly — no need to wrap in fields
alpha = 2.5
saxpy(x, y, out, alpha, n)

result = out.to_numpy()
expected = alpha * np.arange(n, dtype=np.float32) + 3.0
assert np.allclose(result, expected)
print(f"SAXPY with alpha={alpha}: first 5 results = {result[:5]}")

# Call again with a different alpha — same compiled kernel is reused
saxpy(x, y, out, -1.0, n)
result2 = out.to_numpy()
expected2 = -1.0 * np.arange(n, dtype=np.float32) + 3.0
assert np.allclose(result2, expected2)
print(f"SAXPY with alpha=-1.0: first 5 results = {result2[:5]}")
print("\nScalar arguments work correctly!")

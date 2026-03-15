"""01 — Hello PGC: your first kernel.

This is the simplest possible PGC program.  It creates two arrays,
adds them element-wise on the GPU (or CPU), and checks the result.

Key concepts:
  - pgc.init()    — choose a backend (cpu, metal, cuda, hip)
  - pgc.field()   — allocate a device array
  - @pgc.kernel   — mark a function for GPU compilation
  - from_numpy / to_numpy — move data between host and device

Usage:
  uv run python examples/01_hello_pgc.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

# Allocate 1-D fields (device arrays)
n = 10
x = pgc.field(dtype=pgc.f32, shape=(n,))
y = pgc.field(dtype=pgc.f32, shape=(n,))
out = pgc.field(dtype=pgc.f32, shape=(n,))

# Upload data from numpy
x.from_numpy(np.arange(n, dtype=np.float32))        # [0, 1, 2, ..., 9]
y.from_numpy(np.ones(n, dtype=np.float32) * 10.0)   # [10, 10, ..., 10]


@pgc.kernel
def add(x, y, out):
    # The top-level for-range is parallelized across threads
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


# Run the kernel
add(x, y, out)

# Download result to numpy and print
result = out.to_numpy()
print("x  =", x.to_numpy())
print("y  =", y.to_numpy())
print("out=", result)
assert np.allclose(result, np.arange(n) + 10.0)
print("\nAll correct!")

"""01 -- Hello Tack: your first kernel.

This is the simplest possible Tack program.  It creates two arrays,
adds them element-wise on the GPU (or CPU), and checks the result.

Key concepts:
  - tack.set_default_device() -- choose a device (cpu, metal, cuda, hip)
  - tack.empty()   -- allocate a device array
  - @tack.kernel   -- mark a function for GPU compilation
  - from_numpy / to_numpy -- move data between host and device

Usage:
  uv run python examples/01_hello_tack.py
"""

import argparse

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = _parser.parse_args().arch
tack.set_default_device(_arch)

# Allocate 1-D arrays (device arrays)
n = 10
x = tack.empty((n,))
y = tack.empty((n,))
out = tack.empty((n,))

# Upload data from numpy
tack.copyto(x, np.arange(n, dtype=np.float32))        # [0, 1, 2, ..., 9]
tack.copyto(y, np.ones(n, dtype=np.float32) * 10.0)   # [10, 10, ..., 10]


@tack.kernel
def add(x, y, out):
    # The top-level for-range is parallelized across threads
    for i in range(x.shape[0]):
        out[i] = x[i] + y[i]


# Run the kernel
add(x, y, out)

# Download result to numpy and print
result = out.numpy()
print("x  =", x.numpy())
print("y  =", y.numpy())
print("out=", result)
assert np.allclose(result, np.arange(n) + 10.0)
print("\nAll correct!")

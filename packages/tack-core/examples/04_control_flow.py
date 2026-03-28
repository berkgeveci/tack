"""04 -- Control flow: if/else, while loops, break, ternary expressions.

Tack kernels support standard Python control flow.  The top-level for-range
is parallelized; everything inside runs per-thread.

Usage:
  uv run python examples/04_control_flow.py
"""

import numpy as np
import tack

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = getattr(tack, _parser.parse_args().arch)
tack.init(arch=_arch)

n = 256
x = tack.field(dtype=tack.f32, shape=(n,))
out = tack.field(dtype=tack.f32, shape=(n,))

data = np.arange(n, dtype=np.float32) - 128.0  # [-128, ..., 127]
x.from_numpy(data)

# --- Example 1: if/else (ReLU activation) ---

@tack.kernel
def relu(x, out):
    for i in range(x.shape[0]):
        if x[i] > 0.0:
            out[i] = x[i]
        else:
            out[i] = 0.0

relu(x, out)
assert np.allclose(out.to_numpy(), np.maximum(data, 0.0))
print("1. ReLU (if/else): OK")

# --- Example 2: while loop with early exit ---

@tack.kernel
def collatz_steps(x, out):
    """Count Collatz steps until reaching 1."""
    for i in range(x.shape[0]):
        val = abs(x[i]) + 1.0  # ensure positive, > 0
        steps = 0.0
        while val > 1.0:
            if val > 1000.0:
                break  # safety limit
            n_int = int(val)
            if n_int % 2 == 0:
                val = val / 2.0
            else:
                val = val * 3.0 + 1.0
            steps = steps + 1.0
        out[i] = steps

collatz_steps(x, out)
print("2. Collatz steps (while + break): OK, first 5 steps =", out.to_numpy()[:5])

# --- Example 3: ternary expression ---

@tack.kernel
def clamp(x, out):
    """Clamp values to [-50, 50] using ternary expressions."""
    for i in range(x.shape[0]):
        v = x[i]
        v = v if v > -50.0 else -50.0
        v = v if v < 50.0 else 50.0
        out[i] = v

clamp(x, out)
expected = np.clip(data, -50.0, 50.0)
assert np.allclose(out.to_numpy(), expected)
print("3. Clamp (ternary): OK")

# --- Example 4: nested loops ---

n2 = 16
mat = tack.field(dtype=tack.f32, shape=(n2 * n2,))

@tack.kernel
def fill_checkerboard(mat):
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                mat[i * 16 + j] = 1.0
            else:
                mat[i * 16 + j] = 0.0

fill_checkerboard(mat)
board = mat.to_numpy().reshape(16, 16)
print("4. Checkerboard (nested loops): OK")
print("   Top-left 4x4:")
for row in board[:4]:
    print("  ", " ".join(f"{v:.0f}" for v in row[:4]))

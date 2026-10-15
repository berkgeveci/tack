"""11 -- Template classes with @tack.device_class.

Use @tack.device_class to bundle arrays and scalar parameters into a
class that can be passed to kernels.  Array attributes become kernel
buffer parameters, scalar attributes become compile-time constants,
and methods marked with @tack.func are inlined.

This is Tack's way of writing reusable, parameterized compute objects.

Usage:
  uv run python examples/11_templates.py
"""

import argparse

import numpy as np

import tack

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_arch = _parser.parse_args().arch
tack.set_default_device(_arch)

n = 100


@tack.device_class
class Grid:
    """A 1D grid with uniform spacing and stored values."""

    def __init__(self, data, dx):
        self.data = data   # array -> becomes a kernel buffer parameter
        self.dx = dx       # scalar -> becomes a compile-time constant

    @tack.func
    def sample(self, i):
        """Read a grid value scaled by spacing."""
        return self.data[i] * self.dx

    @tack.func
    def gradient(self, i):
        """Central difference gradient."""
        return (self.data[i + 1] - self.data[i - 1]) / (2.0 * self.dx)


# Create a grid with some data
data = tack.empty((n,))
out = tack.empty((n,))
grad_out = tack.empty((n,))

# Fill with a smooth function: sin(x)
x_vals = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
tack.copyto(data, np.sin(x_vals))

dx = float(x_vals[1] - x_vals[0])
grid = Grid(data, dx)


@tack.kernel
def process(grid, out):
    """Sample every grid point."""
    for i in range(out.shape[0]):
        out[i] = grid.sample(i)


@tack.kernel
def compute_gradient(grid, grad_out):
    """Compute gradient at interior points."""
    for i in range(grad_out.shape[0]):
        if i > 0:
            if i < 99:
                grad_out[i] = grid.gradient(i)


process(grid, out)
compute_gradient(grid, grad_out)

# Verify: sample should be sin(x) * dx
sampled = out.numpy()
expected_sample = np.sin(x_vals) * dx
assert np.allclose(sampled, expected_sample, atol=1e-5)
print("Grid sampling: OK")

# Verify: gradient of sin(x) ~= cos(x)
grad = grad_out.numpy()
expected_grad = np.cos(x_vals)
interior = slice(2, n - 2)  # skip boundary effects
max_err = np.max(np.abs(grad[interior] - expected_grad[interior]))
print(f"Gradient of sin(x): max error = {max_err:.6f}")
print("Template classes: OK")

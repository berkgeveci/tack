"""14 — Jacobi iterative solver for the 2D Laplace equation.

Solves ∇²u = 0 on a 2D grid with fixed boundary conditions using
the Jacobi iteration method.  This is a classic stencil computation.

The grid has u=100 on the top edge and u=0 on the other three edges.
After convergence, the solution smoothly interpolates from bottom to top.

Usage:
  uv run python examples/14_jacobi_solver.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

N = 64  # grid size
STEPS = 10000


@pgc.kernel
def jacobi_step(src, dst, n):
    """One Jacobi iteration: dst[i,j] = average of 4 neighbors in src."""
    for i, j in pgc.ndrange(n, n):
        if i > 0 and i < n - 1 and j > 0 and j < n - 1:
            dst[i, j] = 0.25 * (
                src[i - 1, j] + src[i + 1, j] +
                src[i, j - 1] + src[i, j + 1]
            )


@pgc.kernel
def copy_field(src, dst, n):
    """Copy src into dst."""
    for i, j in pgc.ndrange(n, n):
        dst[i, j] = src[i, j]


# Create two fields for ping-pong iteration
u = pgc.field(dtype=pgc.f32, shape=(N, N))
u_new = pgc.field(dtype=pgc.f32, shape=(N, N))

# Boundary conditions: top edge = 100, rest = 0
init = np.zeros((N, N), dtype=np.float32)
init[-1, :] = 100.0  # top edge
u.from_numpy(init)
u_new.from_numpy(init)

print(f"Jacobi solver: {N}x{N} grid, {STEPS} iterations")

for step in range(STEPS):
    jacobi_step(u, u_new, N)
    copy_field(u_new, u, N)

    if (step + 1) % 2000 == 0:
        result = u.to_numpy().reshape(N, N)
        center = result[N // 2, N // 2]
        print(f"  step {step+1:>5d}: center = {center:.4f}")

result = u.to_numpy().reshape(N, N)
print(f"\nFinal solution:")
print(f"  Bottom edge (should be 0):    {result[0, N//2]:.4f}")
print(f"  Center:                       {result[N//2, N//2]:.4f}")
print(f"  Top edge    (should be 100):  {result[-1, N//2]:.4f}")

# With only the top edge at 100 and the other three at 0, the analytic
# center value is about 25 (not 50 — this isn't a 1D problem).
assert 20.0 < result[N // 2, N // 2] < 30.0, "Center should be near 25"

# Try to save image
try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(result, origin="lower", cmap="hot", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Temperature")
    ax.set_title(f"Laplace equation (Jacobi, {STEPS} iterations)")
    plt.savefig("jacobi.png", dpi=150, bbox_inches="tight")
    print("  Saved: jacobi.png")
except ImportError:
    print("  (install matplotlib to save image)")

"""17 -- 2D Heat equation with explicit time stepping.

Solves the heat equation du/dt = alpha * laplacian(u) on a 2D grid using forward
Euler time integration and a 5-point Laplacian stencil.

Initial condition: two hot spots on a cold background.
Boundary condition: edges fixed at 0.

Usage:
  uv run python examples/17_heat_equation.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan', 'level_zero'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

N = 128
ALPHA = 1.0  # thermal diffusivity
DX = 1.0 / N
DT = 0.2 * DX * DX / ALPHA  # CFL condition for stability
STEPS = 2000

u = pgc.field(dtype=pgc.f32, shape=(N, N))
u_new = pgc.field(dtype=pgc.f32, shape=(N, N))


@pgc.kernel
def heat_step(u, u_new, alpha_dt_dx2, n):
    """One explicit Euler step of the heat equation."""
    for i, j in pgc.ndrange(n, n):
        if i > 0:
            if i < n - 1:
                if j > 0:
                    if j < n - 1:
                        laplacian = (
                            u[i + 1, j] + u[i - 1, j] +
                            u[i, j + 1] + u[i, j - 1] -
                            4.0 * u[i, j]
                        )
                        u_new[i, j] = u[i, j] + alpha_dt_dx2 * laplacian
                    else:
                        u_new[i, j] = 0.0
                else:
                    u_new[i, j] = 0.0
            else:
                u_new[i, j] = 0.0
        else:
            u_new[i, j] = 0.0


@pgc.kernel
def copy_field(src, dst, n):
    for i, j in pgc.ndrange(n, n):
        dst[i, j] = src[i, j]


@pgc.kernel
def init_hotspots(u, n):
    for i, j in pgc.ndrange(n, n):
        r1 = sqrt(float((i - n // 3) * (i - n // 3) + (j - n // 3) * (j - n // 3)))
        r2 = sqrt(float((i - 2 * n // 3) * (i - 2 * n // 3) + (j - 2 * n // 3) * (j - 2 * n // 3)))
        if r1 < float(n // 8):
            u[i, j] = 100.0
        else:
            if r2 < float(n // 8):
                u[i, j] = 50.0
            else:
                u[i, j] = 0.0


# Initial condition: two hot spots
u.fill(0.0)
init_hotspots(u, N)

alpha_dt_dx2 = ALPHA * DT / (DX * DX)
print(f"2D Heat equation: {N}x{N}, alpha={ALPHA}, dt={DT:.6f}")
print(f"  CFL parameter alpha*dt/dx^2 = {alpha_dt_dx2:.4f} (must be < 0.25)")

snapshots = []
for step in range(STEPS):
    heat_step(u, u_new, alpha_dt_dx2, N)
    copy_field(u_new, u, N)

    if (step + 1) % 500 == 0:
        data = u.to_numpy().reshape(N, N)
        snapshots.append(data.copy())
        print(f"  step {step+1:>5d}: max={data.max():.4f}  mean={data.mean():.4f}")

# Try to save animation frames
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]
    for ax, snap, step in zip(axes, snapshots, range(500, STEPS + 1, 500)):
        im = ax.imshow(snap, origin="lower", cmap="hot", vmin=0, vmax=100)
        ax.set_title(f"t={step}")
        ax.axis("off")
    plt.suptitle("Heat Equation Diffusion")
    import os
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "results", "heat_equation.png"), dpi=150, bbox_inches="tight")
    print("  Saved: heat_equation.png")
except ImportError:
    print("  (install matplotlib to save image)")

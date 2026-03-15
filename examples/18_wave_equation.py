"""18 — 1D Wave equation simulation.

Solves the wave equation ∂²u/∂t² = c² ∂²u/∂x² using the leapfrog
(Verlet) method.  Three time levels are used: u_prev, u_curr, u_next.

Initial condition: Gaussian pulse at the center.
Boundary condition: fixed ends (u=0).

Usage:
  uv run python examples/18_wave_equation.py
"""

import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'vulkan'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

N = 1024
C = 1.0  # wave speed
DX = 1.0 / N
DT = 0.5 * DX / C  # CFL: c*dt/dx < 1
STEPS = 2000

u_prev = pgc.field(dtype=pgc.f32, shape=(N,))
u_curr = pgc.field(dtype=pgc.f32, shape=(N,))
u_next = pgc.field(dtype=pgc.f32, shape=(N,))


@pgc.kernel
def wave_step(u_prev, u_curr, u_next, courant2, n):
    """Leapfrog update: u_next = 2*u_curr - u_prev + c²dt²/dx² * (u[i+1] - 2u[i] + u[i-1])"""
    for i in range(n):
        if i > 0:
            if i < n - 1:
                laplacian = u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1]
                u_next[i] = 2.0 * u_curr[i] - u_prev[i] + courant2 * laplacian
            else:
                u_next[i] = 0.0  # fixed boundary
        else:
            u_next[i] = 0.0  # fixed boundary


@pgc.kernel
def shift(u_prev, u_curr, u_next, n):
    """Shift time levels: prev ← curr, curr ← next."""
    for i in range(n):
        u_prev[i] = u_curr[i]
        u_curr[i] = u_next[i]


# Initial condition: Gaussian pulse
x = np.linspace(0, 1, N, dtype=np.float32)
sigma = 0.02
pulse = np.exp(-((x - 0.5) ** 2) / (2 * sigma**2)).astype(np.float32)
u_curr.from_numpy(pulse)
u_prev.from_numpy(pulse)  # zero initial velocity

courant2 = (C * DT / DX) ** 2
print(f"1D Wave equation: N={N}, c={C}, Courant²={courant2:.4f}")

snapshots = []
for step in range(STEPS):
    wave_step(u_prev, u_curr, u_next, courant2, N)
    shift(u_prev, u_curr, u_next, N)

    if step % 400 == 0:
        data = u_curr.to_numpy()
        snapshots.append((step, data.copy()))
        print(f"  step {step:>5d}: max={data.max():.6f}  energy~{np.sum(data**2):.6f}")

# Note: the Gaussian splits into two pulses traveling in opposite directions.
# With fixed boundaries (u=0), pulse energy is absorbed at the edges.
# Compare energy between two early snapshots to verify conservation mid-flight.
e1 = np.sum(snapshots[1][1] ** 2)
e2 = np.sum(snapshots[2][1] ** 2)
print(f"\nEnergy at step {snapshots[1][0]}: {e1:.4f}")
print(f"Energy at step {snapshots[2][0]}: {e2:.4f}")
print(f"Ratio: {e2/e1:.6f} (should be ~1.0 before boundary absorption)")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for step, data in snapshots:
        ax.plot(x, data, label=f"t={step}", alpha=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("1D Wave Equation — Gaussian pulse splitting")
    ax.legend()
    plt.savefig("wave_equation.png", dpi=150, bbox_inches="tight")
    print("  Saved: wave_equation.png")
except ImportError:
    print("  (install matplotlib to save image)")

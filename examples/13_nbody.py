"""13 -- N-body gravitational simulation.

Compute gravitational forces between N particles using the direct
O(N^2) algorithm.  Each particle accumulates force from every other
particle -- a classic GPU compute workload.

Usage:
  uv run python examples/13_nbody.py
"""

import time
import numpy as np
import pgc

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu', choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero', 'wgpu'])
_arch = getattr(pgc, _parser.parse_args().arch)
pgc.init(arch=_arch)

N = 1024
DT = 0.001
SOFTENING = 0.01
STEPS = 10

# Position, velocity, and force fields (separate x, y, z components)
px = pgc.field(dtype=pgc.f32, shape=(N,))
py = pgc.field(dtype=pgc.f32, shape=(N,))
pz = pgc.field(dtype=pgc.f32, shape=(N,))
vx = pgc.field(dtype=pgc.f32, shape=(N,))
vy = pgc.field(dtype=pgc.f32, shape=(N,))
vz = pgc.field(dtype=pgc.f32, shape=(N,))
fx = pgc.field(dtype=pgc.f32, shape=(N,))
fy = pgc.field(dtype=pgc.f32, shape=(N,))
fz = pgc.field(dtype=pgc.f32, shape=(N,))
mass = pgc.field(dtype=pgc.f32, shape=(N,))


@pgc.kernel
def compute_forces(px, py, pz, mass, fx, fy, fz, n):
    """O(N^2) all-pairs gravitational force computation."""
    for i in range(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(1024):
            dx = px[j] - px[i]
            dy = py[j] - py[i]
            dz = pz[j] - pz[i]
            dist_sq = dx * dx + dy * dy + dz * dz + 0.01
            inv_dist = 1.0 / sqrt(dist_sq)
            inv_dist3 = inv_dist * inv_dist * inv_dist
            ax = ax + mass[j] * dx * inv_dist3
            ay = ay + mass[j] * dy * inv_dist3
            az = az + mass[j] * dz * inv_dist3
        fx[i] = ax
        fy[i] = ay
        fz[i] = az


@pgc.kernel
def integrate(px, py, pz, vx, vy, vz, fx, fy, fz, mass, dt, n):
    """Symplectic Euler integration."""
    for i in range(n):
        # Update velocity: v += dt * F / m
        vx[i] = vx[i] + dt * fx[i] / mass[i]
        vy[i] = vy[i] + dt * fy[i] / mass[i]
        vz[i] = vz[i] + dt * fz[i] / mass[i]
        # Update position: x += dt * v
        px[i] = px[i] + dt * vx[i]
        py[i] = py[i] + dt * vy[i]
        pz[i] = pz[i] + dt * vz[i]


# Initialize: random positions in a unit sphere, zero velocity
np.random.seed(42)
pos = np.random.randn(N, 3).astype(np.float32) * 0.5
px.from_numpy(pos[:, 0])
py.from_numpy(pos[:, 1])
pz.from_numpy(pos[:, 2])
mass.from_numpy(np.ones(N, dtype=np.float32))

print(f"N-body simulation: {N} particles, {STEPS} steps")
print(f"{'Step':>6s} {'KE':>12s} {'time (ms)':>10s}")

for step in range(STEPS):
    t0 = time.perf_counter()
    compute_forces(px, py, pz, mass, fx, fy, fz, N)
    integrate(px, py, pz, vx, vy, vz, fx, fy, fz, mass, DT, N)
    dt = (time.perf_counter() - t0) * 1000

    # Compute kinetic energy for monitoring
    v2 = vx.to_numpy()**2 + vy.to_numpy()**2 + vz.to_numpy()**2
    ke = 0.5 * np.sum(v2)
    print(f"{step:>6d} {ke:>12.4f} {dt:>10.1f}")

print("\nFinal center of mass:")
print(f"  x={px.to_numpy().mean():.6f}  "
      f"y={py.to_numpy().mean():.6f}  "
      f"z={pz.to_numpy().mean():.6f}")

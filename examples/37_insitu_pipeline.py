"""37 -- In situ pipeline: simulation → isosurface → path trace → image.

Demonstrates the full zero-copy GPU pipeline:
  1. Generate scalar field on GPU (simulating in situ data)
  2. Extract isosurface with Flying Edges (GPU)
  3. Path trace with BVH (GPU)
  4. Denoise with OIDN (CPU)
  5. Write image

No geometry ever touches the host — fields flow directly from
flying edges output to the renderer.

Usage:
  uv run python examples/37_insitu_pipeline.py
  uv run python examples/37_insitu_pipeline.py --arch metal
  uv run python examples/37_insitu_pipeline.py --arch metal --grid 128 --denoise
"""

import time
import numpy as np
import pgc
from pgc.algorithms.flying_edges import flying_edges, UniformGrid
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='cpu',
                choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_p.add_argument('--grid', type=int, default=128,
                help='Grid cells per dimension')
_p.add_argument('--samples', type=int, default=1)
_p.add_argument('--bounces', type=int, default=0)
_p.add_argument('--denoise', action='store_true')
_p.add_argument('--resolution', type=int, default=1024)
_args = _p.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))


# ================================================================
# STEP 1: SIMULATE — generate scalar field on GPU
# ================================================================

@pgc.kernel
def compute_field(scalar, grid: pgc.template(), n_points, time_val):
    """Gyroid with time-varying phase (simulates evolving data)."""
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.get_x(ix, iy, iz)
        y = grid.get_y(ix, iy, iz)
        z = grid.get_z(ix, iy, iz)
        scalar[i] = sin(x + time_val) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x + time_val)


n = _args.grid
domain = np.pi
grid = UniformGrid(n, n, n,
                   -domain, -domain, -domain,
                   2 * domain / n, 2 * domain / n, 2 * domain / n)
n_points = (n + 1) ** 3
scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))

print(f"In situ pipeline: {_args.arch}, {n}^3 grid, {_args.resolution}x{_args.resolution}")
print(f"  {_args.samples} spp, {_args.bounces} bounces" +
      (", OIDN denoise" if _args.denoise else ""))
print()

# Scene setup (reused across timesteps)
scene = Scene()
scene.add(PointLight(position=(8, 10, 6), intensity=200.0))

camera = PerspectiveCamera(
    position=(8, 6, 8), look_at=(0, 0, 0), fov=45,
    width=_args.resolution, height=_args.resolution,
)
canvas = Canvas(_args.resolution, _args.resolution)

# Warmup JIT
compute_field(scalar, grid, n_points, 0.0)
fe_result = flying_edges(scalar, grid, 0.0)
if fe_result:
    scene.add(Actor(fe_result['points_field'], fe_result['conn_field'],
                    color=(0.7, 0.85, 0.95), smooth=True))
    render(canvas, scene, camera, samples=1, max_bounces=0)

# ================================================================
# STEP 2-5: SIMULATE → ISOSURFACE → RENDER → IMAGE
# ================================================================

# Simulate 3 timesteps
for timestep in range(3):
    t_val = timestep * 0.3

    # --- Simulate ---
    t0 = time.perf_counter()
    compute_field(scalar, grid, n_points, t_val)
    t_sim = time.perf_counter() - t0

    # --- Isosurface (GPU, zero-copy output) ---
    t0 = time.perf_counter()
    result = flying_edges(scalar, grid, 0.0)
    t_iso = time.perf_counter() - t0

    if result is None:
        print(f"  Timestep {timestep}: no isosurface")
        continue

    # --- Build scene from GPU fields (zero copy) ---
    t0 = time.perf_counter()
    scene = Scene()
    scene.add(Actor(result['points_field'], result['conn_field'],
                    color=(0.7, 0.85, 0.95), smooth=True))
    scene.add(PointLight(position=(8, 10, 6), intensity=200.0))

    # --- Render (GPU, zero-copy geometry) ---
    render(canvas, scene, camera,
           samples=_args.samples, max_bounces=_args.bounces)
    t_render = time.perf_counter() - t0

    n_tris = result['total_tris']

    # --- Denoise + save ---
    t0 = time.perf_counter()
    if _args.denoise:
        import oidn
        w, h = canvas.width, canvas.height
        r = canvas.color_r.to_numpy().reshape(h, w)
        g = canvas.color_g.to_numpy().reshape(h, w)
        b = canvas.color_b.to_numpy().reshape(h, w)
        color = np.stack([r, g, b], axis=-1).astype(np.float32)
        output = np.zeros_like(color)
        dev = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
        oidn.CommitDevice(dev)
        f = oidn.NewFilter(dev, 'RT')
        oidn.SetSharedFilterImage(f, 'color', color, oidn.FORMAT_FLOAT3, w, h)
        oidn.SetSharedFilterImage(f, 'output', output, oidn.FORMAT_FLOAT3, w, h)
        oidn.CommitFilter(f)
        oidn.ExecuteFilter(f)
        oidn.ReleaseFilter(f)
        oidn.ReleaseDevice(dev)
        img = np.zeros((h, w, 4), dtype=np.uint8)
        img[:, :, 0] = np.clip(output[:, :, 0] * 255, 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(output[:, :, 1] * 255, 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(output[:, :, 2] * 255, 0, 255).astype(np.uint8)
        img[:, :, 3] = 255
    else:
        img = canvas.to_numpy()
    t_save = time.perf_counter() - t0

    fname = f"insitu_t{timestep}.png"
    try:
        from PIL import Image
        Image.fromarray(img).save(fname)
    except ImportError:
        pass

    print(f"  t={t_val:.1f}  tris={n_tris:,}  "
          f"sim={t_sim*1000:.0f}ms  iso={t_iso*1000:.0f}ms  "
          f"render={t_render*1000:.0f}ms  "
          + (f"denoise={t_save*1000:.0f}ms  " if _args.denoise else "")
          + f"total={((t_sim+t_iso+t_render+t_save)*1000):.0f}ms  [{fname}]")

"""32 -- Path tracing: render an isosurface with global illumination.

Extracts a gyroid isosurface using Flying Edges, then path traces it
with the pgc.rendering module.

Usage:
  uv run python examples/32_pathtrace.py
  uv run python examples/32_pathtrace.py --arch metal
  uv run python examples/32_pathtrace.py --samples 16 --bounces 3
  uv run python examples/32_pathtrace.py --resolution 1024
"""

import time
import numpy as np
import pgc
from pgc.algorithms.flying_edges import flying_edges, UniformGrid
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--samples', type=int, default=4)
_parser.add_argument('--bounces', type=int, default=2)
_parser.add_argument('--resolution', type=int, default=512)
_parser.add_argument('--grid_size', type=int, default=64)
_args = _parser.parse_args()
_arch = getattr(pgc, _args.arch)
pgc.init(arch=_arch)


# ================================================================
# GENERATE ISOSURFACE
# ================================================================

@pgc.kernel
def compute_gyroid(scalar, grid: pgc.template(), n_points):
    for i in range(n_points):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.get_x(ix, iy, iz)
        y = grid.get_y(ix, iy, iz)
        z = grid.get_z(ix, iy, iz)
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)


n = _args.grid_size
domain = np.pi
grid = UniformGrid(n, n, n,
                   -domain, -domain, -domain,
                   2 * domain / n, 2 * domain / n, 2 * domain / n)
n_points = (n + 1) ** 3
scalar = pgc.field(dtype=pgc.f32, shape=(n_points,))
compute_gyroid(scalar, grid, n_points)

print(f"Extracting isosurface ({n}^3 grid)...")
t0 = time.perf_counter()
result = flying_edges(scalar, grid, 0.0)
t_fe = time.perf_counter() - t0

if result is None:
    print("No isosurface produced.")
    exit(1)

print(f"  {result['total_points']:,} points, {result['total_tris']:,} triangles")
print(f"  Flying Edges: {t_fe:.4f}s")


# ================================================================
# RENDER
# ================================================================

w = h = _args.resolution

scene = Scene()
scene.add(Actor(result['points_field'], result['conn_field'],
                color=(0.7, 0.85, 0.95), smooth=True))
scene.add(PointLight(position=(8, 10, 6), intensity=200.0))

camera = PerspectiveCamera(
    position=(8, 6, 8),
    look_at=(0, 0, 0),
    fov=45,
    width=w, height=h,
)

canvas = Canvas(w, h)

print(f"\nPath tracing ({w}x{h}, {_args.samples} spp, {_args.bounces} bounces)...")
t0 = time.perf_counter()
render(canvas, scene, camera,
       samples=_args.samples,
       max_bounces=_args.bounces)
t_render = time.perf_counter() - t0
print(f"  Render: {t_render:.4f}s")

# Save image
img = canvas.to_numpy()
try:
    from PIL import Image
    Image.fromarray(img).save("pathtrace_gyroid.png")
    print(f"  Saved: pathtrace_gyroid.png")
except ImportError:
    np.save("pathtrace_gyroid.npy", img)
    print(f"  Saved: pathtrace_gyroid.npy (install Pillow for PNG)")

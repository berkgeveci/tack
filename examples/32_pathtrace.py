"""32 -- Path tracing: render an isosurface with global illumination.

Extracts a gyroid isosurface with tack.dataset's contour, then path traces
it with the tack.rendering module. The surface goes from one to the other
on the device: the renderer reads the contour's own fields.

Usage:
  uv run python examples/32_pathtrace.py
  uv run python examples/32_pathtrace.py --arch metal
  uv run python examples/32_pathtrace.py --samples 16 --bounces 3
  uv run python examples/32_pathtrace.py --resolution 1024
"""

import argparse
import os
import time

import numpy as np

import tack
import tack.dataset as td
from tack.rendering import (
    Actor,
    Canvas,
    PerspectiveCamera,
    PointLight,
    Scene,
    render,
)

_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--samples', type=int, default=4)
_parser.add_argument('--bounces', type=int, default=2)
_parser.add_argument('--resolution', type=int, default=512)
_parser.add_argument('--grid_size', type=int, default=64)
_args = _parser.parse_args()
_arch = getattr(tack, _args.arch)
tack.init(arch=_arch)


# ================================================================
# GENERATE ISOSURFACE
# ================================================================

@tack.kernel
def compute_gyroid(xyz, scalar):
    for p in range(scalar.shape[0]):
        x = xyz.get(p)                      # the grid computes point p's position
        scalar[p] = sin(x[0]) * cos(x[1]) + sin(x[1]) * cos(x[2]) + sin(x[2]) * cos(x[0])


n = _args.grid_size
domain = np.pi
grid = td.uniform_grid((n + 1, n + 1, n + 1), origin=(-domain, -domain, -domain),
                       spacing=(2 * domain / n,) * 3)
scalar = tack.field(dtype=tack.f32, shape=(grid.num_points,))
compute_gyroid(grid.geometry.values, scalar)
grid.fields["gyroid"] = td.Field(td.H1(grid), scalar)

print(f"Extracting isosurface ({n}^3 grid)...")
t0 = time.perf_counter()
surface = td.contour(grid, "gyroid", 0.0)
tack.sync()
t_contour = time.perf_counter() - t0

if surface.num_cells == 0:
    print("No isosurface produced.")
    exit(1)

print(f"  {surface.num_points:,} points, {surface.num_cells:,} triangles")
print(f"  Contour: {t_contour:.4f}s")


# ================================================================
# RENDER
# ================================================================

w = h = _args.resolution

scene = Scene()
positions = surface.geometry.values              # 3-vectors, stored interleaved
triangles = surface.topology.connectivity
if triangles.dtype != tack.i32:                  # a very large surface keeps i64 ids
    triangles = triangles.astype(tack.i32)
scene.add(Actor(positions.reshape(positions.size), triangles,
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
os.makedirs("results", exist_ok=True)
try:
    from PIL import Image
    Image.fromarray(img).save("results/pathtrace_gyroid.png")
    print("  Saved: results/pathtrace_gyroid.png")
except ImportError:
    np.save("results/pathtrace_gyroid.npy", img)
    print("  Saved: results/pathtrace_gyroid.npy (install Pillow for PNG)")

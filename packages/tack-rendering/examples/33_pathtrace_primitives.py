"""33 -- Path trace a cube and a sphere."""

import time
import numpy as np
import tack
from tack.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--samples', type=int, default=4)
_parser.add_argument('--bounces', type=int, default=2)
_parser.add_argument('--resolution', type=int, default=512)
_args = _parser.parse_args()
tack.init(arch=getattr(tack, _args.arch))


# ================================================================
# GEOMETRY HELPERS
# ================================================================

def make_cube(center=(0, 0, 0), size=1.0):
    """Unit cube as (verts, tris) numpy arrays."""
    s = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ], dtype=np.float32)
    t = np.array([
        [0,1,2], [0,2,3],  # -Z
        [4,6,5], [4,7,6],  # +Z
        [0,4,5], [0,5,1],  # -Y
        [2,6,7], [2,7,3],  # +Y
        [0,3,7], [0,7,4],  # -X
        [1,5,6], [1,6,2],  # +X
    ], dtype=np.int32)
    return v, t


def make_sphere(center=(0, 0, 0), radius=1.0, subdivisions=16):
    """UV sphere as (verts, tris) numpy arrays."""
    cx, cy, cz = center
    n_lat = subdivisions
    n_lon = subdivisions * 2
    verts = []
    # Top pole
    verts.append([cx, cy + radius, cz])
    # Latitude rings
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = cx + radius * np.sin(phi) * np.cos(theta)
            y = cy + radius * np.cos(phi)
            z = cz + radius * np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])
    # Bottom pole
    verts.append([cx, cy - radius, cz])
    verts = np.array(verts, dtype=np.float32)

    tris = []
    # Top cap
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([0, 1 + j, 1 + j_next])
    # Middle bands
    for i in range(n_lat - 2):
        row = 1 + i * n_lon
        next_row = row + n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            tris.append([row + j, next_row + j, next_row + j_next])
            tris.append([row + j, next_row + j_next, row + j_next])
    # Bottom cap
    bot = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([bot, last_row + j_next, last_row + j])
    tris = np.array(tris, dtype=np.int32)
    return verts, tris


# ================================================================
# BUILD SCENE
# ================================================================

cube_v, cube_t = make_cube(center=(-1.5, 0, 0), size=2.0)
sphere_v, sphere_t = make_sphere(center=(1.5, 0, 0), radius=1.0, subdivisions=20)

# Ground plane
plane_v = np.array([
    [-5, -1, -5], [5, -1, -5], [5, -1, 5], [-5, -1, 5],
], dtype=np.float32)
plane_t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

def upload_mesh(verts, tris):
    pts = tack.field(dtype=tack.f32, shape=(verts.size,))
    pts.from_numpy(verts.reshape(-1))
    conn = tack.field(dtype=tack.i32, shape=(tris.size,))
    conn.from_numpy(tris.reshape(-1))
    return pts, conn

cube_pts, cube_conn = upload_mesh(cube_v, cube_t)
sphere_pts, sphere_conn = upload_mesh(sphere_v, sphere_t)
plane_pts, plane_conn = upload_mesh(plane_v, plane_t)

scene = Scene()
scene.add(Actor(cube_pts, cube_conn, color=(0.9, 0.3, 0.2)))
scene.add(Actor(sphere_pts, sphere_conn, color=(0.2, 0.5, 0.9), smooth=True))
scene.add(Actor(plane_pts, plane_conn, color=(0.7, 0.7, 0.7)))
scene.add(PointLight(position=(5, 8, 5), intensity=150.0))

w = h = _args.resolution
camera = PerspectiveCamera(
    position=(0, 3, 8),
    look_at=(0, 0, 0),
    fov=45,
    width=w, height=h,
)

canvas = Canvas(w, h)

print(f"Cube:   {cube_t.shape[0]} tris")
print(f"Sphere: {sphere_t.shape[0]} tris")
print(f"Rendering {w}x{h}, {_args.samples} spp, {_args.bounces} bounces...")

# Warmup (JIT compile)
render(canvas, scene, camera, samples=1, max_bounces=0)

# Timed run (includes BVH construction)
t0 = time.perf_counter()
render(canvas, scene, camera, samples=_args.samples, max_bounces=_args.bounces)
t_render = time.perf_counter() - t0
print(f"Render: {t_render:.4f}s")

img = canvas.to_numpy()
try:
    from PIL import Image
    Image.fromarray(img).save("pathtrace_primitives.png")
    print("Saved: pathtrace_primitives.png")
except ImportError:
    np.save("pathtrace_primitives.npy", img)
    print("Saved: pathtrace_primitives.npy (install Pillow for PNG)")

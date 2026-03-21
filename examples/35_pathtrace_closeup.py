"""35 -- Close-up of 1M triangle sphere to inspect sampling patterns."""

import time
import math
import numpy as np
import pgc
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--arch', default='cpu',
                     choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_parser.add_argument('--samples', type=int, default=4)
_parser.add_argument('--bounces', type=int, default=2)
_args = _parser.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))


def make_sphere(center=(0, 0, 0), radius=1.0, subdivisions=16):
    cx, cy, cz = center
    n_lat = subdivisions
    n_lon = subdivisions * 2
    verts = []
    verts.append([cx, cy + radius, cz])
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = cx + radius * np.sin(phi) * np.cos(theta)
            y = cy + radius * np.cos(phi)
            z = cz + radius * np.sin(phi) * np.sin(theta)
            verts.append([x, y, z])
    verts.append([cx, cy - radius, cz])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([0, 1 + j, 1 + j_next])
    for i in range(n_lat - 2):
        row = 1 + i * n_lon
        next_row = row + n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            tris.append([row + j, next_row + j, next_row + j_next])
            tris.append([row + j, next_row + j_next, row + j_next])
    bot = len(verts) - 1
    last_row = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        tris.append([bot, last_row + j_next, last_row + j])
    tris = np.array(tris, dtype=np.int32)
    return verts, tris


def upload_mesh(verts, tris):
    pts = pgc.field(dtype=pgc.f32, shape=(verts.size,))
    pts.from_numpy(verts.reshape(-1))
    conn = pgc.field(dtype=pgc.i32, shape=(tris.size,))
    conn.from_numpy(tris.reshape(-1))
    return pts, conn


subdiv = 501
sv, st = make_sphere(center=(0, 0, 0), radius=1.0, subdivisions=subdiv)
print(f"Sphere: {st.shape[0]:,} tris")
sp, sc = upload_mesh(sv, st)

# Ground plane
plane_v = np.array([[-5, -1, -5], [5, -1, -5], [5, -1, 5], [-5, -1, 5]], dtype=np.float32)
plane_t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
plane_pts, plane_conn = upload_mesh(plane_v, plane_t)

scene = Scene()
scene.add(Actor(sp, sc, color=(0.2, 0.5, 0.9)))
scene.add(Actor(plane_pts, plane_conn, color=(0.7, 0.7, 0.7)))
scene.add(PointLight(position=(5, 8, 5), intensity=150.0))

# Close-up camera looking at the sphere's equator
camera = PerspectiveCamera(
    position=(0, 0.2, 2.5),
    look_at=(0, 0, 0),
    fov=40,
    width=1024, height=1024,
)

canvas = Canvas(1024, 1024)

# Warmup
render(canvas, scene, camera, samples=1, max_bounces=0)

t0 = time.perf_counter()
render(canvas, scene, camera, samples=_args.samples, max_bounces=_args.bounces)
t = time.perf_counter() - t0
print(f"Render: {t:.4f}s  (1024x1024, {_args.samples}spp, {_args.bounces} bounces)")

img = canvas.to_numpy()
try:
    from PIL import Image
    Image.fromarray(img).save("pathtrace_closeup.png")
    print("Saved: pathtrace_closeup.png")
except ImportError:
    pass

"""39 -- Scalar field coloring: sphere colored by height using ColorTable."""

import numpy as np
import pgc
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, ColorTable, render,
)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='cpu',
                choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_p.add_argument('--preset', default='viridis',
                choices=ColorTable.available_presets())
_args = _p.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))


def make_sphere(center=(0, 0, 0), radius=1.0, subdivisions=32):
    cx, cy, cz = center
    n_lat, n_lon = subdivisions, subdivisions * 2
    verts = [[cx, cy + radius, cz]]
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            verts.append([cx + radius*np.sin(phi)*np.cos(theta),
                          cy + radius*np.cos(phi),
                          cz + radius*np.sin(phi)*np.sin(theta)])
    verts.append([cx, cy - radius, cz])
    verts = np.array(verts, dtype=np.float32)
    tris = []
    for j in range(n_lon):
        tris.append([0, 1+j, 1+(j+1)%n_lon])
    for i in range(n_lat-2):
        row = 1 + i*n_lon
        for j in range(n_lon):
            jn = (j+1)%n_lon
            tris.append([row+j, row+n_lon+j, row+n_lon+jn])
            tris.append([row+j, row+n_lon+jn, row+jn])
    bot = len(verts)-1
    lr = 1+(n_lat-2)*n_lon
    for j in range(n_lon):
        tris.append([bot, lr+(j+1)%n_lon, lr+j])
    return verts, np.array(tris, dtype=np.int32)


def upload(v, t):
    p = pgc.field(dtype=pgc.f32, shape=(v.size,))
    p.from_numpy(v.reshape(-1))
    c = pgc.field(dtype=pgc.i32, shape=(t.size,))
    c.from_numpy(t.reshape(-1))
    return p, c


# Sphere geometry
sv, st = make_sphere((0, 0, 0), 1.0, 32)
sp, sc = upload(sv, st)

# Per-vertex scalar: height (y coordinate)
height_scalars = sv[:, 1].copy()

# Color table maps height → color
ct = ColorTable(_args.preset)

# Ground plane (uniform grey)
pv = np.array([[-3, -1, -3], [3, -1, -3], [3, -1, 3], [-3, -1, 3]],
              dtype=np.float32)
pt = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
pp, pc = upload(pv, pt)

scene = Scene()
scene.add(Actor(sp, sc, scalars=height_scalars, color_table=ct, smooth=True))
scene.add(Actor(pp, pc, color=(0.7, 0.7, 0.7)))
scene.add(PointLight(position=(5, 8, 5), intensity=150.0))

camera = PerspectiveCamera(
    position=(0, 2, 4), look_at=(0, 0, 0), fov=45, width=1024, height=1024)
canvas = Canvas(1024, 1024)

print(f"Rendering sphere with '{_args.preset}' color table...")
render(canvas, scene, camera, samples=1, max_bounces=0)  # warmup
render(canvas, scene, camera, samples=4, max_bounces=1)

img = canvas.to_numpy()
try:
    from PIL import Image
    fname = f"pathtrace_scalar_{_args.preset}.png"
    Image.fromarray(img).save(fname)
    print(f"Saved: {fname}")
except ImportError:
    print("Install Pillow to save images: pip install Pillow")

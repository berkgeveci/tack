"""40 -- Multiple colored lights illuminating a scene."""

import numpy as np
import tack
from tack.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='cpu',
                choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_args = _p.parse_args()
tack.init(arch=getattr(tack, _args.arch))


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
    p = tack.field(dtype=tack.f32, shape=(v.size,))
    p.from_numpy(v.reshape(-1))
    c = tack.field(dtype=tack.i32, shape=(t.size,))
    c.from_numpy(t.reshape(-1))
    return p, c


# Sphere
sv, st = make_sphere((0, 0, 0), 1.0, 32)
sp, sc = upload(sv, st)

# Ground plane
pv = np.array([[-4, -1, -4], [4, -1, -4], [4, -1, 4], [-4, -1, 4]],
              dtype=np.float32)
pt = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
pp, pc = upload(pv, pt)

scene = Scene()
scene.add(Actor(sp, sc, color=(0.9, 0.9, 0.9), smooth=True))
scene.add(Actor(pp, pc, color=(0.8, 0.8, 0.8)))

# Three colored lights from different directions
scene.add(PointLight(position=(-4, 6, 2), intensity=80.0,
                     color=(1.0, 0.3, 0.3)))   # red from the left
scene.add(PointLight(position=(4, 6, 2), intensity=80.0,
                     color=(0.3, 0.3, 1.0)))    # blue from the right
scene.add(PointLight(position=(0, 6, -4), intensity=60.0,
                     color=(0.3, 1.0, 0.3)))    # green from behind

camera = PerspectiveCamera(
    position=(0, 2, 5), look_at=(0, 0, 0), fov=45, width=1024, height=1024)
canvas = Canvas(1024, 1024)

print("Rendering with 3 colored lights (red, blue, green)...")
render(canvas, scene, camera, samples=1, max_bounces=0)  # warmup
render(canvas, scene, camera, samples=4, max_bounces=1)

img = canvas.to_numpy()
try:
    from PIL import Image
    Image.fromarray(img).save("pathtrace_multilights.png")
    print("Saved: pathtrace_multilights.png")
except ImportError:
    print("Install Pillow to save images: pip install Pillow")

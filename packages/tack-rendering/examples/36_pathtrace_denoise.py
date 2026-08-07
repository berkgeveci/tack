"""36 -- Path trace + OIDN denoising: 1 spp → clean image."""

import argparse
import time

import numpy as np
import oidn

import tack
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
_parser.add_argument('--samples', type=int, default=1)
_parser.add_argument('--bounces', type=int, default=2)
_args = _parser.parse_args()
tack.init(arch=getattr(tack, _args.arch))


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


def make_cube(center=(0, 0, 0), size=1.0):
    s = size / 2.0
    cx, cy, cz = center
    v = np.array([
        [cx-s, cy-s, cz-s], [cx+s, cy-s, cz-s],
        [cx+s, cy+s, cz-s], [cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s], [cx+s, cy-s, cz+s],
        [cx+s, cy+s, cz+s], [cx-s, cy+s, cz+s],
    ], dtype=np.float32)
    t = np.array([
        [0,1,2], [0,2,3], [4,6,5], [4,7,6],
        [0,4,5], [0,5,1], [2,6,7], [2,7,3],
        [0,3,7], [0,7,4], [1,5,6], [1,6,2],
    ], dtype=np.int32)
    return v, t


def upload_mesh(verts, tris):
    pts = tack.field(dtype=tack.f32, shape=(verts.size,))
    pts.from_numpy(verts.reshape(-1))
    conn = tack.field(dtype=tack.i32, shape=(tris.size,))
    conn.from_numpy(tris.reshape(-1))
    return pts, conn


def denoise(canvas):
    """Run OIDN on canvas color buffers (linear HDR float)."""
    w, h = canvas.width, canvas.height
    r = canvas.color_r.to_numpy().reshape(h, w)
    g = canvas.color_g.to_numpy().reshape(h, w)
    b = canvas.color_b.to_numpy().reshape(h, w)
    color = np.stack([r, g, b], axis=-1).astype(np.float32)
    output = np.zeros_like(color)

    device = oidn.NewDevice(oidn.DEVICE_TYPE_DEFAULT)
    oidn.CommitDevice(device)

    filt = oidn.NewFilter(device, 'RT')
    oidn.SetSharedFilterImage(filt, 'color', color, oidn.FORMAT_FLOAT3, w, h)
    oidn.SetSharedFilterImage(filt, 'output', output, oidn.FORMAT_FLOAT3, w, h)
    oidn.CommitFilter(filt)
    oidn.ExecuteFilter(filt)
    oidn.ReleaseFilter(filt)
    oidn.ReleaseDevice(device)

    # Convert to uint8 RGBA
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, 0] = np.clip(output[:, :, 0] * 255, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(output[:, :, 1] * 255, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(output[:, :, 2] * 255, 0, 255).astype(np.uint8)
    img[:, :, 3] = 255
    return img


# ================================================================
# SCENE
# ================================================================

cube_v, cube_t = make_cube(center=(-1.5, 0, 0), size=2.0)
sphere_v, sphere_t = make_sphere(center=(1.5, 0, 0), radius=1.0, subdivisions=40)
plane_v = np.array([[-5, -1, -5], [5, -1, -5], [5, -1, 5], [-5, -1, 5]], dtype=np.float32)
plane_t = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

cube_pts, cube_conn = upload_mesh(cube_v, cube_t)
sphere_pts, sphere_conn = upload_mesh(sphere_v, sphere_t)
plane_pts, plane_conn = upload_mesh(plane_v, plane_t)

scene = Scene()
scene.add(Actor(cube_pts, cube_conn, color=(0.9, 0.3, 0.2)))
scene.add(Actor(sphere_pts, sphere_conn, color=(0.2, 0.5, 0.9)))
scene.add(Actor(plane_pts, plane_conn, color=(0.7, 0.7, 0.7)))
scene.add(PointLight(position=(5, 8, 5), intensity=150.0))

w = h = 1024
camera = PerspectiveCamera(
    position=(0, 3, 8), look_at=(0, 0, 0), fov=45, width=w, height=h,
)
canvas = Canvas(w, h)

print(f"Scene: {cube_t.shape[0] + sphere_t.shape[0] + plane_t.shape[0]} tris")
print(f"Rendering {w}x{h}, {_args.samples}spp, {_args.bounces} bounces...")

# Warmup
render(canvas, scene, camera, samples=1, max_bounces=0)

# Timed render
t0 = time.perf_counter()
render(canvas, scene, camera, samples=_args.samples, max_bounces=_args.bounces)
t_render = time.perf_counter() - t0
print(f"Render: {t_render:.4f}s")

# Save noisy
noisy = canvas.to_numpy()

# Denoise
t0 = time.perf_counter()
denoised = denoise(canvas)
t_denoise = time.perf_counter() - t0
print(f"Denoise: {t_denoise:.4f}s")

try:
    from PIL import Image
    Image.fromarray(noisy).save("pathtrace_noisy.png")
    Image.fromarray(denoised).save("pathtrace_denoised.png")
    print("Saved: pathtrace_noisy.png, pathtrace_denoised.png")
except ImportError:
    np.save("pathtrace_noisy.npy", noisy)
    np.save("pathtrace_denoised.npy", denoised)

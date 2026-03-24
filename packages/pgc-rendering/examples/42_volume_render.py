"""42 -- Volume rendering of a gyroid scalar field on a uniform grid."""

import numpy as np
import pgc
from pgc.rendering import (
    PerspectiveCamera, Canvas, Volume, TransferFunction, render_volume,
)

import argparse
_p = argparse.ArgumentParser()
_p.add_argument('--arch', default='cpu',
                choices=['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
_p.add_argument('--size', type=int, default=64,
                help='Grid points per dimension')
_p.add_argument('--width', type=int, default=512)
_p.add_argument('--height', type=int, default=512)
_args = _p.parse_args()
pgc.init(arch=getattr(pgc, _args.arch))

# Build uniform grid with gyroid scalar field
N = _args.size
x = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
y = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
z = np.linspace(-np.pi, np.pi, N, dtype=np.float32)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
scalars = (np.sin(xx) * np.cos(yy) + np.sin(yy) * np.cos(zz)
           + np.sin(zz) * np.cos(xx)).astype(np.float32)

vmin, vmax = float(scalars.min()), float(scalars.max())
print(f"Grid: {N}x{N}x{N} = {N**3:,} points")
print(f"Scalar range: [{vmin:.3f}, {vmax:.3f}]")

spacing = 2 * np.pi / (N - 1)

# Transfer function: cool-to-warm with opacity peaks at extremes
def opacity(t):
    dist = abs(t - 0.5) * 2.0  # 0 at center, 1 at edges
    return 0.005 + 0.06 * dist ** 1.5

tf = TransferFunction('cool_to_warm', opacity_func=opacity,
                      range=(vmin, vmax))

vol = Volume(scalars.ravel(), dims=(N, N, N),
             origin=(-np.pi, -np.pi, -np.pi),
             spacing=(spacing, spacing, spacing),
             transfer_function=tf, opacity_scale=8.0)

camera = PerspectiveCamera(
    position=(8, 6, 8), look_at=(0, 0, 0), fov=45,
    width=_args.width, height=_args.height)
canvas = Canvas(_args.width, _args.height)

print("Rendering volume...")
render_volume(canvas, vol, camera)  # warmup
render_volume(canvas, vol, camera)

img = canvas.to_numpy()
try:
    from PIL import Image
    Image.fromarray(img).save("pathtrace_gyroid_vol.png")
    print("Saved: pathtrace_gyroid_vol.png")
except ImportError:
    print("Install Pillow to save images: pip install Pillow")

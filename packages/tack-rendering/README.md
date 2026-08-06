# tack-rendering

GPU path tracing and volume rendering for [Tack](https://github.com/berkgeveci/tack).

Built entirely from `@tack.kernel` functions, so it runs on every backend `tack-core`
supports — CPU, Metal, CUDA, HIP and Level Zero — from one source.

```python
import tack
from tack.rendering import Scene, Actor, PerspectiveCamera, Canvas, render

tack.init(arch=tack.cpu)

scene = Scene()
scene.add(Actor(points, triangles))
canvas = Canvas(800, 600)
render(canvas, scene, PerspectiveCamera(position=(0, 0, 5), look_at=(0, 0, 0)))
canvas.save("out.png")
```

## What's here

- **Path tracer** — GPU BVH build and traversal, multi-bounce global illumination
- **Materials** — matte, specular, and transparent with Snell refraction and Schlick Fresnel
- **Volume rendering** — ray marching with trilinear sampling and front-to-back compositing,
  integrated with the path tracer so volumes and surfaces compose correctly
- **Cameras** — perspective and orthographic
- **Rasterization** — wireframe and point rendering with atomic depth testing
- **Colour** — `ColorTable` presets, per-vertex scalar colouring, transfer functions
- **Annotations** — colour bars, axis indicators, text overlays

## Install

```bash
pip install tack-rendering
```

Requires `tack-core` with a backend extra — see the
[tack-core README](https://pypi.org/project/tack-core/).

BSD 3-Clause licensed.

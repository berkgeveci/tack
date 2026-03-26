# Path Tracing Renderer

The `pgc-rendering` package provides a GPU-accelerated path tracer that
operates directly on PGC fields. Geometry from visualization algorithms
(like flying edges) can be rendered without any host copies.

```bash
pip install pgc-rendering    # pulls in pgc-core automatically
```

## Quick Start

```python
import numpy as np
import pgc
from pgc.rendering import (
    PerspectiveCamera, Canvas, Scene, Actor, PointLight, render,
)

pgc.init(arch=pgc.metal)

# Create geometry (triangle mesh as numpy arrays)
points = np.array([[0,0,0], [1,0,0], [0.5,1,0]], dtype=np.float32)
triangles = np.array([[0, 1, 2]], dtype=np.int32)

# Build scene
scene = Scene()
scene.add(Actor(points, triangles, color=(0.8, 0.2, 0.2)))
scene.add(PointLight(position=(2, 3, 2), intensity=1.5))

# Set up camera and canvas
camera = PerspectiveCamera(
    position=(0.5, 0.5, 3),
    look_at=(0.5, 0.5, 0),
    fov=45,
    width=512, height=512,
)
canvas = Canvas(512, 512)

# Render
render(canvas, scene, camera, samples=16, max_bounces=3)

# Get result as numpy RGBA image
image = canvas.to_numpy()  # (512, 512, 4) uint8
```

## Components

### PerspectiveCamera

A `@pgc.data_oriented` perspective camera. All parameters become compile-time
constants in the path tracing kernel.

```python
camera = PerspectiveCamera(
    position=(0, 0, 5),       # eye position
    look_at=(0, 0, 0),        # target
    up=(0, 1, 0),             # up vector
    fov=45.0,                 # vertical field of view (degrees)
    width=1024, height=1024,  # image resolution
)
```

### Canvas

An RGBA framebuffer backed by PGC fields (three f32 fields for R, G, B):

```python
canvas = Canvas(width, height)

# After rendering:
image = canvas.to_numpy()        # (H, W, 4) uint8 RGBA
```

### Scene and Actor

A scene contains actors (triangle meshes) and lights:

```python
scene = Scene()

# Add a triangle mesh actor
actor = Actor(
    points,                      # (N, 3) float32 vertex positions
    triangles,                   # (M, 3) int32 triangle indices
    color=(0.8, 0.2, 0.2),      # diffuse color (RGB, 0-1)
)
scene.add(actor)

# Add a point light
scene.add(PointLight(position=(2, 3, 2), intensity=1.5))
```

Multiple actors can be added to a scene. The renderer builds a unified
BVH (bounding volume hierarchy) across all geometry for efficient ray
traversal.

### Transforms

Each actor can have a 4x4 affine transformation matrix applied to its
vertices during scene preparation:

```python
import numpy as np

# Translation
xform = np.eye(4, dtype=np.float32)
xform[0, 3] = 2.0  # shift x by 2

actor = Actor(points, triangles, transform=xform)

# Rotation (90° around Z axis)
rot = np.array([
    [0, -1, 0, 0],
    [1,  0, 0, 0],
    [0,  0, 1, 0],
    [0,  0, 0, 1],
], dtype=np.float32)

actor = Actor(points, triangles, transform=rot)

# Scale
scale = np.diag([2.0, 2.0, 2.0, 1.0]).astype(np.float32)
actor = Actor(points, triangles, transform=scale)
```

The transform is applied on GPU. Normals are automatically transformed
using the inverse-transpose of the 3x3 rotation part, ensuring correct
lighting. The original points field is not modified — a copy is made.

Different actors in the same scene can have different transforms, enabling
instanced rendering of the same mesh at multiple positions/orientations.

### render()

```python
render(
    canvas,              # Canvas to render into
    scene,               # Scene with actors and lights
    camera,              # PerspectiveCamera
    samples=16,          # samples per pixel (more = less noise)
    max_bounces=3,       # max path bounces (more = more indirect light)
)
```

Each sample launches one GPU kernel where every pixel traces a complete
light path. Multiple samples are accumulated progressively.

### Depth Buffer

The path tracer writes a depth buffer alongside the color output.
After rendering, access it via `canvas.depth_to_numpy()`:

```python
render(canvas, scene, camera, samples=4)

# Get depth as (H, W) float32 array
depth = canvas.depth_to_numpy()

# -1 means background (no hit), positive values are ray distances
hit_mask = depth >= 0
print(f"Closest: {depth[hit_mask].min():.2f}")
print(f"Farthest: {depth[hit_mask].max():.2f}")
```

The depth values are ray hit distances from the camera origin (not
clip-space Z). Useful for depth-of-field effects, compositing, or
exporting depth maps.

## GPU Pipeline

The renderer is built entirely from PGC kernels:

1. **BVH construction** — GPU kernel builds a bounding volume hierarchy
   over all triangles using Morton codes and radix-like insertion
2. **Path tracing** — Each pixel casts rays through the BVH, computing
   direct and indirect illumination with Russian roulette termination
3. **Accumulation** — Multiple samples are averaged into the canvas

All data stays on GPU throughout. When combined with `flying_edges`,
the full pipeline (scalar field → isosurface → render) runs without
any host-device transfers.

## Integration with Visualization

The most powerful use case combines `pgc-vis` with `pgc-rendering`:

```python
from pgc.algorithms.flying_edges import flying_edges, UniformGrid
from pgc.rendering import PerspectiveCamera, Canvas, Scene, Actor, PointLight, render

# Extract isosurface (GPU)
points, conn, n_pts, n_tris = flying_edges(scalar, grid, isovalue=0.0)

# Convert to numpy for scene setup
pts_np = points.to_numpy().reshape(-1, 3)
tri_np = conn.to_numpy().reshape(-1, 3)

# Render (GPU)
scene = Scene()
scene.add(Actor(pts_np, tri_np, color=(0.7, 0.85, 1.0)))
scene.add(PointLight(position=(5, 5, 5), intensity=2.0))

camera = PerspectiveCamera((3, 3, 3), (0, 0, 0), width=1024, height=1024)
canvas = Canvas(1024, 1024)
render(canvas, scene, camera, samples=64, max_bounces=4)
```

## Denoising

For production-quality images with fewer samples, use
[Intel Open Image Denoise (OIDN)](https://www.openimagedenoise.org/)
on the rendered output:

```python
try:
    import oidn
    device = oidn.NewDevice()
    image_f32 = canvas.to_numpy().astype(np.float32) / 255.0
    denoised = oidn.denoise(device, image_f32[:, :, :3])
except ImportError:
    pass  # OIDN not installed
```

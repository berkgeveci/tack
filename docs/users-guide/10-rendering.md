# Rendering fields and meshes

`tack-rendering` turns triangle fields and uniform-grid volumes into images. Its
unified `render` function chooses path tracing for solid surfaces, rasterization
for wireframes/points, and ray casting for standalone volumes. A CPU backend can
run the same examples; a GPU is useful for larger images and sample counts.

```bash
pip install 'tack-core[cpu]' tack-rendering
```

For a complete illustrated computation, see [Isosurface to image](tutorials/isosurface.md).

## A triangle you can actually see

Mesh geometry is stored as **flat Tack fields**, rather than `(N, 3)` NumPy
arrays passed directly to `Actor`. Upload host geometry explicitly:

```python
import numpy as np
import tack
from tack.rendering import Actor, Canvas, PerspectiveCamera, PointLight, Scene, render

tack.init(arch=tack.cpu)
vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
triangles = np.array([[0, 1, 2]], dtype=np.int32)
points = tack.field_like(vertices.ravel())
connectivity = tack.field_like(triangles.ravel())

scene = Scene()
scene.add(Actor(points, connectivity, color=(0.8, 0.2, 0.2)))
scene.add(PointLight(position=(2, 3, 2), intensity=40.0))
camera = PerspectiveCamera(position=(0.5, 0.5, 3), look_at=(0.5, 0.5, 0),
                           fov=45, width=128, height=128)
canvas = Canvas(128, 128)
render(canvas, scene, camera, samples=4, max_bounces=1)
image = canvas.to_numpy()                # (128, 128, 4), uint8 RGBA
assert np.count_nonzero(canvas.depth_to_numpy() >= 0) > 0
```

Save `image` with an image library, for example `PIL.Image.fromarray(image).save("triangle.png")`
when Pillow is installed. A depth-hit check is useful: a render completing
without an exception does not prove that the scene contained visible triangles.

## Scene, camera and canvas

A `Scene` holds actors, volumes and lights. An `Actor` describes geometry and
appearance. A `Camera` supplies rays; a `Canvas` stores the framebuffer and
reusable working fields. Match the camera and canvas dimensions.

`PerspectiveCamera` uses a vertical field of view in degrees. `OrthographicCamera`
uses a world-space `view_height` and parallel rays:

```python
from tack.rendering import OrthographicCamera

camera = OrthographicCamera((0, 0, 3), (0, 0, 0), view_height=2.0,
                            width=128, height=128)
```

Both are data-oriented classes with precomputed ray parameters as instance
scalars. Those are runtime kernel parameters, rather than class constants that
require a new specialization whenever the view changes. Pixel `(0, 0)` is at
the image's top left, with image y increasing downward.

`Canvas.to_numpy()` returns display RGB plus an opaque alpha channel. Surface
path tracing produces ray-distance depth: `canvas.depth_to_numpy()` has shape
`(height, width)`, with `-1` for background and nonnegative distances for hits.
Standalone volume rendering does not produce a comparable surface-hit depth.

## Geometry and appearance

`Actor(points, connectivity, ...)` expects flat `f32` XYZ coordinates and flat
`i32` triangle vertex indices. The triangle indices refer to vertices, not scalar
component offsets. Optional appearance arguments include:

| Argument | Use |
|---|---|
| `color=(r, g, b)` | Uniform diffuse RGB |
| `smooth=True` | Compute interpolated vertex normals |
| `normals=...` | Supply flat field normals or a NumPy normal array |
| `point_colors=...` | Per-vertex RGB field, or a NumPy RGB array |
| `scalars=..., color_table=...` | Map a scalar per vertex through a colour table |
| `material=...` | Matte, specular or transparent surface |
| `render_mode="wireframe"` or `"points"` | Rasterize edges or vertex discs |
| `transform=...` | Apply a host-specified 4×4 affine transform during preparation |

For scalar colouring, choose a fixed range when comparing frames:

```python
from tack.rendering import ColorTable

table = ColorTable("viridis", range=(0.0, 1.0))
actor = Actor(points, connectivity, scalars=vertex_values, color_table=table, smooth=True)
```

A changing automatically chosen range can make two images look similar even
when their scalar magnitudes differ. Per-vertex colours take precedence over
scalar mapping, which takes precedence over uniform colour.

Materials provide different light interactions:

```python
from tack.rendering import Material

matte = Material(Material.MATTE)
specular = Material(Material.SPECULAR)
glass = Material(Material.TRANSPARENT, ior=1.5)
```

See the [materials example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/33_pathtrace_primitives.py)
and [scalar-colouring example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/39_scalar_coloring.py)
for complete scenes.

## Transforms and lighting

An actor's transform is a 4×4 affine matrix. For a translation:

```python
xform = np.eye(4, dtype=np.float32)
xform[0, 3] = 2.0
actor = Actor(points, connectivity, transform=xform)
```

Preparation applies it to geometry using kernels, retaining the source positions.
Normals use the inverse-transpose of the linear part. Reusing source fields
across transformed actors is useful for instancing.

A `PointLight` has a position, colour and scalar intensity. A `DirectionalLight`
has a direction toward the light. Multiple lights contribute to the scene:

```python
from tack.rendering import DirectionalLight

scene.add(DirectionalLight(direction=(1, 2, 1), intensity=0.5))
```

Start with a simple light and a visible object, then adjust brightness and
materials. Samples and bounces are distinct controls: more samples reduce
noise, while more bounces allow additional indirect light paths.

## Retain geometry between pipeline stages

After a nonempty flying-edges extraction:

```python
scene = Scene()
scene.add(Actor(mesh["points_field"], mesh["conn_field"], smooth=True))
```

This avoids uploading the already-retained geometry fields. Flying edges still
reads counts and downloads its NumPy results, as described in
[Visualization](09-visualization.md#result-layout-and-empty-output).
Scene preparation and BVH construction operate on fields, while small material
and light tables are prepared on the host. Final image readback is another
explicit boundary. Describe these costs when measuring an end-to-end pipeline.

Scene geometry/BVH preparation is cached. Adding actors changes the scene's
version; editing an actor's existing geometry fields does not automatically
notify the cache. For changing geometry, construct a fresh scene before
rendering so it prepares current geometry and bounds. See the
[in-situ example](https://github.com/berkgeveci/tack/blob/main/examples/37_insitu_pipeline.py).

## Wireframes and points

The unified renderer handles actors with `render_mode="wireframe"` or `"points"`.
`point_size` controls point-disc radius in pixels. Mixed scenes composite these
actors over surface or volume images; a surface provides depth for occlusion,
whereas a standalone volume does not provide surface-hit depth.

Within an actor, equal-depth raster fragments choose the lowest primitive index.
Across actors, later actors win equal-depth ties. Selection uses separate completed
passes for depth, winner selection and colour, so a depth winner cannot be paired
with a competing primitive's RGB. NaN/infinite projected depths are discarded.
See the [rendering contracts](../reference/language-contract.md#atomic-field-updates)
for atomic limitations and [scene example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/41_interactive.py)
for rendering modes.

## Uniform-grid volumes

A `Volume` takes point dimensions and x-fast scalar data. For a flat scalar
field from a uniform grid with `nx`, `ny`, `nz` cells:

```python
from tack.rendering import TransferFunction, Volume

tf = TransferFunction("viridis", opacity_func=lambda t: 0.3 * t, range=(-1.0, 1.0))
volume = Volume(scalar, dims=(nx + 1, ny + 1, nz + 1),
                origin=(-1, -1, -1), spacing=(2 / nx, 2 / ny, 2 / nz),
                transfer_function=tf)
scene.add(volume)
```

The transfer function maps scalars to colour **and opacity**. Fix its range for
comparable frames. Opacity scale, step size and number of steps control how a
ray accumulates samples; the standalone renderer uses trilinear texture sampling
and refreshes the texture snapshot at render time. Surface and volume actors can
also compose in the path tracer. The
[volume example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/42_volume_render.py)
shows complete setups and transfer functions.

## Annotations and output processing

Annotations operate on the read-back NumPy image:

```python
from tack.rendering import TextOverlay, annotate

image = annotate(canvas.to_numpy(), [TextOverlay("Tack render", position=(10, 10))])
```

`ColorBar` shows a colour table's scalar range, and `AxisIndicator` shows the
camera orientation. Drawing uses Pillow; see the
[annotation example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/43_annotated_render.py).
For optional Open Image Denoise processing, the
[denoising example](https://github.com/berkgeveci/tack/blob/main/packages/tack-rendering/examples/36_pathtrace_denoise.py)
uses the binding's explicit device/filter calls. Treat denoising and annotation
as separate postprocessing costs when timing the renderer.

# From a scalar field to an image

Generate a scalar field on a uniform grid, extract its zero level set, and pass
the retained geometry fields to the renderer. A sphere makes the calculation
easy to check independently before trying more complicated implicit shapes.

![An extracted and rendered sphere](../../assets/tutorials/isosurface.png)

**Origin:** adapted from [Tack's isosurface/path-tracing example](https://github.com/berkgeveci/tack/blob/main/examples/32_pathtrace.py),
Copyright © 2026 Kitware, Inc., BSD-3-Clause. This version replaces the gyroid with
an analytic sphere and adds geometric and rendered-depth checks.

## Sample an analytic field

The level set `x² + y² + z² - 0.7² = 0` is a sphere of radius 0.7. A grid with
`n` cells per axis has `(n + 1)³` node samples:

```python
--8<-- "docs/examples/isosurface.py:scalar"
```

The input is flat, with x varying fastest. The index decomposition matches
`UniformGrid`'s point dimensions. This explicit layout also matches the
visualization algorithms' scalar input; it is not an arbitrary 3-D array stride.

## Extract the level set

```python
--8<-- "docs/examples/isosurface.py:extract"
```

`UniformGrid` takes cell dimensions, origin and spacing. `flying_edges` returns
a dictionary or `None` when no surface is produced. Its `points` and `conn` are
NumPy arrays, while `points_field` and `conn_field` retain the flat Tack geometry.

The current implementation reads counts to the host, forms offsets with NumPy,
uploads them, and also produces the NumPy geometry copies. This pipeline
therefore includes host traffic even when the next stage uses retained fields.
Passing those fields to `Actor` avoids downloading and re-uploading geometry
for scene setup; it does not remove earlier extraction transfers.

## Build a scene and render

```python
--8<-- "docs/examples/isosurface.py:render"
```

`Actor` expects interleaved `f32` positions and `i32` triangle indices. A scene
combines geometry and lighting; the camera defines rays; the canvas holds the
image and depth. `smooth=True` computes vertex normals for a smoother appearance.

More samples reduce path-tracing noise. Grid resolution controls geometric
approximation; image resolution controls pixel detail. These are independent
choices. Reading `canvas.to_numpy()` downloads the final image on a device-memory
backend and returns an `(height, width, 4)` `uint8` RGBA array.

## Run and check

```bash
uv run python docs/examples/isosurface.py --check
uv run --with matplotlib python docs/examples/isosurface.py --output sphere.png
```

The small check verifies the vertex/triangle shapes, valid indices and a sphere
residual bounded by the grid's interpolation error. It also requires a substantial
number of visible depth hits. That catches a blank scene even when rendering
returns without an exception.

Try replacing the sphere expression with the gyroid
`sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)` on a larger domain. Keep an explicit
`None` check for thresholds outside the scalar range. See
[Visualization](../09-visualization.md) for multiblock inputs and
[Rendering](../10-rendering.md) for materials, volumes and scalar colouring.

## Complete program

[Download isosurface.py](../../examples/isosurface.py).

```python
--8<-- "docs/examples/isosurface.py"
```

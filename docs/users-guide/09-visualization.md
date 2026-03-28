# Visualization Algorithms

The `tack-vis` package provides GPU-accelerated scientific visualization
algorithms that operate directly on Tack fields.

```bash
pip install tack-vis    # pulls in tack-core automatically
```

## Flying Edges (Isosurface Extraction)

`flying_edges` extracts an isosurface from a scalar field on a uniform grid.
It implements the Flying Edges algorithm with merged unique points — the
output is a triangle mesh ready for rendering or export.

```python
import numpy as np
import tack
from tack.algorithms.flying_edges import flying_edges, UniformGrid

tack.init(arch=tack.metal)

# Define a 64^3 uniform grid
nx, ny, nz = 64, 64, 64
grid = UniformGrid(nx, ny, nz,
                   origin_x=-3.14, origin_y=-3.14, origin_z=-3.14,
                   spacing_x=6.28/nx, spacing_y=6.28/ny, spacing_z=6.28/nz)

# Compute a scalar field (gyroid function)
n_points = (nx + 1) * (ny + 1) * (nz + 1)
scalar = tack.field(dtype=tack.f32, shape=(n_points,))

@tack.kernel
def compute_gyroid(scalar, grid: tack.template(), n_pts):
    for i in range(n_pts):
        ix = i % grid.nx_p1
        iy = (i // grid.nx_p1) % grid.ny_p1
        iz = i // grid.nxy_p1
        x = grid.get_x(ix, iy, iz)
        y = grid.get_y(ix, iy, iz)
        z = grid.get_z(ix, iy, iz)
        scalar[i] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)

compute_gyroid(scalar, grid, n_points)

# Extract isosurface at isovalue = 0
points, conn, n_pts, n_tris = flying_edges(scalar, grid, isovalue=0.0)

print(f"Isosurface: {n_pts} vertices, {n_tris} triangles")
```

The output fields (`points`, `conn`) stay on GPU — no host copy needed
if you pass them directly to the renderer or another algorithm.

### UniformGrid

`UniformGrid` is a `@tack.data_oriented` descriptor that encodes grid
dimensions, origin, and spacing as compile-time constants:

```python
grid = UniformGrid(nx, ny, nz,
                   origin_x, origin_y, origin_z,
                   spacing_x, spacing_y, spacing_z)
```

It provides `@tack.func` methods for coordinate computation:
- `grid.get_x(ix, iy, iz)`, `grid.get_y(...)`, `grid.get_z(...)` — world coordinates
- `grid.nx_p1`, `grid.ny_p1` — point dimensions (nx+1, ny+1)

### Multi-Block

For AMR or domain-decomposed data, `flying_edges_multiblock` processes
multiple blocks into a single unified output:

```python
from tack.algorithms.flying_edges import flying_edges_multiblock

blocks = [(scalar_1, grid_1), (scalar_2, grid_2), ...]
points, conn, n_pts, n_tris = flying_edges_multiblock(blocks, isovalue=0.0)
```

## Compute Normals

`compute_normals` calculates smooth per-vertex normals from a triangle mesh
using atomic scatter-add of face normals:

```python
from tack.algorithms.compute_normals import compute_normals

# points: (n_pts * 3,) f32 field, conn: (n_tris * 3,) i32 field
normals = compute_normals(points, conn, n_pts, n_tris)
```

The entire computation runs on GPU via atomic operations — no host roundtrip.

## Cell to Point

`cell_to_point` averages cell-centered data to vertices:

```python
from tack.algorithms.cell_to_point import cell_to_point

point_data = cell_to_point(cell_data, connectivity, n_points, n_cells)
```

## Parallel Scan

The scan primitives live in `tack-core` (they are general-purpose):

```python
from tack.algorithms import exclusive_scan, inclusive_scan

# Parallel prefix sum on GPU
exclusive_scan(input_field, output_field, n)
total = inclusive_scan(input_field, output_field, n)
```

These are used internally by flying edges and other variable-output algorithms.

## VTK Interop

`tack.interop.vtk` provides zero-copy exchange between VTK arrays and Tack fields:

```python
from tack.interop.vtk import vtk_to_field, field_to_vtk

# VTK → Tack (zero-copy for both host and device arrays)
field = vtk_to_field(vtk_data_array)

# Tack → VTK (host arrays)
vtk_array = field_to_vtk(field, n_components=3)
```

This works with both regular `vtkDataArray` (host memory) and
`vtkmDataArray` (GPU device memory from VTK-m/Viskores). For device
arrays, the GPU pointer is wrapped directly — no host-device copy.

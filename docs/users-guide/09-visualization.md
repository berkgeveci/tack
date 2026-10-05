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

It runs as two kernels on the active backend, accumulating face normals
with `tack.atomic_add`, and nothing is copied to the host.

## Cell to Point

`cell_to_point` averages cell-centered data to vertices:

```python
from tack.algorithms.cell_to_point import cell_to_point

point_data = cell_to_point(cell_data, connectivity, n_points, n_cells)
```

## Parallel Scan

`exclusive_scan` and `inclusive_scan` are general-purpose `tack-core`
functions in `tack.algorithms`, useful for turning per-item output counts
into write offsets. They are described in
[Reductions and Scans](11-reductions-and-scans.md#prefix-scans).

## VTK Interop

`tack.interop.vtk` provides zero-copy exchange between VTK arrays and Tack fields:

```python
from tack.interop.vtk import vtk_to_field, field_to_vtk

# VTK → Tack (zero-copy for both host and device arrays)
field = vtk_to_field(vtk_data_array)

# Tack → VTK (zero-copy)
vtk_array = field_to_vtk(field)
```

A vtkDataArray is *tuples × components* and a Tack field is
n-dimensional, so the shapes line up on their own — a `(1000, 3)` field is
1000 tuples of 3 components, and `field.shape[1]` is the component count.
Nothing has to be declared.

Tack's own visualization algorithms are the exception: `flying_edges` and
`compute_normals` work on *flat interleaved* arrays, so ask for that form
explicitly.

```python
points = vtk_to_field(vtk_points, flatten=True)   # (n*3,)
array  = field_to_vtk(points, n_components=3)     # back to n × 3
```

Both directions go through DLPack, which VTK speaks via
`vtkmodules.util.dlpack_support`. This works with regular `vtkDataArray`
(host memory) and `vtkmDataArray` (device memory from Viskores); for
device arrays the GPU pointer is wrapped directly — no host-device copy.

### Level Zero

On CUDA and HIP a device pointer identifies itself, because the runtime
keeps one context per device for the whole process. A Level Zero pointer
means something only inside the context that allocated it, and DLPack has
no field for a context. So on Level Zero, start Tack inside the context
VTK's Viskores device already uses, before creating any field:

```python
from tack.interop.vtk import init_level_zero

init_level_zero()   # instead of tack.init(arch=tack.level_zero)
```

This needs VTK built with Viskores on Kokkos' SYCL backend, whose
`dlpack_support` provides `level_zero_handles()`. Fields created in a
context of Tack's own cannot be exchanged: `field_to_vtk` and
`vtk_to_field` refuse them, because the Intel driver cannot tell the two
contexts apart and nothing downstream would catch the mistake.

The underlying option is general: `tack.init(arch=tack.level_zero,
external_context={"driver": ..., "device": ..., "context": ...})` adopts
any Level Zero context another library owns. Tack never destroys an
adopted context.

# Scientific visualization

`tack-vis` turns fields into geometric data and provides finite-element helpers
and VTK array interoperability. Its algorithms are ordinary Tack kernels with
Python orchestration; use the same backend initialization as for your own kernels.

```bash
pip install 'tack-core[cpu]' tack-vis
```

For an illustrated complete pipeline, start with [Isosurface to image](tutorials/isosurface.md).
The examples here explain the input layouts and return values you need when
building your own pipeline.

## Flying edges: a uniform-grid level set

`flying_edges` extracts a triangle surface where node scalars cross an isovalue.
`UniformGrid(nx, ny, nz, x0, y0, z0, dx, dy, dz)` describes **cell** dimensions,
origin and spacing; the scalar input has `(nx + 1) * (ny + 1) * (nz + 1)` nodes.
Store scalars flat with x varying fastest.

This complete example generates a gyroid on a small grid:

```python
import tack
from tack.algorithms.flying_edges import UniformGrid, flying_edges

tack.init(arch=tack.cpu)
n = 16
grid = UniformGrid(n, n, n, -3.14, -3.14, -3.14, 6.28 / n, 6.28 / n, 6.28 / n)
scalar = tack.field(dtype=tack.f32, shape=((n + 1) ** 3,))

@tack.kernel
def gyroid(scalar, grid: tack.template(), count):
    for idx in range(count):
        i = idx % grid.nx_p1
        j = (idx // grid.nx_p1) % grid.ny_p1
        k = idx // grid.nxy_p1
        x, y, z = grid.get_x(i, j, k), grid.get_y(i, j, k), grid.get_z(i, j, k)
        scalar[idx] = sin(x) * cos(y) + sin(y) * cos(z) + sin(z) * cos(x)

gyroid(scalar, grid, scalar.size)
mesh = flying_edges(scalar, grid, isovalue=0.0)
if mesh is not None:
    print(mesh["total_points"], "vertices,", mesh["total_tris"], "triangles")
```

`UniformGrid` is a data-oriented descriptor. Its dimensions, origin and spacing
are instance attributes supplied to kernels as runtime parameters; field shapes
used for indexing can still specialize the generated code.

### Result layout and empty output

The result is a dictionary, or `None` when no surface is produced:

| Key | Value |
|---|---|
| `points` | NumPy `float32` coordinates, `(number_of_points, 3)` |
| `conn` | NumPy `int32` triangles, `(number_of_triangles, 3)` |
| `points_field` | Retained Tack `f32` field, flat interleaved XYZ |
| `conn_field` | Retained Tack `i32` field, flat triangle vertex indices |
| `total_points`, `total_tris` | Python counts |
| `block_point_counts`, `block_tri_counts` | Counts for each input block |

Use the NumPy arrays for export to a CPU library. Use the retained fields for
Tack normal calculation or rendering. Always handle `None` before accessing keys.

The current algorithm reads per-row counts to the host and constructs prefix
sums with NumPy. It also downloads the output geometry for the NumPy result keys.
Retained fields avoid another geometry upload at the next Tack stage, but
extraction is not a pipeline without host transfers.

### Multiple blocks

`flying_edges_multiblock` takes a list of dictionaries:

```python
from tack.algorithms.flying_edges import flying_edges_multiblock

blocks = [
    {"scalar": scalar_a, "grid": grid_a},
    {"scalar": scalar_b, "grid": grid_b},
]
mesh = flying_edges_multiblock(blocks, isovalue=0.0)
```

Each block may also provide an `i32` cell `mask`, with nonzero entries suppressing
cells. An omitted mask is filled with zeros. Outputs occupy unified fields with
per-block offsets; vertices are not welded between independently processed blocks.
For AMR data, appropriate cell blanking is needed to avoid extracting overlapping
coarse/refined regions. See Tack's [multiblock example](https://github.com/berkgeveci/tack/blob/main/packages/tack-vis/examples/31_multiblock_flying_edges.py).

## Vertex normals

For a nonempty result:

```python
from tack.algorithms.compute_normals import compute_normals

normals = compute_normals(mesh["points_field"], mesh["conn_field"],
                          mesh["total_points"], mesh["total_tris"])
```

This returns flat interleaved `f32` normal components in a Tack field. One kernel
atomically accumulates area-weighted face normals at their vertices; another
normalizes them. The output stays in a field. For convenience, an `Actor` can
compute these normals when created with `smooth=True`.

## Cell-centred values to grid nodes

`cell_to_point` averages adjacent cell values on a **uniform structured grid**;
it does not take an arbitrary connectivity array:

```python
from tack.algorithms.cell_to_point import cell_to_point

# cell_data is flat, x-fast, with nx * ny * nz values.
point_data = cell_to_point(cell_data, nx, ny, nz)
```

The output has `(nx + 1) * (ny + 1) * (nz + 1)` values and preserves the input's
`f32` or `f64` dtype, within backend support. Interior nodes average eight cells;
faces, edges and corners use fewer. This is a common preparation step before
contouring cell-centred simulation output. For the reverse averaging direction,
see the [point-to-cell example](https://github.com/berkgeveci/tack/blob/main/packages/tack-vis/examples/25_point_to_cell.py).

## Finite-element fields

The `tack.fe` module supplies Lagrange bases, geometry maps and accessors for
contiguous or gathered degrees of freedom. Finite-element sampling is distinct
from interpolating values on a uniform Cartesian grid: basis functions map
parametric cell coordinates to the field and geometry.

The [FE isoline example](https://github.com/berkgeveci/tack/blob/main/packages/tack-vis/examples/40_fe_isoline.py)
and [FE isosurface example](https://github.com/berkgeveci/tack/blob/main/packages/tack-vis/examples/41_fe_isosurface.py)
show complete setups. They are good next steps after understanding grid layout,
templates and local-array scratch storage. Tack does not yet offer every cell
and filter of a general VTK-like dataset API.

## Cell shapes

`tack.data.shapes` defines VTK's ten linear cells (`Vertex`, `Line`,
`Triangle`, `Pixel`, `Quad`, `Tetra`, `Voxel`, `Hexahedron`, `Wedge` and
`Pyramid`) as `@tack.data_oriented` classes. They follow VTK exactly: the cell type
id (`Hexahedron.ID` is 12), the point order, the parametric coordinates, the
edges and faces, and the shape functions. A kernel takes a shape object as a
template argument and calls its methods, so each compiled variant is
specialized to one shape and does no dispatch on it.

The methods give a shape's parametric points and center, its shape
functions and their gradients, its edges and faces as indices into the
cell's points, and an inside test. On a cell whose points the kernel has
gathered into a local array, they also interpolate values and positions,
form the Jacobian, and invert a world point to parametric coordinates
with Newton's method. See the [API reference](../reference/api.md#cell-shapes)
for the list.

A mesh with several kinds of cell runs a kernel once per shape present. Here
the input is a VTK unstructured grid's arrays (`types`, `offsets`,
`connectivity`), and the result is each cell's position at its parametric
center:

```python
import numpy as np
import tack
from tack.data import shapes


@tack.kernel
def centers(cell, conn, xyz, out):
    for c in range(out.shape[0]):
        pc = cell.parametric_center()
        x = tack.Vector([0.0, 0.0, 0.0])
        for j in range(cell.NUM_POINTS):
            x += cell.shape_function(j, pc) * xyz[conn[c, j]]
        out[c] = x


def cell_centers(types, offsets, connectivity, points):
    """Parametric centers of a VTK unstructured grid's cells, one launch per shape."""
    xyz = tack.Vector.field(3, tack.f32, shape=(len(points),))
    xyz.from_numpy(points)
    result = np.empty((len(types), 3), np.float32)
    for type_id in np.unique(types):
        cell = shapes.shape_class(type_id)()
        ids = np.flatnonzero(types == type_id)
        conn = tack.field(tack.i32, shape=(len(ids), cell.NUM_POINTS))
        conn.from_numpy(np.stack([connectivity[offsets[c]:offsets[c + 1]] for c in ids]))
        out = tack.Vector.field(3, tack.f32, shape=(len(ids),))
        centers(cell, conn, xyz, out)
        result[ids] = out.to_numpy(vectors=True)
    return result


# A unit cube and the tetrahedron on its top face.
points = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 2)], np.float32)
types = np.array([shapes.HEXAHEDRON, shapes.TETRA])
offsets = np.array([0, 8, 12])
connectivity = np.array([0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 7, 8], np.int32)
print(cell_centers(types, offsets, connectivity, points))
# [[0.5  0.5  0.5 ]
#  [0.25 0.25 1.25]]
```

Grouping the cells by shape happens on the host here; the variants it
selects are compiled once per shape and reused.

## VTK interop

`tack.interop.vtk` exchanges compatible VTK arrays and fields through DLPack:

```python
from tack.interop.vtk import vtk_to_field, field_to_vtk

field = vtk_to_field(vtk_array)
vtk_array = field_to_vtk(field)
```

VTK's tuples and components become a field shape: a `(1000, 3)` scalar field
represents 1000 tuples with three components. Flying edges and normal calculation
instead use flat interleaved fields; specify that representation explicitly:

```python
points = vtk_to_field(vtk_points_array, flatten=True)
vtk_array = field_to_vtk(points, n_components=3)
```

This requires a VTK build exposing `vtkmodules.util.dlpack_support`. Compatible
host/device arrays can share storage; device/context support still matters, and
not every VTK installation provides device arrays. See
[Sharing arrays with other libraries](16-interoperability.md) for lifetime and
read-only rules.

### Level Zero

Level Zero device pointers need a shared allocation context. With a compatible
VTK/Viskores Kokkos-SYCL build, initialize before allocating fields:

```python
from tack.interop.vtk import init_level_zero

init_level_zero()                       # instead of tack.init(arch=tack.level_zero)
```

The VTK DLPack support must provide `level_zero_handles()`. Fields created in a
separate Tack context cannot be exchanged through this path. For another library,
`tack.init(arch=tack.level_zero, external_context={"driver": ..., "device": ...,
"context": ...})` adopts its context; Tack never destroys an adopted context.

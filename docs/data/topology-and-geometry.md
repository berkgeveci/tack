# Topology and geometry

A Tack dataset keeps its cells and its point positions apart. The cells are
the **topology**: which points each cell connects. The positions are the
**geometry**. This page explains both, for readers who know VTK's points and
cells. After it you can tell which topology and which geometry hold your
mesh, use positions that are computed rather than stored, and find a mesh's
faces and edges.

## One topology, three geometries

A `vtkImageData`, a `vtkRectilinearGrid` and a `vtkStructuredGrid` of the
same dimensions have the same cells. They differ only in where the points
are. In Tack they can share one topology object, each dataset with its own
geometry. The example makes a grid of 3 x 2 x 2 points three times, once
with each kind of geometry.

<!-- tack-check: one-topology -->
```python
import numpy as np

import tack
import tack.dataset as td
from tack.arrays import CartesianProduct, UniformCoordinates

tack.set_default_device("cpu")

cells = td.StructuredTopology((3, 2, 2))            # 12 points, 2 hexahedra

# An image: positions from an origin and a spacing, stored nowhere.
image = td.DataSet(cells, UniformCoordinates((3, 2, 2), spacing=(0.5, 1.0, 1.0)))

# A rectilinear grid: one array of coordinates per axis.
rectilinear = td.DataSet(cells, CartesianProduct([0.0, 0.25, 1.0], [0.0, 1.0], [0.0, 1.0]))

# A curvilinear grid: a position per point, here the image's, bent upward.
bent = image.positions()
bent[:, 2] += 0.1 * bent[:, 0] * bent[:, 1]
curvilinear = td.DataSet(cells, bent)

for data in (image, rectilinear, curvilinear):
    print(type(data.geometry.values).__name__, data.positions()[5])
# UniformCoordinates [1. 1. 0.]
# CartesianProduct [1. 1. 0.]
# Array [1.  1.  0.1]
assert image.topology is rectilinear.topology is curvilinear.topology
```

Point 5 is the point at grid index (i, j, k) = (2, 1, 0). It has the same
index in all three datasets, because the topology numbers the points. Its
position differs, because the geometry places them.

## The topology is the cells

A topology lists the cells and the points each one connects. It holds no
positions. Each cell has **corners**, numbered in the order its shape
defines. Tack uses VTK's corner order for every shape, so a hexahedron's
corners 0 to 3 are its bottom face and 4 to 7 its top, as in VTK. The
topology maps each corner to a **point id**, the index of a point in the
dataset.

There are two ways to say which points a cell uses:

- A `StructuredTopology` is a grid of points numbered x fastest. Cell
  (i, j, k) connects the points around it, so the topology stores nothing
  but the grid's dimensions. Images, rectilinear grids and curvilinear grids
  all use one.
- An `UnstructuredTopology` stores what a `vtkUnstructuredGrid` stores: a VTK
  cell type per cell, offsets, and the connectivity, the point ids of each
  cell in turn. Its cells may be of any mix of types.

Polygons, polylines and polyhedra have topologies of their own, with point
rings and face lists instead of a fixed corner order. They are described in
[Polyhedra, polygons and polylines](polyhedra.md). [Cells and meshes](cells-and-meshes.md)
lists the cell types and shows how to build each kind of mesh.

## The geometry is a field of positions

The geometry gives each point a position, a 3-vector, even for a flat mesh.
It plays the part of `vtkPoints`. In Tack the geometry is itself a field,
with a 3-vector per point, stored as point data is. `data.geometry` returns
it, and [Fields](fields.md) explains fields in general. Algorithms treat it apart from the other fields:
they move it with the cells, and never list it among the fields they carry
or select.

You can give `DataSet` the geometry in several forms. A NumPy array of
shape `(num_points, 3)` is copied to the device. A Tack array of 3-vectors
is used as it is. An implicit array, described in the next section, computes
the positions. Replacing the geometry moves the points without touching the
cells:

<!-- tack-check: move-points -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))
cells = data.topology

data.geometry = data.positions() * 2.0              # every point twice as far out
assert data.topology is cells
print(data.positions()[11])                         # [2. 2. 2.]
```

`data.positions()` returns the positions as a NumPy array on the host,
whatever form the geometry has. It copies from the device, so use it to
inspect a dataset rather than inside a computation.

Two further features of the geometry have their own pages. A geometry can
be of higher order than the cells' corners, which makes cells with curved
edges and faces. [Higher-order and finite-element data](higher-order.md)
explains it. Positions are also stored relative to an origin, which keeps
coordinates far from zero accurate in single precision.
[Coordinates, units and time](units-coordinates-time.md) explains the
origin.

## Positions that are computed, not stored

An **implicit array** computes its values when a kernel reads them instead
of storing them, as VTK's implicit arrays do. Tack's are in `tack.arrays`,
and [Storage and layouts](storage-and-layouts.md) describes them all. Two of
them are geometries:

| Geometry values | Position of point (i, j, k) | Stored | VTK |
|---|---|---|---|
| `UniformCoordinates(dims, origin, spacing, direction)` | `origin + direction @ ((i, j, k) * spacing)` | the origin, spacing and direction | `vtkImageData`, with its direction matrix |
| `CartesianProduct(x, y, z)` | `(x[i], y[j], z[k])` | the three axes | `vtkRectilinearGrid` |
| a Tack array of 3-vectors | one stored position per point | every position | `vtkStructuredGrid`, `vtkPoints` |

An image of a billion points stores a few numbers, and a rectilinear grid
stores `nx + ny + nz` coordinates for `nx * ny * nz` points. Kernels read
both as if every position were stored. `td.uniform_grid` and
`td.rectilinear_grid` make a `StructuredTopology` with these geometries, so
you rarely build them by hand.

The direction matrix lets an image be rotated. A structured topology does
not imply that the grid is aligned with the axes.

## Cells that are computed, not stored

The arrays of an `UnstructuredTopology` may be implicit too. A mesh whose
cells are all one type needs no array of types. A mesh whose cells all have
the same number of points needs no array of offsets, since offset `c` is
`c` times that number. `ConstantArray` and `CountingArray` express these,
like VTK's constant and affine arrays.

<!-- tack-check: implicit-cells -->
```python
import numpy as np

import tack
import tack.dataset as td
from tack.arrays import ConstantArray, CountingArray

tack.set_default_device("cpu")

positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float32)
count = 2
types = ConstantArray(td.shapes.Tetra.ID, count, dtype=tack.u8)   # every cell VTK type 10
offsets = CountingArray(count + 1, start=0, step=4)              # 0, 4, 8
cells = td.UnstructuredTopology(types, offsets, [0, 1, 2, 3, 1, 2, 3, 4])
mesh = td.DataSet(cells, positions)

print(mesh.num_cells, "tetrahedra,", type(cells.offsets).__name__)   # 2 tetrahedra, CountingArray
assert mesh.num_points == 5
```

Filters whose output has one cell type, such as a contour's triangles, make
their topologies this way. Code that reads a topology's arrays on the host
should therefore accept an implicit array. `tack.arrays.to_host` turns any
array, implicit or not, into a NumPy array.

## Faces and edges

VTK computes a cell's faces and edges when you ask a cell for them. Tack
derives a mesh's faces and edges once, as entities of their own with ids,
and keeps them on the topology. `topology.faces()` and `topology.edges()`
make them on first use. The first call costs a pass over the cells, and
later calls return the kept result.

A **face** is one dimension below the mesh's cells. The faces of a 3D mesh
are polygons, and the faces of a 2D mesh are its cells' edges. VTK says a 2D
cell has no faces, so this is one place where the two differ. A face shared
by two cells is one face with two **sides**. A face on the boundary has one.

```mermaid
flowchart LR
    C0["cell on side 0"] --- F["shared face"] --- C1["cell on side 1"]
    F -. "the face's normal points out of side 0" .-> C1
```

Each face stores its points in the order that winds around side 0's outward
normal. A face value with a direction, such as a normal velocity or a flux,
is measured along that normal. [Fields](fields.md#values-on-faces-and-edges)
explains how to declare such a value.

<!-- tack-check: faces-and-edges -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))   # two hexahedra side by side
faces = data.topology.faces()
edges = data.topology.edges()

print(faces.num_faces, "faces,", edges.num_edges, "edges")   # 11 faces, 20 edges
boundary = faces.boundary().numpy()                          # the faces with one side
print(len(boundary), "on the boundary")                      # 10 on the boundary

sides = faces.sides.numpy()          # per face: cell, local face, cell, local face
shared = [f for f in range(faces.num_faces) if sides[f, 2] >= 0]
print(sides[shared])                 # [[0 1 1 0]]: cell 0's face 1 is cell 1's face 0
assert len(shared) == 1 and len(boundary) == 10
```

The two hexahedra have six faces each, and the one they share counts once,
so the mesh has 11 faces. In `sides`, the second pair is -1 for a boundary
face. Face ids are stable for one topology. They are not shared with another
solver's numbering of the same mesh.

## Details

**Point order in structured grids.** A structured grid's points are numbered
`p = i + nx * (j + ny * k)`, with x fastest, as in VTK. A NumPy array of
shape `(nz, ny, nx)`, indexed `host[k, j, i]`, flattens into this order
directly. An array indexed `host[i, j, k]` must be transposed first. A field
that depends on x alone is a quick check of the order before you upload a
whole simulation.

**Structured dimensions count points.** A grid of `(nx, ny, nz)` points has
`(nx - 1) * (ny - 1) * (nz - 1)` cells. `StructuredTopology` takes one to
three dimensions of at least two points each, and makes lines, quads or
hexahedra. `uniform_grid` and `rectilinear_grid` also accept a dimension of
one point, which adds no cells along that axis.

**Ids.** A topology stores point ids as 32-bit integers when they fit, and
as 64-bit integers when the mesh is too large or you pass
`id_dtype=tack.i64`. The same type covers the offsets, the connectivity and
the ids of derived faces and edges.

**Unused points.** An `UnstructuredTopology` has as many points as one past
the highest id it uses. Pass `num_points=` when the geometry has points no
cell uses, as a VTK dataset may.

**Lines have no faces.** A mesh whose highest-dimensional cells are lines
raises an error from `faces()`, because its faces would be single points. A
`PolylineTopology` reports its chains' ends instead.

**A grid inside a larger grid.** `StructuredTopology(..., offset=)` gives the
index of the grid's first point in a larger grid, as the start of a VTK
extent does. Pieces of a split grid use it, as [Data in pieces](pieces.md) explains.

## Where to go next

- [Fields and where values live](fields.md) puts values on the points,
  cells, faces and edges this page described.
- [Cells and meshes](cells-and-meshes.md) lists the cell types and builds
  each kind of mesh.
- The [data model reference](../vis-guide/reference/data-model.md#topologies)
  has every topology's full signature.

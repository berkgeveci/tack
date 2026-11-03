# Cells and meshes

This page lists the cells a Tack dataset can hold and the kinds of mesh they
make: images, rectilinear and curvilinear grids, unstructured grids of mixed
cells, polygons, polylines, polyhedra and point clouds. It is written for
readers who know VTK's dataset types, and maps each to Tack's. After it you
can choose the topology for your mesh and build it.

## A mesh of mixed cells

VTK's `vtkUnstructuredGrid` holds cells of any type side by side, and so does
Tack's `UnstructuredTopology`. This mesh has a cube, a wedge glued to its
side and a pyramid on its top. Each cell lists its corners in VTK's order.

<!-- tack-check: mixed-cells -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],    # 0-3: the cube's bottom
                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],    # 4-7: its top
                      [2, 0.5, 0], [2, 0.5, 1],                      # 8-9: the wedge's far edge
                      [0.5, 0.5, 1.6]], np.float32)                  # 10: the pyramid's apex

HEXAHEDRON, WEDGE, PYRAMID = (td.shapes.Hexahedron.ID, td.shapes.Wedge.ID,
                              td.shapes.Pyramid.ID)                  # VTK types 12, 13, 14
cells = td.UnstructuredTopology(
    types=[HEXAHEDRON, WEDGE, PYRAMID],
    offsets=[0, 8, 14, 19],
    connectivity=[0, 1, 2, 3, 4, 5, 6, 7,        # the cube
                  1, 8, 2, 5, 9, 6,              # the wedge, on the cube's face 1-2-6-5
                  4, 5, 6, 7, 10])               # the pyramid, on the cube's top
mesh = td.DataSet(cells, positions)

print([(group.shape.__name__, group.count) for group in cells.groups()])
# [('Hexahedron', 1), ('Wedge', 1), ('Pyramid', 1)]
faces = cells.faces()
print(faces.num_faces, "faces,", len(faces.boundary().numpy()), "on the boundary")
# 14 faces, 12 on the boundary
assert mesh.num_cells == 3 and faces.num_faces == 6 + 5 + 5 - 2
```

The arrays are a `vtkUnstructuredGrid`'s: a VTK cell type per cell, offsets
and the connectivity, with each cell's corners in VTK's order. The offsets have
one more entry than there are cells, and cell `c` uses the connectivity entries
from `offsets[c]` up to `offsets[c + 1]`. Giving `DataSet` a NumPy array of
positions makes its geometry, one 3-vector per point.

The three cells have 16 faces between them. Two are shared, so the mesh has
14. `cells.groups()` sorts the cells by shape, which is how kernels see a
mixed mesh: once per shape present, with no test of the type inside the
loop.

## The cell types

A cell's shape fixes its number of corners and their order. Most shapes have
a **reference element**, a fixed cell in parametric coordinates that every
cell of the shape is a map of. Values inside a cell are computed there, from
the shape's interpolation functions. Tack's shapes are VTK's linear cells,
with VTK's type ids, corner orders and parametric coordinates, so VTK's
documentation of a cell applies to Tack's.

| Shape | Dimension | Corners | Reference element | VTK type |
|---|---|---|---|---|
| `Vertex` | 0 | 1 | a point | `VTK_VERTEX` (1) |
| `Line` | 1 | 2 | `[0, 1]` | `VTK_LINE` (3) |
| `Triangle` | 2 | 3 | the unit triangle | `VTK_TRIANGLE` (5) |
| `Pixel` | 2 | 4 | `[0, 1]²`, corners x fastest | `VTK_PIXEL` (8) |
| `Quad` | 2 | 4 | `[0, 1]²`, corners counterclockwise | `VTK_QUAD` (9) |
| `Tetra` | 3 | 4 | the unit tetrahedron | `VTK_TETRA` (10) |
| `Voxel` | 3 | 8 | `[0, 1]³`, corners x fastest | `VTK_VOXEL` (11) |
| `Hexahedron` | 3 | 8 | `[0, 1]³`, each face counterclockwise | `VTK_HEXAHEDRON` (12) |
| `Wedge` | 3 | 6 | the unit triangle times `[0, 1]` | `VTK_WEDGE` (13) |
| `Pyramid` | 3 | 5 | `[0, 1]³` with its top collapsed to the apex | `VTK_PYRAMID` (14) |
| `PolyLine` | 1 | any | none | `VTK_POLY_LINE` (4) |
| `Polygon` | 2 | any | none | `VTK_POLYGON` (7) |
| `Polyhedron` | 3 | any | none | `VTK_POLYHEDRON` (42) |

The shapes are in `td.shapes`, and each has its VTK id as `ID`. The last
three have no reference element and live in topologies of their own,
described below. VTK's quadratic, Lagrange and Bézier cells have no shape of
their own in Tack. The cell stays linear, and a higher-order field or
geometry adds the values on its edges, faces and interior. See
[Higher-order and finite-element data](higher-order.md).

## Structured grids: images, rectilinear and curvilinear grids

A `StructuredTopology` is a grid of points numbered x fastest, with a cell
between each pair of neighbouring points along every axis. It stores only
its dimensions. VTK's three structured types differ only in their geometry,
so in Tack they share this one topology, each with a different geometry:

| VTK | Make it with | Geometry |
|---|---|---|
| `vtkImageData` | `td.uniform_grid(dims, origin, spacing, direction)` | computed from origin, spacing and direction |
| `vtkRectilinearGrid` | `td.rectilinear_grid(x, y, z)` | computed from one array per axis |
| `vtkStructuredGrid` | `td.DataSet(td.StructuredTopology(dims), positions)` | one position per point |

[Topology and geometry](topology-and-geometry.md#positions-that-are-computed-not-stored)
explains the computed geometries.

<!-- tack-check: structured-grids -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

image = td.uniform_grid((9, 5), spacing=(0.25, 0.25))                  # a 2D image
rectilinear = td.rectilinear_grid([0.0, 0.1, 0.3, 0.7], [0.0, 1.0], [0.0, 0.5, 1.0])

# A curvilinear grid: a quarter annulus, one position per point, x fastest.
radius, angle = np.meshgrid(np.linspace(1, 2, 5), np.linspace(0, np.pi / 2, 9))
ring = np.stack([radius * np.cos(angle), radius * np.sin(angle), 0 * radius], -1)
annulus = td.DataSet(td.StructuredTopology((5, 9)), ring.reshape(-1, 3).astype(np.float32))

for name, grid in (("image", image), ("rectilinear", rectilinear), ("annulus", annulus)):
    shape = grid.topology.groups()[0].shape.__name__
    print(f"{name:12} {grid.num_points:3} points {grid.num_cells:3} cells of {shape}")
# image         45 points  32 cells of Quad
# rectilinear   24 points   6 cells of Hexahedron
# annulus       45 points  32 cells of Quad
assert annulus.num_cells == (5 - 1) * (9 - 1)
```

Dimensions count points, as in VTK, so a grid of `(nx, ny, nz)` points has
`(nx - 1) * (ny - 1) * (nz - 1)` cells. Two dimensions make a 2D grid of
quads, and one makes a line of segments. `uniform_grid` and
`rectilinear_grid` also take a dimension of one point, which adds no cells
along that axis, so `(9, 5, 1)` is the same 2D image as `(9, 5)`. A grid with
no axis of two or more points has no cells.

A structured grid's cells are hexahedra and quads, in `vtkStructuredGrid`'s
corner order, even for an image. VTK calls an image's cells voxels and
pixels, whose corners are in x-fastest order. The cells and positions are
the same. Only the numbering of a cell's corners differs, which matters if
you read corners by number in a kernel.

## Unstructured grids

An `UnstructuredTopology` takes the three arrays of a `vtkUnstructuredGrid`:

- `types`: a VTK cell type per cell.
- `offsets`: where each cell's points start in `connectivity`, with one more
  entry than there are cells. Cell `c` uses entries `offsets[c]` up to
  `offsets[c + 1]`.
- `connectivity`: the point ids of every cell, one cell after another.

The entries of `connectivity` are point ids, not byte offsets or component
indices. They must follow the shape's corner order. The right points in the
wrong order can turn a cell inside out or change its faces, as in VTK. Pass
`num_points=` when the geometry has points that no cell uses.

One topology may mix every shape that has a reference element, of any
dimension: tetrahedra beside hexahedra, triangles beside tetrahedra. The
faces of a mixed mesh are those of its highest-dimensional cells. The
arrays may be NumPy arrays, lists or Tack arrays, and they may be implicit,
as [Topology and geometry](topology-and-geometry.md#cells-that-are-computed-not-stored)
shows for a mesh of one cell type.

## Polygons, polylines and polyhedra

A polygon, a polyline or a polyhedron has any number of points, so it has no
fixed corner order and no reference element. Each kind has a topology of its
own:

| Topology | Its cells | VTK |
|---|---|---|
| `PolygonalTopology` | polygons, each a ring of points | `VTK_POLYGON`, the polygons of a `vtkPolyData` |
| `PolylineTopology` | polylines, each a chain of points | `VTK_POLY_LINE`, the lines of a `vtkPolyData` |
| `PolyhedralTopology` | polyhedra, each a list of faces | `VTK_POLYHEDRON` |

Triangles and quads are polygons too, and a line is a two-point polyline.
These topologies hold one kind of cell each, so polygons cannot sit beside
tetrahedra in one mesh. A field on such cells is linear on pieces of each
cell, such as the triangles that fan a polygon from its centre.
[Polyhedra, polygons and polylines](polyhedra.md) explains how values live in them and
how to build them.

## Point clouds

A point cloud is a set of positions with point data and no cells that
connect them. `td.point_cloud(positions, fields=...)` makes one, with one
vertex cell per point. The vertex cells are implicit arrays, so they cost no
memory. [Particles and point clouds](particles.md) explains particles and
the searches over them.

## Which topology holds which cells

| Topology | Its cells | VTK types that map to it |
|---|---|---|
| `StructuredTopology` | lines, quads or hexahedra of a grid, numbered by index | `vtkImageData`, `vtkRectilinearGrid`, `vtkStructuredGrid` |
| `UnstructuredTopology` | any mix of the shapes with a reference element | `vtkUnstructuredGrid`, or a `vtkPolyData` of triangles and quads |
| `PolygonalTopology` | polygons | a `vtkPolyData` with larger polygons |
| `PolylineTopology` | polylines | a `vtkPolyData` of lines |
| `PolyhedralTopology` | polyhedra | a `vtkUnstructuredGrid` of `VTK_POLYHEDRON` cells |

When the VTK interop reads a `vtkPolyData`, a surface of triangles and quads
becomes an `UnstructuredTopology`, and its triangle strips become triangles.
A `vtkPolyData` of lines becomes a `PolylineTopology`. Polygons or polylines
beside other kinds of cell are refused. See
[VTK](../visualization/vtk.md).

## More than one mesh

Some data are more than one mesh, and each such kind has its own page in
this layer:

- A mesh split into pieces over blocks or ranks is a `PartitionedDataSet`,
  VTK's `vtkPartitionedDataSet`. See [Data in pieces](pieces.md).
- Levels of refined boxes are an `AMRDataSet`, VTK's `vtkOverlappingAMR`. A
  tree of refined cells is an `OctreeDataSet`, VTK's `vtkHyperTreeGrid`. See
  [AMR hierarchies and octrees](amr-and-octrees.md).

## Details

**Faces of 2D cells.** VTK says a 2D cell has no faces. Tack calls a 2D
cell's edges its faces, so a surface mesh has faces, as a polygonal one does.
A mesh whose highest-dimensional cells are lines has no faces.

**Degenerate hexahedra.** Some file formats write wedges, pyramids and
tetrahedra as hexahedra with repeated corners. The VTK interop turns them
into those shapes when it reads them.

**Inside-out cells.** A cell whose corners are in the reverse of VTK's
order has a negative volume and inward faces. The VTK interop finds such
cells on reading and warns, and can turn them outward.

## Where to go next

- [Polyhedra, polygons and polylines](polyhedra.md) covers the cells without a
  reference element.
- [Higher-order and finite-element data](higher-order.md) covers fields and
  geometry above order 1 on these cells.
- [Build meshes and fields](build-meshes-and-fields.md) builds each kind of
  mesh from your own arrays.

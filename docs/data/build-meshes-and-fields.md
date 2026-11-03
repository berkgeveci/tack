# Build meshes and fields

This page shows how to make a dataset from your own numbers: the mesh, the
fields on it, and named selections of its points, cells and faces. It is for
anyone whose data does not come from a reader, such as a test case, a
synthetic field or a mesh generator's output. After it you can build every
kind of mesh Tack holds, from an image to an unstructured grid of mixed cells,
and attach point data, cell data and higher-order fields to it. What the parts
mean is explained in [Topology and geometry](topology-and-geometry.md) and
[Fields and where values live](fields.md). This page is about the calls.

You build three parts: a mesh (its topology and geometry), fields, and sets.
This dataset is an image of 4 x 3 x 2 cells with a height at each point, a
material in each cell, and its boundary faces as a set:

<!-- tack-check: build-first -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# The mesh: an image of 5 x 4 x 3 points, as VTK's vtkImageData.
data = td.uniform_grid((5, 4, 3), spacing=(0.25, 1 / 3, 0.5), unit="m")

# Fields from arrays: point data and cell data.
x = data.positions()                                  # a host copy, (60, 3)
height = tack.array(x[:, 2].astype(np.float32))
data.fields["height"] = td.Field(td.H1(data), height, unit="m")
material = tack.array(np.arange(data.num_cells) % 2, dtype=tack.i32)
data.fields["material"] = td.Field(td.Constant(data), material)

# A set: the faces with one cell.
data.sets["boundary"] = td.boundary_faces(data)

print(data.num_points, data.num_cells, len(data.sets["boundary"]))   # 60 24 52
assert data.fields["height"].space.size == data.num_points
assert data.fields["material"].space.size == data.num_cells
```

A field is a space and an array of values. `td.H1(data)` is the space of an
`H1` field (point data): one value per point, interpolated linearly inside each
cell. `td.Constant(data)` is the space of a `Constant` field (cell data): one
value per cell. The rest of the page goes through each part in turn.

## Choose a constructor

Each kind of mesh VTK has maps onto one Tack constructor. The first column
names the VTK class you may already use.

| VTK | Tack | Geometry stored |
|---|---|---|
| `vtkImageData` | `td.uniform_grid(dims, origin, spacing, direction)` | nothing: a formula |
| `vtkRectilinearGrid` | `td.rectilinear_grid(x, y, z)` | one array per axis |
| `vtkStructuredGrid` | `td.DataSet(td.StructuredTopology(dims), positions)` | a position per point |
| `vtkUnstructuredGrid` | `td.DataSet(td.UnstructuredTopology(types, offsets, connectivity), positions)` | a position per point |
| `vtkUnstructuredGrid` of polyhedra | `td.PolyhedralTopology.from_cell_faces(...)` | a position per point |
| `vtkPolyData` of polygons or lines | `td.PolygonalTopology`, `td.PolylineTopology` | a position per point |
| `vtkPolyData` of vertices | `td.point_cloud(positions, fields)` | a position per point |
| `vtkOverlappingAMR` | `td.AMRDataSet` ([AMR and octrees](amr-and-octrees.md)) | per box |
| `vtkHyperTreeGrid` | `td.OctreeDataSet` ([AMR and octrees](amr-and-octrees.md)) | per root cell |
| `vtkPartitionedDataSet` | `td.PartitionedDataSet` ([Data in pieces](pieces.md)) | per piece |

The grid constructors and the point cloud take `unit=` for the spatial unit
and `dtype=` for the coordinates' floating-point type. They also take
`device=` to make the dataset on a device other than the default one
([Data on devices](devices.md)).

## Build a structured grid

A structured grid's topology is its point dimensions alone: no connectivity
is stored. Its geometry can be a formula, one array per axis, or a position
per point. The three examples below make the same two hexahedra in the unit
cube, so you can compare them. They assume `numpy`, `tack` and `tack.dataset`
are imported as `np`, `tack` and `td`.

An image's positions are computed from its origin and spacing whenever a kernel
reads them. `tack.arrays.UniformCoordinates` is that formula, an implicit
array:

```python
--8<-- "docs/examples/data_meshes.py:uniform"
```

A rectilinear grid keeps one array of coordinates per axis.
`tack.arrays.CartesianProduct` combines them into a position per point without
storing one:

```python
--8<-- "docs/examples/data_meshes.py:rectilinear"
```

A curvilinear grid has the same topology and an explicit position for every
point. The positions are an array of 3-vectors (`tack.vec3()`), one per point,
in the order described below:

```python
--8<-- "docs/examples/data_meshes.py:curvilinear"
```

For images and rectilinear grids the shortcuts build the same topology and
geometry in one call:

```python
--8<-- "docs/examples/data_meshes.py:grids"
```

### Count points, not cells

Dimensions count points. A grid of `(nx, ny, nz)` points has
`(nx - 1) * (ny - 1) * (nz - 1)` cells, so `(3, 2, 2)` is two hexahedra.
In an image, the spacing is the distance between neighbouring points.

A dimension of 1 adds no cells. So `uniform_grid((9, 5, 1))` is a 2D grid of
quadrilaterals, and `uniform_grid((1, 3))` is a line of two cells. A grid with
no dimension of two or more points has points and no cells, and a dimension of
0 makes an empty grid. `StructuredTopology` itself takes one to three
dimensions of at least two points; the shortcuts handle the other cases.

### Number points x fastest

Tack numbers a structured grid's points with x fastest, as VTK does: point
`(i, j, k)` is number `i + nx * (j + ny * k)`. Values you upload must follow the
same order. A C-ordered NumPy volume of shape `(nz, ny, nx)`, indexed
`[k, j, i]`, already does, so `ravel()` is enough. A volume indexed
`[i, j, k]`, as a Fortran code stores one, needs `ravel(order="F")`:

<!-- tack-check: build-x-fastest -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

nx, ny, nz = 5, 4, 3
data = td.uniform_grid((nx, ny, nz))             # spacing 1: positions are indices
i, j, k = data.positions().T

# A C-ordered volume indexed [k, j, i].
c_volume = np.fromfunction(lambda k, j, i: i + 10 * j + 100 * k, (nz, ny, nx),
                           dtype=np.float32)
data.fields["c"] = td.Field(td.H1(data), tack.array(c_volume.ravel()))

# A Fortran-ordered volume indexed [i, j, k].
f_volume = np.asfortranarray(np.fromfunction(lambda i, j, k: i + 10 * j + 100 * k,
                                             (nx, ny, nz), dtype=np.float32))
data.fields["f"] = td.Field(td.H1(data), tack.array(f_volume.ravel(order="F")))

expected = i + 10 * j + 100 * k
assert np.array_equal(data.fields["c"].values.numpy(), expected)
assert np.array_equal(data.fields["f"].values.numpy(), expected)
```

A field that depends on one index alone, as this one does on each, is a quick
check that a simulation's volume arrives in the right order.

## Build an unstructured grid

An unstructured grid lists its cells. Tack uses VTK's three arrays for them:

- `types` holds each cell's VTK cell type: 12 for a hexahedron, 14 for a
  pyramid, 10 for a tetrahedron, and so on.
- `connectivity` holds each cell's point ids, one cell after another.
- `offsets` says where each cell's ids start in `connectivity`. It has one more
  entry than there are cells, and the last entry is the length of
  `connectivity`.

This mesh is a hexahedron with a pyramid on its top face:

```python
--8<-- "docs/examples/datasets.py:unstructured"
```

The offsets `[0, 8, 13]` say that the hexahedron's eight ids come first and the
pyramid's five come next. The pyramid's base is the hexahedron's top face, so
the two cells share points 4 to 7. Each cell lists its corners in VTK's order
for its shape. The right points in the wrong order make a different cell, or
turn it inside out. [Cells and meshes](cells-and-meshes.md) lists the shapes
and their corner orders.

By default the number of points is one more than the largest id used. Pass
`num_points=` when the geometry has points no cell uses.

The three arrays can be lists, NumPy arrays or Tack arrays. Tack arrays of the
right types are kept as they are, without a copy: `u8` for the types, and
`i32` or `i64` for the offsets and connectivity. [Connect a simulation's
data](connect-simulation-data.md) relies on this. A mesh whose cells all have
one shape can give its types and offsets as implicit arrays, which store
nothing: `tack.arrays.ConstantArray(12, n, dtype=tack.u8)` for the types of
`n` hexahedra and `tack.arrays.CountingArray(n + 1, step=8)` for their offsets.

## Build polyhedra, polygons and polylines

Polyhedra, polygons and polylines are cells without a fixed shape, so their
topologies list more than corners. [Polyhedra, polygons and
polylines](polyhedra.md) explains how Tack evaluates fields inside them.

A polyhedral topology is made from each cell's faces, each face a ring of
point ids. Copies of a shared face are matched by their points and stored once.
With `orient=True` and the positions, the faces are first wound consistently
and outward:

```python
--8<-- "docs/examples/data_meshes.py:polyhedron"
```

A polygon is a ring of points, and a polyline is a chain of points. Both use
offsets and connectivity, as an unstructured grid does, without types:

```python
--8<-- "docs/examples/data_meshes.py:polygon"
```

```python
--8<-- "docs/examples/data_meshes.py:polyline"
```

`td.as_polyhedra(data)` converts a 3D dataset of shapes, such as hexahedra and
tetrahedra, to a polyhedral one and carries its fields. `td.as_polygons(data)`
does the same for triangles and quadrilaterals.

## Build a point cloud

A point cloud is a dataset of points, each its own vertex cell.
`td.point_cloud` takes the positions and a dictionary of fields, one value or
one row of components per point. Its cell arrays are implicit, so they cost no
memory:

```python
--8<-- "docs/examples/data3_build.py:point-cloud"
```

Floating-point fields take the dataset's precision, single by default.
Integer fields, such as a LiDAR classification, keep their type. Positions far
from zero, such as survey coordinates, are stored relative to an origin so that
single precision keeps their digits. `point_cloud` chooses that origin itself
for double-precision positions far from zero for their extent.
[Coordinates, units and time](units-coordinates-time.md) explains the origin,
and [Particles and point clouds](particles.md) what a point cloud is for.

## Start from a source

A source makes a dataset from a formula, for examples and tests. Tack has the
three that VTK and Viskores examples use most:

```python
--8<-- "docs/examples/data3_build.py:sources"
```

- `td.wavelet` is VTK's `vtkRTAnalyticSource`: an image over an extent of point
  indices, with the point field `RTData`.
- `td.tangle` is Viskores' tangle: a quartic on the unit cube whose level 0.5
  is a surface with several handles, in the point field `tangle`.
- `td.amr_gaussian_pulse` is VTK's `vtkAMRGaussianPulseSource`: an AMR
  hierarchy of two levels with a Gaussian in a cell field.

## Add fields from arrays

A field is a space and one value for each of the space's degrees of freedom.
Make the space for the dataset, wrap your array in a `td.Field`, and put it in
`data.fields` under a name:

```python
--8<-- "docs/examples/data3_build.py:from-arrays"
```

The space decides how many values the field needs. Allocate with
`space.size`, not with a count you work out yourself: an `H1` space of order 2
or an `L2` space needs more values than there are points. A field of the wrong
length is refused when you make it.

The values are any Tack array: scalars of any type, or vectors and matrices
(`tack.vec3()`, `tack.mat3()`). An implicit array works too. For example,
`tack.arrays.ConstantArray(5.0, data.num_points)` is a constant field stored as
one number. [Storage and layouts](storage-and-layouts.md) describes the
implicit arrays and when to store values instead.

`unit=` records the values' physical unit, such as `"m/s"`. It is meaning, not
storage: kernels never see it. A field without a unit has an unknown unit, and
filters keep it unknown rather than guess one.

Two fields may share a name if they are in different spaces, as VTK keeps a
point array and a cell array of one name apart. `data.fields["p", td.Constant]
= field` puts a cell field beside the point field `"p"`, and
`data.fields["p", td.Constant]` reads it back. The name alone is refused when
two fields share it, so code never reads the wrong one by accident.
[Fields and where values live](fields.md) explains names and keys in full.

### Copy each cell's own values

A discontinuous field (an `L2` field) gives each cell its own values, so a
corner shared by two cells has two values. This is how a discontinuous
Galerkin solver stores its solution. The space's `offsets` say where each
cell's values start. At order 1 a cell's values are its corners' values, in
the cell's corner order:

<!-- tack-check: build-l2 -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((3, 2, 2), spacing=(0.5, 1.0, 1.0))    # two hexahedra
space = td.L2(data, order=1)                                  # 8 values a cell
print(space.size, space.offsets.numpy())                      # 16 [ 0  8 16]

# p = x in cell 0 and x + 2 in cell 1: a jump of 2 across the shared face.
per_cell = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0],
                     [2.5, 3.0, 3.0, 2.5, 2.5, 3.0, 3.0, 2.5]], np.float32)
data.fields["pressure"] = td.Field(space, tack.array(per_cell.ravel()))
assert data.fields["pressure"].space.size == 2 * 8
```

The 16 values are more than the 12 points, because the four corners of the
shared face have one value from each cell. Higher orders, other bases and how
the values are laid out are in [Higher-order and finite-element
data](higher-order.md).

## Fill a field from a function

Often a field is a formula of position: an analytic solution to compare with,
an initial condition, or a test pattern. There are two ways to compute it on
the device without reading positions back to the host.

### Write a kernel over the geometry

For point data on a mesh with one position per point, a kernel can read
`data.geometry.values` directly. On an image those values are computed as the
kernel reads them, so nothing is downloaded or stored:

```python
--8<-- "docs/examples/data3_build.py:kernel"
```

```python
--8<-- "docs/examples/data3_build.py:from-kernel"
```

### Interpolate a device function into any space

`td.interpolate(data, function, space)` evaluates a `@tack.func` of a world
position at each of the space's nodes and returns the field. It works for
point data, cell data and higher-order spaces alike, so one function fills all
three here:

```python
--8<-- "docs/examples/data3_build.py:function"
```

```python
--8<-- "docs/examples/data3_build.py:interpolate"
```

The nodes are where the geometry puts them. For point data they are the
points, and for cell data each cell's center. For an `H1` field of order 2
they are the points and the middles of the edges, faces and cells. A node that
several cells share is evaluated once. `dtype=` gives the function's result
type: the dataset's float type by default, `tack.vec3()` for a vector. The
function receives world positions, so it needs no change for a dataset stored
relative to an origin.

## Select points, cells and faces with sets

A set is a named selection of a dataset's points, cells, faces or edges, such
as a boundary condition's faces or a material's cells. It plays the part of a
VTK selection, or of an Exodus node, element or side set. Make one with
`td.Set` and put it in `data.sets`:

```python
--8<-- "docs/examples/data3_build.py:sets"
```

- `td.Set(data, on, ids)` selects entities by id. `on` is `"points"`,
  `"cells"`, `"faces"` or `"edges"`. The ids are kept sorted and unique, and an
  id outside the mesh is refused.
- `td.Set.from_flags(data, on, flags)` selects the entities whose flag is not
  zero. The flags can be a Tack array a kernel wrote, a field, or a NumPy array.
- `td.Set.from_sides(data, cells, local_faces)` selects faces by (cell, local
  face) pairs, the form Exodus side sets use. Local faces are numbered in VTK's
  order for each shape.
- `td.boundary_faces(data)` is the faces with one cell.

Face and edge ids are the ones `data.topology.faces()` and
`data.topology.edges()` number. Sets travel with their dataset: filters carry
them to their outputs as they carry fields. A cell, face or edge set can also
be the domain a kernel iterates over (see [Write a mesh
algorithm](../visualization/write-a-mesh-algorithm.md)).

## Details

**Id types.** A topology stores its ids, offsets and connectivity as `i32`
when every id, and every array derived from them, fits. Otherwise it uses
`i64`. Pass `id_dtype=tack.i64` to a topology's constructor to ask for `i64`.
Tack arrays of `i64` ids also make the topology `i64`.

**Derived connectivity.** `data.topology.faces()` and `data.topology.edges()`
are built the first time something needs them, and then kept. A structured
grid derives them by arithmetic, and any other topology by sorting. The first
operation that needs faces pays for building them, which matters when you time
a pipeline's first call.

**Checking a mesh.** `topology.validate()` checks an unstructured topology's
ids. Offsets must rise from 0 to the connectivity's length, ids must lie within
the points, and each cell's point count must match its type. In debug mode
(`tack.set_debug(True)`), every topology checks its ids when it is made.

**Float type.** Coordinates and fields made from Python floats are single
precision (`tack.f32`) unless you pass `dtype=` or change the default with
`tack.set_default_float(tack.f64)`. Metal has no double precision.

## Where to go next

- [Connect a simulation's data](connect-simulation-data.md) builds the same
  objects on a simulation's own arrays, without copying them.
- [Data on devices](devices.md) puts a dataset on a GPU and moves it between
  devices.
- [Fields and where values live](fields.md) explains spaces, names and keys
  in full.

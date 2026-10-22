# Dataset API

`tack.dataset`, in the `tack-vis` package. This page describes the API as it
stands; the [Datasets and Visualization guide](../users-guide/09-visualization.md)
develops the model, mesh construction, field spaces, filters, mesh kernels and
complete pipelines with checked examples. For why it is shaped this way and how it got here, see the
design records: [Dataset API](../design/dataset-api.md) and
[Polyhedra](../design/polyhedra.md). They are history; this page is the
current state. What is missing is listed with each part, and under
[Settled, and open](#settled-and-open).

```python
import tack.dataset as td
from tack.dataset import algorithms as alg
```

## The model

- A **`DataSet`** is a topology, named fields and named sets.
- A **topology** is cells and the entities derived from them: faces, edges
  and points, and which cells lie on each side of a face.
- A **`Field`** is a space and its values.
- A **space** says where values live and how they are interpolated, if at
  all. It is built on a topology and owns its layout.
- The **geometry** is a field, so point positions are data like any other.
  It is kept apart from the named fields, as `data.geometry`.
- **Kernels** run over an iteration domain (cells, faces, edges or a face
  set) through `for_each`. That makes one launch per group of like entities,
  and each field argument becomes a view specialized to the group.

```python
data = td.rectilinear_grid(np.linspace(0, 1, 11), np.linspace(0, 1, 11), [0.0, 1.0])
x = data.positions()
data.fields["height"] = td.Field(td.H1(data), scalars(x[:, 2] + 0.25 * x[:, 0]))
gradient = alg.gradients(data, data.fields["height"])     # a field on the cells
```

Here `scalars` stands for any function that puts a host array in a
`tack.Array`.

## Import path migration

The package is `tack.dataset` (previously `tack.data`). Change
`import tack.data as td` to `import tack.dataset as td`, and use
`tack.dataset.<submodule>` for direct imports. Existing `td.DataSet`,
`td.Field`, spaces, topologies and algorithm calls keep their names and signatures.
The shorthand `td` means Tack datasets.

## Topologies

There are two kinds. **Shape-based** topologies have cells of fixed shapes,
each with a reference element: counts, tables and shape functions are
compile-time constants. **Polyhedral** topologies have cells that are lists
of faces of any size, with no reference element. They are separate paths
that share everything else: spaces without a basis, fields, sets,
`for_each`, the face conventions, and a common set of methods on cell views.
`topology.reference_cells` says which kind a topology is.

| Topology | Built from | Kind |
|---|---|---|
| `UnstructuredTopology(types, offsets, connectivity, num_points=None, id_dtype=None)` | VTK cell types and point rows, of the linear shapes (vertex to pyramid) | shape-based |
| `StructuredTopology(point_dims, id_dtype=None)`, usually through `rectilinear_grid(x, y, z)` | grid dimensions; cells addressed by (i, j, k) | shape-based |
| `PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, cell_face_sides, num_points=None, id_dtype=None)` | faces stored once, as point rings wound out of side 0; each cell a list of faces, with the side (u8, 0 or 1) it is on | polyhedral, 3D |
| `PolyhedralTopology.from_cell_faces(cells, num_points, positions=None, orient=False, id_dtype=None)` | each cell's own face rings, as VTK and most readers give them; copies of a face matched by point set | polyhedral, 3D |
| `PolygonalTopology(loop_offsets, loop_points, num_points=None, id_dtype=None)` | each polygon's loop of points | polyhedral, 2D |
| `PolylineTopology(line_offsets, line_points, num_points=None, id_dtype=None)` | each polyline's chain of points, two or more; a chain may close | polyhedral, 1D |

`num_points` exists because a dataset may have points that no cell uses.

**Implicit topology arrays.** An `UnstructuredTopology`'s `types`, `offsets`
and `connectivity` may each be an implicit array from `tack.arrays`, kept as
it is and computed where it is read, as Viskores' single-type cell sets keep
theirs:

- the types of cells all of one shape, a `ConstantArray` (u8);
- the offsets of cells all of one size, a `CountingArray` with that size as
  its step;
- the connectivity of cells that each have their own points (vertex cells,
  a 2D contour's lines), a `CountingArray` too.

`uniform_cells(shape_id, count, id_dtype)` in `tack.dataset.topology` makes
the first two. Filters whose output cells are all one shape give them:
`contour` and `slice` (triangles, or lines in 2D), `tetrahedralize`,
`triangulate`, `extract_edges`, the point extractions and `probe`'s vertex
cells (all three implicit), and `threshold`, `extract_cells`,
`extract_geometry` and `mask` of a topology of one shape. A topology of one
shape groups its cells without a histogram of their types. Everything else
reads them as it reads stored ones; `numpy()` computes them on the host.

`dataset_to_vtk` hands them over as VTK's own: a `ConstantArray` of types
becomes the cell type given as a number (VTK keeps it as a
`vtkConstantArray`), a `CountingArray` a `vtkAffineArray`.
`vtk_to_dataset` takes VTK's constant and affine arrays back as
`ConstantArray` and `CountingArray`.

### Ids

Every id a topology stores or derives has one integer type,
`topology.id_dtype` (also `data.id_dtype`): point and cell ids, offsets,
faces, edges, point links, and the orders and keys that sort them. It is
`i32` unless the topology:

- is asked for `i64` (`id_dtype=tack.i64`);
- is built from `i64` fields;
- or is too large for `i32`: an id, an offset or a derived array
  past 2^31 − 1 entries (a mesh's edges number up to twelve per cell).

Asked for `i32`, a topology too large for it raises, so a valid mesh's ids
(each below `num_points`) are never wrapped.
Arrays from NumPy or VTK (whose ids are 64-bit) take the narrowest type that
holds them. Kernels specialize on their fields' types, so the same kernels
serve both, and an `i32` mesh pays nothing for the `i64` path.

A filter's output keeps its input's type, or takes `i64` when the output
can be too large for `i32` (a hexahedron tetrahedralizes into 20
connectivity entries), so a pipeline that starts in `i64` stays there.
Sort keys are tuples of ids, never ids packed into one integer, so 64-bit
ids sort as 32-bit ones do.

On GPUs without 64-bit atomics (Metal, Level Zero), `CellLocator` refuses a
dataset of more than 2^30 − 1 cells (it ranks a point's candidate cells up to
twice the cell count); everything else works there.

`pytest --ids=i64` runs the dataset tests with every topology in `i64`.

### Derived entities

- `topology.faces()` and `topology.edges()` are derived on the device by
  sorting, the first time they are asked for, and then kept.
- Faces are **codimension 1** of the cells of the highest dimension
  (`topology.dimension`): a 3D mesh's faces are polygons, a 2D mesh's are its
  edges, on the shape path as on the polygonal one. A 2D cell's faces go round
  it (VTK counts none). A mesh mixing dimensions takes its faces from its
  top-dimensional cells alone.
- On a 2D mesh `face_geometry` gives each edge's normal in the plane of its
  side 0's cell, pointing out of it, and its length as its "area";
  `face_centers` its midpoint. `net_flux`, `divergence`, `perot`, `jump` and
  `upwind_flux` then work on 2D meshes as on 3D ones.
- A mesh of line cells (a 1D grid's, or VTK lines) has no faces yet: a
  line's would be its end points, which are not derived, so `faces()` and
  everything built on it (`face_geometry`, `extract_surface`, side traces,
  `jump`, `upwind_flux`, ...) raise `NotImplementedError`. A polyline
  topology's have them.
- A cell of no volume (a hexahedron whose points coincide, a grid of
  spacing 0), or a face or polygon of no area, has measure 0, its centroid at
  the mean of its points, and normals of zero length, so `upwind_flux`
  through it is 0. What divides by the measure has no value there: its
  `gradients`, `flow_quantities` and `perot` are NaN.
- A polyline topology's faces are the chains' ends, one per point that
  chains end at: side 0 is the first chain ending there (in cell order),
  side 1 the second, and the normal runs along side 0's chain, out of it, with
  an "area" of 1. A vertex has no winding, so any two chains meeting end to
  end are consistent. Three or more ending at one point (a branch) have no
  faces: `faces()` refuses them, as an edge of three polygons is refused,
  while everything that does not need faces still works.
- A face's points are wound out of its **side 0**. **Side 1** is the cell on
  the other side, or none (`-1`) on the boundary.
- Each cell knows, per local face, the face's id and which side of it the
  cell is on.
- On a polyhedral topology the faces are given, not derived, and keep their
  given numbering.
- A polyhedral topology also derives each cell's distinct points
  (`cell_points()`), for point data.

### Converting and checking

| Function | What it does |
|---|---|
| `as_polyhedra(data)` | a 3D shape-based dataset as polyhedra, carrying its fields |
| `as_polygons(data)` | a 2D shape-based dataset as polygons |
| `check_winding(data)` | of polyhedra or polygons, a cell `Set` of the cells whose faces do not wind consistently; empty when all is well (a shape's faces come from its table) |
| `orient(topology)` | turns around faces wound into their only cell |
| `from_cell_faces(..., orient=True)` | repairs input whose faces or cells are inconsistently wound, then turns each connected piece outward by the sign of its total volume |

Winding comes from the input, not from geometry. Inconsistent input is
refused unless the caller asks for the repair.

## Spaces

| Space | Values | Interpolated | Topologies |
|---|---|---|---|
| `H1(data, order=1)` | one per point, shared by the cells around it; order 2 adds one per edge and quad face | yes, continuous | order 1 on all; order 2 on tetrahedra, hexahedra, voxels, wedges, triangles, quads and pixels |
| `L2(data, order=1)` | each cell's own copies: order 1 one per corner, order 0 one per cell; `order` may be an array, one order per cell | yes, discontinuous | shape-based |
| `Constant(data)` | one per cell | yes, constant in the cell | all |
| `Values(data, on, oriented=False)` | one per point, edge, face or cell: data about the entity, with no basis | no | all |
| `SideTraces(data)` | per face, per side, per face point; made by `project` into it | through the face's shape | shape-based, 3D |

- **Identity.** A space is interned on its topology: `H1(data)` is the same
  object every time, so fields on it share its layout. Two spaces with
  different parameters are different objects.
- **Size.** Every space knows its `size`. A field of the wrong length, or a
  field from another topology, is refused.
- **Continuity** follows from where the values live (MFEM's model): values
  on shared entities are continuous, and values owned per cell are not.
- **Oriented face values.** `oriented=True` is for a quantity measured along
  the face's normal, such as a normal flux. Anything that turns a face
  around (threshold, `orient`) negates such values. Plain face values (an
  area, an id) are left alone.

Attributes: `topology`, `size`, `on` (the entity kind the values are
indexed by), `interpolated`, and `order` where it applies.

### Moving a field between spaces

`project(data, field, space)` gives `field` in another space of the same
topology, as a new field. Values on points or cells count as order-1 `H1`
and `Constant`, as source and as target.

| Into | Each value is | Exact when |
|---|---|---|
| `H1` (order 1 or 2) | the mean, over the cells around the node, of the source evaluated there | the source is continuous: a lower order comes across exactly |
| `Constant`, order-0 `L2` | the mean of the cell's corner values (vtkPointDataToCellData) | the source is constant in the cell |
| order-1 `L2` (or an order per cell) | the source's value at each corner (the corner mean at order 0) | the source's order is at most 1 |
| `SideTraces` | each side's value from its cell | always |

Where cells disagree -- cell data, DG data -- the `H1` mean is VTK's
averaging (vtkCellDataToPointData). Sums run in a fixed order. Scalars or
vectors; floating point. Values on faces or edges have no basis and are
refused. On polyhedra, polygons and polylines only cell and point values
project, each into the other.

Unused points retain their values when the source is H1 (or values on points).
Cell or DG sources give unused points zero, since no cell contributes there.

`project` is the one way to move a field: into `H1(data)` for point values,
`Constant(data)` for one value per cell, `L2(data)` for each cell's own copies,
`SideTraces(data)` for each side's value at the faces.

### A field from a function of position

`interpolate(data, function, space, dtype=None)` is the field in `space`
whose values are `function` at the space's nodes. `function` is a
`@tack.func` of a world position (a 3-vector of the geometry's type) that
returns a scalar or a vector; `dtype` is what it returns, `data.dtype` by
default (`dtype=tack.vec3()` for 3-vectors).

```python
@tack.func
def temperature(x):
    return x.x * x.x + 2.0 * x.y

data.fields["t"] = td.interpolate(data, temperature, td.H1(data, order=2))
```

| Space | Its nodes |
|---|---|
| `H1` | the points; at order 2 also edge middles, quad face centers and cell centers |
| order-1 `L2` | each cell's corners |
| `Constant`, order-0 `L2`, `Values(data, "cells")` | each cell's parametric center |
| `Values(data, "points")` | the points |

- Node positions come from the geometry as they are needed (a curved
  geometry's included) and are never stored.
- A node shared by several cells is evaluated once, by the first cell that
  has it (cells in group order). Which cell writes each node is kept on the
  space, so a changed geometry is seen. A discontinuous (`L2`) geometry
  gives a shared node the first cell's position.
- A point no cell uses takes `function` at its own position when the
  geometry is `H1`, and zero when it is `L2`.
- A polynomial of the space's order on straight cells is represented
  exactly; anything else gets the space's nodal interpolant.
- On polyhedra, polygons and polylines only point values (`H1` of order 1,
  `Values(data, "points")`) interpolate, from an `H1` geometry.

## Fields and arrays

`Field(space, values)`: the values are a `tack.Array` with a scalar or vector
element dtype, or an implicit array from `tack.arrays` (in tack-core). An
array's shape counts coefficient values; its dtype controls components. The
space supplies their mesh association and basis. The implicit arrays, all
in `tack.arrays`, are indexable in any kernel as a stored array is (`a[k]`,
`len(a)`), not only through a field:

| Array | Value `k` | VTK | Viskores |
|---|---|---|---|
| `ConstantArray(value, size, dtype=None, device=None)` | `value`, a scalar or a vector | `vtkConstantArray` | `ArrayHandleConstant` |
| `CountingArray(size, start=0, step=1, dtype=None)` | `start + step * k` | `vtkAffineArray` | `ArrayHandleCounting` |
| `CartesianProduct(x, y=(0.0,), z=(0.0,), device=None)` | a rectilinear grid's point, from three axes | `vtkStructuredPointArray` | `ArrayHandleCartesianProduct` |
| `UniformCoordinates(dims, origin, spacing, direction=None)` | an image's point, `origin + direction @ (ijk * spacing)` | `vtkStructuredPointArray` | `ArrayHandleUniformPointCoordinates` |
| `Permutation(ids, base)` | `base[ids[k]]` | `vtkIndexedArray` | `ArrayHandlePermutation` |
| `Concatenate(a, b, ...)` | the arrays end to end | `vtkCompositeArray` | `ArrayHandleConcatenate` |
| `Strided(values, stride, offset=0, size=None)` | `values[offset + stride * k]` of a scalar field | `vtkStridedArray` | `ArrayHandleStride` |
| `View(base, start, size)` | `base[start + k]` | -- | `ArrayHandleView` |
| `ExtractComponent(base, c)` | component `c` of `base[k]` | -- | `ArrayHandleExtractComponent` |
| `Components(a, b, ...)` | two to four scalar arrays as a vector | -- | `ArrayHandleCompositeVector` |
| `Function(func, size, dtype=tack.f32, width=None)` | `func(k)`, a `@tack.func` | `vtkStdFunctionArray` | `ArrayHandleImplicit` |
| `Transform(func, base, dtype=None, width=None)` | `func(base[k])` | -- | `ArrayHandleTransform` |
| `Cast(base, dtype)` | `base[k]` converted | -- | `ArrayHandleCast` |
| `RandomUniform(size, seed, low, high)`, `RandomNormal(size, seed, mean, stddev)` | draws from `tack.random`, a pure function of `k` and `seed` | -- | `ArrayHandleRandomUniformReal`, `...StandardNormal` |

An array's operands are any arrays -- implicit, or device fields -- so they
compose to any depth: `Permutation(ids, Concatenate(field, CountingArray(...)))`
is an array. Every algorithm takes one unchanged: a field's view holds it as
a nested template (`tack.device_class` objects may hold others) and reads
`array.get(k)`, which inlines to the expression one would write by hand.
A composed array therefore costs nothing per value beyond what it computes;
each level of nesting costs about 8 us per kernel launch on the host. VTK's
`vtkStdFunctionArray` runs a C++ function on the CPU; a `Function` or
`Transform` is a `@tack.func`, so it runs in the kernel on every backend.
`materialize(array)` stores one, by a kernel; `to_host` reads one back. A
NumPy array is refused as values: copy it with `tack.array(host, dtype=...)` first.

`uniform_grid(dims, origin, spacing, direction=None)` is a dataset on
`UniformCoordinates`, as `rectilinear_grid(x, y, z)` is on a
`CartesianProduct`; `sources.wavelet` and `sources.tangle` are uniform
grids. The VTK interop reads and writes `vtkImageData` as uniform grids,
direction matrix included. A size of 1 adds no cells, in any position:
`(1, 3)` is a line of three points along y, as `(3, 1)` is along x. A grid
with no axis of two or more points has no cells: `(1, 1, 1)` is one point,
and a size of 0 makes it empty. Their topology is then an
`UnstructuredTopology` without cells, and the VTK interop still writes them
as `vtkImageData` (or `vtkRectilinearGrid`).

**Geometry.** `DataSet(topology, geometry, fields=None, sets=None,
dtype=tack.f32)` takes the geometry as an `H1` or `L2` field of 3-vectors,
or as positions per point (any array of 3-vectors), which become an `H1`
field. It is kept apart from `fields`: `data.geometry` returns it, assigning
`data.geometry` replaces it (checked as the constructor checks it), and
`data.positions()` gives the positions on the host. Algorithms move it with
the cells; it is never among the fields they carry or select. An `L2` geometry pulls
cells apart without changing the topology. A geometry of order 2 is curved.

## Devices

A dataset is on the device its arrays are on, and it computes there: every
algorithm, every derived entity (faces, edges, point links, groups, layouts),
`for_each`, `Set`, `CellLocator` and the sources run and allocate on the
dataset's device, whatever the default device is. Results stay on it.

- `data.device`, `data.topology.device`, `field.device`, `a_set.device`: the
  device, a `tack.Device`. A structured topology holds no arrays, so it is
  on the device it was made for (`device=`, by default the default device
  when it was made).
- A dataset's parts must share one device: a `Field` whose values are not on
  its topology's device is refused, and so is a dataset whose topology,
  geometry, fields or sets are on two (`ValueError`, "arrays on two
  devices"), when it is made and when an algorithm is given it.
- `data.to_device(device)` copies it there: the topology from its arrays,
  each field in the same space, the sets by the same ids. What the topology
  derives is derived there afresh.
- `uniform_grid`, `rectilinear_grid`, `wavelet` and `tangle` take `device=`;
  a rectilinear grid of device-array axes is on their device.
- An implicit array is on the device of the arrays it holds:
  `ConstantArray`, `UniformCoordinates` and `CartesianProduct` take
  `device=` for the small arrays they store, `to_device` moves one, and one
  made from arrays on two devices is refused. One that holds no array
  (`CountingArray`, `Function`) belongs to no device.

```python
with tack.device("cuda"):
    data = td.wavelet(20)                     # every array on the GPU
surface = td.contour(data, "RTData", 150.0)   # computed there, whatever the default
on_cpu = surface.to_device("cpu")
```

## Sets

A `Set(data, on, ids)` selects some of a topology's points, cells, faces or
edges (`on` is `"points"`, `"cells"`, `"faces"` or `"edges"`; edge ids are
those of `topology.edges()`). Its `ids` are kept sorted and
unique, in the topology's `id_dtype`; an id outside the topology is refused
when the set is made. `flags()` gives membership, 1 or 0 per entity.

- **Naming.** `data.sets` maps names to sets of the dataset's own topology.
  Anything else is refused when it is put in: an array (wrap it in a `Set`),
  a set of another topology, or a name that is a domain's (`"cells"`).
- **Builders return the set.** `boundary_faces(data)` is a face set; name
  it to keep it: `data.sets["boundary"] = boundary_faces(data)`.
- **Side sets.** A face set is a side set: a boundary face has one side.
  `Set.from_sides(data, cells, local_faces)` takes (cell, local face) pairs,
  as Exodus side sets give them; local faces are numbered as the cell view
  numbers them. Which side was named is not kept.
- **Iteration.** A cell, face or edge set's name is an iteration domain:
  `for_each(kernel, data, "inlet", ...)`. A cell set runs over the cell
  groups' subgroups of its members, so field views and varying spaces work
  as on all cells. A point set is not a domain (points have no view): loop
  over its `ids`.
- **Carried through filters**, by the maps that carry fields: an output
  entity is in a set when the input entity it came from is. Kept cells keep
  their sets (`threshold`, `extract_cells`), pieces of cells their cells'
  (`contour`, `clip`), and a face or edge set becomes a cell set where faces
  or edges become cells (`extract_surface`, `extract_edges`). A set of
  entities the filter
  makes anew -- interpolated points, faces derived afresh -- is left
  behind. Filters that keep the topology keep every set.

**Open:** boundary entities from readers mapped onto sets (Exodus side and
node sets, MFEM boundary attributes).

## Partitioned datasets

A `PartitionedDataSet(partitions)` is a dataset in pieces: an ordered list
of `DataSet`s, as VTK's `vtkPartitionedDataSet`. Each piece is a whole
dataset, with its own topology, numbering, geometry, fields, sets and
device, and may have no cells. Iterate over it, index it (`pds[i]`) or take
its length; `pds.partitions` is the tuple of pieces.

- **Work on one piece takes the piece.** Filters, analyses, selections and
  `for_each` take a `DataSet`, as everywhere else; loop over the pieces:
  `for piece in pds: piece.fields["grad"] = td.gradients(piece, "T")`.
- **Only algorithms whose answer needs more than one piece take the
  partitioned dataset**, and they name the field (a `Field` belongs to one
  piece's topology), which every piece must have:
  - `statistics(pds, "T")` takes each piece's moments on its own device and
    combines them pairwise (Pébay's update), so the result is the whole's
    up to rounding;
  - `entropy(pds, "T", bins)` bins the whole's range and sums each piece's
    histogram over it.

  Pieces with no values are skipped. Each piece counts only what it owns
  (below), so where every shared cell and point has one owner, the result
  is the whole's.
- **`split(data, pieces)`** cuts a structured grid by index ranges, x
  fastest. `pieces` is a count, factored over the axes (each prime factor,
  largest first, cuts the axis whose pieces have most cells), or a tuple of
  counts, one per axis of the topology. Each axis's cells are shared out as
  evenly as they go, at least one per piece, and neighbouring pieces share
  the points between them. A piece keeps the geometry's kind (uniform
  coordinates with their origin moved, rectilinear axes cut, explicit
  points taken) and carries its block of every point, cell and `L2` field
  and every set, as filters carry them; fields on faces or edges are left
  behind. A point a piece shares with a piece before it is that piece's, and
  marked a duplicate in this one; the input's own ghost marks come along.
  An unstructured mesh is partitioned by VTK
  (`vtkRedistributeDataSetFilter`) or by its host application.

**Ghost marks.** Every cell and point that several pieces hold has exactly one
owner; every other copy is a ghost. `data.ghosts` is a
`tack.dataset.Ghosts` (or `None`), kept apart from the fields as the
geometry is: a `u8` per cell (`cells`) and per point (`points`), either of
which may be `None`, holding VTK's `vtkGhostType` bits (`Ghosts.DUPLICATECELL`
1, `HIDDENCELL` 32, `DUPLICATEPOINT` 1, `HIDDENPOINT` 2, and the rest, carried
unread). A duplicate is a copy of what another piece owns; hidden is data not
valid there (an AMR cell a finer level covers, a blanked cell).

- `vtk_to_dataset` takes a `vtkGhostType` array as the marks, not as a field,
  and `dataset_to_vtk` writes them back; `split` makes them;
  `data.to_device` moves them. Marks of the wrong length or type are refused
  when they are put in.
- `statistics` and `entropy` leave out duplicate and hidden cells and points,
  on one dataset too, so each value counts once across all pieces. An `L2`
  field goes by its cells' marks; a field on faces or edges, or of order 2,
  is refused on a dataset with marks until it is decided who owns those.
- **Not yet:** filters do not carry the marks to their outputs. The plan:
  filters process duplicates and mark what they make from them, as VTK does,
  with one owner for every shared point, edge and face (the owner of an
  edge or face being the owner of its smallest point, by global id or
  position); they skip hidden cells; `remove_ghosts` removes ghosts before
  rendering or writing one file.

```python
pieces = td.split(td.wavelet(20), 8)                    # 8 image-data pieces
for piece in pieces:
    piece.fields["RTData_cells"] = td.project(piece, "RTData", td.Constant(piece))
s = td.statistics(pieces, "RTData_cells")               # the whole grid's
```

`packages/tack-vis/examples/44_partitioned_data.py` runs all of it: pieces,
their ghost marks, work in a loop, reductions checked against the whole,
pieces spread over several GPUs where there are, and a `.vtpd` file for
ParaView.

**Next** (the partitioned-data proposal's later phases): filters that carry
ghost marks; global ids; ghost cells generated in one process; then MPI.

## Kernels

```python
@tack.kernel
def _jump(faces, values, out):
    for f in faces:
        c0 = faces.side_cell(f, 0)
        c1 = faces.side_cell(f, 1)                 # -1 on the boundary
        out[faces.entity_id(f)] = values[c1] - values[c0] if c1 >= 0 else 0.0

td.for_each(_jump, data, "faces", cell_values, out)
```

`for_each(kernel, data, domain, *args)`:

- `domain` is `"cells"`, `"faces"`, `"edges"` or the name of a cell, face or
  edge set.
- It makes one launch per group. The kernel's first argument is the view of
  that group (`for i in view`). Each `Field` among `args` becomes its view
  for the same group, so `u.value(i, pc)` reads entity `i`. Other arguments
  are passed through unchanged.

**What makes a group.** On a shape-based topology, each shape is a group. A
group is split further, on the device, by:

- the spaces among the arguments that vary from cell to cell (an `L2` with
  an order per cell), so each launch is specialized to one order;
- any `keys`, such as `SizeBuckets(topology, caps=(16, 32, 64, 128, 256))`
  on a polyhedral topology. That gives each launch a class constant
  `MAX_SCRATCH`, so that `tack.local_array(dtype, cells.MAX_SCRATCH)` has a
  size known at compile time.

Shape, order and size all go through this one mechanism. An algorithm that
needs the groups itself uses `data.launch_groups(domain, fields=(),
keys=())`, `data.domain_view(domain, group)` and `field.view(group)`.

**Fields are separate arguments.** Views are not nested in a dataset
object: each field is its own argument, viewed for the group being
launched. A view is put together from mixins (entity kind, shape, geometry,
incidence, space, storage), which share one namespace. A view refuses to
build if two of them define the same name.

### What views offer

Every view supports `for i in view` and `entity_id(i)`, the global id of
the entity the loop is at. Names follow one rule per kind of answer:

- a global id ends in `_id`: `point_id(c, j)`, `face_id(c, k)`,
  `side_point_id(c, k, j)`;
- a position ends in `_position`: `corner_position(i, j)`,
  `side_position(c, k, j)`, `point_position(p)` (by global id), and
  `position(c, pc)` at parametric coordinates;
- a shape's local corner number ends in `_corner`: `edge_corner(e, k)`,
  `face_corner(f, k)`;
- a count starts with `num_`: `num_points`, `num_faces`, `side_num_points`.

**Cells.** These methods work on every kind of topology, so face-based
algorithms are written once:

| Method | |
|---|---|
| `num_faces(c)` | the cell's faces (codimension-1 entities) |
| `face_id(c, k)`, `face_side(c, k)` | local face `k`'s global id, and the side the cell is on |
| `side_num_points(c, k)`, `side_point_id(c, k, j)` | local face `k`'s points, walked outward from this cell |
| `side_position(c, k, j)` | that point's position |

- Shape-based cells add: the class constants `NUM_POINTS`, `NUM_FACES`,
  `NUM_EDGES` and `DIMENSION`; `point_id(c, j)`; the geometry through
  `corner_position(c, j)`, `position(c, pc)` and `geometry_jacobian(c, pc)`;
  the shape's tables (`parametric_point(j)`, `edge_corner(e, k)`,
  `face_corner(f, k)`, `face_num_points(f)`, `face_shape(f)`, shape
  functions); face incidence (`face_orientation`, `face_position`); and edges
  (`edge_id`, `edge_sign`). A face's corners go round it, a voxel's too.
- Polyhedral, polygonal and polyline cells add `num_points(c)` and
  `point_id(c, j)` over the cell's distinct points, and `point_position(p)`.
  Polyline cells add their chain in order: `line_num_points(c)`,
  `line_point_id(c, j)`.

**Faces.** `num_points(f)`, `point_id(f, j)`, `corner_position(f, j)`,
`num_sides(f)`, and `side_cell(f, s)` / `side_local(f, s)` give the cell on
side `s` and its local face number.

**Edges.** `point_id(e, j)`, with the lower point id first, and
`corner_position(e, j)`.

**Fields.**

| Field view | Methods |
|---|---|
| `H1`, `L2`, `Constant` | `value(c, pc)`, `parametric_gradient(c, pc)` at parametric coordinates; `dof(c, k)` |
| `Values` | `at(i)` |
| `SideTraces` | `trace(f, s, j)`, `value(f, s, pc)` |

## Algorithms and filters

Every algorithm is in `tack.dataset.algorithms`, one module each, and
`tack.dataset` re-exports them all. They share one set of conventions:

- the dataset comes first: `f(data, ...)`;
- a field argument is a `Field` or the name of one of `data`'s fields;
- an algorithm that makes a new dataset takes `fields=` (all by default, a
  name, a list of names, or `[]` for none) and returns a `DataSet`;
- an algorithm that computes a field returns a `Field`, and several fields a
  named tuple of them (`cell_geometry`, `face_geometry`, `flow_quantities`);
- a summary returns a Python number (`entropy`), or a named tuple of them
  (`statistics`);
- a selection returns a `Set` (`boundary_faces`, `check_winding`);
- one name per operation, a variant a parameter (`extract_surface` defaults to
  the boundary; `project` moves a field into any space), and one parameter
  name per meaning (`inside=` keeps a region, `value` is a level, `lower` and
  `upper` a range);
- filters that cut a mesh into a subset of its cells compact the points;
  `extract_surface` and `extract_edges` keep the input's points, so point ids
  stay valid. The
  point filters (`mask_points`, `threshold_points`, `extract_points`) are
  the point twins of `mask`, `threshold` and `extract_geometry`: the points as
  vertex cells, point fields carried and cell fields left behind;
- nothing changes its input: a new dataset is new, and sets are kept by
  storing what `boundary_faces` returns.

Field arguments are checked against the input topology, including on empty
datasets. A raw device array needs a `Field(space, values)` wrapper. A
`locator=` must have been built for that same dataset; rebuild it after changing
the geometry, as described under locating points below.

**Numerical types.** `composite`, `dot` and `cross` require matching input
dtypes and retain that dtype. Use `Cast(values, dtype)` to convert explicitly.
Algorithms that need fractional arithmetic (`clip` weights, `statistics`,
`magnitude` and `log_values`) compute in f64 for f64 input, otherwise f32,
converting integer values before arithmetic. `statistics` sums integer
values exactly in f64 (to 2^53, without wrapping) and takes its moments in
that floating precision; its extrema come from the original values. Converting wide
integers may round values beyond the exact integer range of f32 (2^24) or
f64 (2^53); use f64 input for greater precision on a backend that supports it.
Integer `magnitude` and `log_values` therefore return floating point fields.

Empty supported filters return valid empty topologies and carry selected fields
and sets by the same maps as nonempty filters; missing `fields=` names still
raise `KeyError`. Empty polyline contouring retains the documented refusal.

In the table,
"polyhedral" means `PolyhedralTopology`, "polygonal" `PolygonalTopology`
and "polyline" `PolylineTopology`.

| | Shape-based | Polyhedral | Polygonal | Polyline |
|---|---|---|---|---|
| `face_geometry` (normals, areas), `face_centers`, `cell_geometry` (volume, area or length; centroid) | yes | yes | yes | yes |
| `edge_lengths`, `extract_edges` | yes | yes | yes | no (no derived edges) |
| `boundary_faces`, `extract_surface` | yes (a 2D mesh's: line cells) | yes (the surface is polygonal) | yes (line cells) | yes (vertex cells) |
| `jump` and `upwind_flux` of cell data; `net_flux`, `divergence`, `perot` of face values | yes | yes | yes | yes |
| `threshold`, `extract_geometry`, `extract_cells`, `mask` | yes | yes | yes | yes |
| `contour`, `slice`, `slice_plane` | yes (a 2D mesh's: line cells; an `L2` field or geometry gives each cell its own points) | yes (López, face-based) | yes (line cells) | no (would give points) |
| `clip` | yes (with an `H1` geometry) | not yet | not yet | not yet |
| `extract_points`, `threshold_points`, `mask_points` | yes | yes | yes | yes |
| `implicit_values` | yes | yes | yes | yes |
| `CellLocator`, `probe` | yes | yes (in sub-tetrahedra; cells star-shaped from their cell point) | not yet | not yet |
| `advect`, `streamlines` | yes (2D and 3D cells) | yes (in sub-tetrahedra; cells star-shaped from their cell point) | no | no |
| `cell_centers`, `values_at_centers` | yes | yes (at the cell point, the mean of its points; point and cell data) | yes (likewise) | yes (likewise) |
| `gradients(at="cells" | "points")` of scalars or vectors, `flow_quantities` | yes | yes (Green–Gauss over each cell's faces, fanned from their points' means: exact for linear fields) | yes (zero, as for 2D cells on shapes) | yes (zero) |
| `project` between cell and point values | yes | yes (over each cell's distinct points) | yes | yes |
| `project` into or from `L2`, order-2 `H1`, `SideTraces` | yes | no | no | no |
| side traces, and `jump`/`upwind_flux` of point data | yes | no | no | no |

Where the table says "no", the algorithm needs a reference element and
raises `NotImplementedError` saying so.

**How fields carry through filters.** Every filter that makes a new dataset
(contour, slice, threshold, external faces and `extract_surface`,
`as_polyhedra`, `as_polygons`) carries fields the same way, through
`tack.dataset.carry`. The filter states how its output's entities come from the
input's -- the same entity (`Same`), input entity `ids[i]` (`Take`), a piece
of input cell `ids[i]` (`Pieces`), or a point between two input points
(`Interpolate`) -- and one set of rules applies to every field:

| Input field | On the output |
|---|---|
| point data (`H1`, values on points) | through the point map; an order-2 field brings its corner values, except on a solid's surface (`extract_surface`), whose nodes are the solid's, gathered at order 2; interpolated only if floating point |
| cell data (`Constant`, values on cells) | through the cell map, keeping its kind |
| `L2` (DG) data | copied cell by cell, each cell keeping its order, when whole cells are kept (not pieces) and the output has reference cells |
| values on faces | through the face map, oriented values negated where a face is turned; or onto the output's cells when they are input faces (a surface) |
| values on edges | through the edge map, given only where the edges are the input's (`as_polyhedra`) |
| traces, anything without a map | left behind |

What each filter gives:

| Filter | points | cells | faces | edges |
|---|---|---|---|---|
| `threshold`, `extract_geometry`, `extract_cells`, `mask` | kept points | kept cells | kept faces, turned where needed (polyhedra) | — |
| `contour`, `slice`, `slice_plane` | interpolated along edges | pieces of the cut cells | — | — |
| `clip` | kept points, edge crossings and centroids | pieces of the cut cells, whole kept ones | — | — |
| `extract_points`, `threshold_points`, `mask_points` | kept points, each a vertex cell | — | — | — |
| `extract_surface` | the same points | pieces (each face's cell); face values become cell values | — | — |
| `extract_edges` | the same points | — (an edge has several cells) | — | the lines' cells: edge values become cell values |
| `as_polyhedra` | same | same | same | same |
| `as_polygons` | same | same | — | — |

Every one of them takes `fields=`: all fields by default, a name or a list
of names, or `[]` for none; a name the input lacks raises `KeyError`. A new
filter states its maps and calls `carry(data, out, points=..., cells=...,
fields=fields)`.

**Gradients.** `algorithms.gradients(data, field, at="cells")` gives each
cell's gradient at its parametric center; `at="points"` evaluates each cell's
gradient at the point and averages over the cells around it, as VTK's
vtkGradientFilter does (a pyramid's apex extrapolated from just below it, as
vtkPyramid does). A vector field's gradient is `3k` values per entity, row by
row (`d(u_i)/d(x_j)` at `3i + j`). `flow_quantities(data, gradient)`
gives divergence, vorticity and the Q-criterion from a 3-vector field's, as a
`FlowQuantities` named tuple. Cells below three dimensions get zero, and a
cell of no volume NaN.

## Implicit functions

`tack.dataset.Plane(origin, normal)`, `Sphere(center, radius)`,
`Cylinder(center, axis, radius)` (infinite), `Box(lower, upper)`
(axis-aligned) and `Planes(origins, normals, dtype=tack.f32)` (the convex region
below every plane: a frustum, a box at any angle; `dtype=tack.f64` for f64 data). `value(p)` is negative inside, zero on
the surface, positive outside, in VTK's and Viskores' forms -- quadrics for
the sphere and cylinder, signed distances for the plane, planes and box -- so
a slice by one lands on the same points as theirs.

Each is a `@tack.device_class` object whose parameters are instance values,
so moving or resizing one compiles nothing anew; a kernel takes it as a
template and calls `f.value(p)`. Taking one:

| | |
|---|---|
| `algorithms.implicit_values(data, f)` | `f` at the geometry's values, a field in the geometry's space |
| `slice(data, f)` | where `f` is zero: a contour of those values |
| `extract_geometry(data, f, inside=True, boundary=False)` | whole cells inside (every point at or below zero) or outside, plus with `boundary` those `f` cuts |
| `extract_points(data, f, inside=True)` | the points inside or outside, as vertex cells |

`clip(data, by, value=0.0, inside=False)` keeps the part of every cell where
`by` -- point data or an implicit function -- is at or above `value` (below,
with `inside`: inside an implicit function's region), cutting cells by Viskores' case tables (VisIt's): whole cells
keep their shape (pixels and voxels become quads and hexahedra, as in VTK),
cut cells become tetrahedra, pyramids, wedges and hexahedra (triangles and
quads in 2D). The tables number a wedge's corners as VisIt does, mirrored
from VTK's; Tack reads wedges through that order and writes wedge pieces back
in VTK's, which taken directly would be inside out. Edge points are shared
between cells; point data is interpolated onto them and onto the centroid
points some cases add.

On an edge with an infinite or NaN value at one end, `contour`, `slice` and
`clip` put the crossing at the finite end, and the fields carried to it
take that end's values; interpolating would give NaN.

Without a function: `extract_cells(data, cells)`, where `cells` is ids (a
host array or an integer field), a cell `Set` of `data`, or the name of one
in `data.sets` (cells keep their order, not the order given),
`mask(data, stride)`, `threshold_points(data, field, lower, upper)` and
`mask_points(data, stride)`, the last two as vertex cells.

## Field and geometry transforms

In `tack.dataset.algorithms` (and `tack.dataset`), in Viskores' conventions:

| Function | Result |
|---|---|
| `magnitude(data, f)`, `dot(data, a, b)`, `cross(data, a, b)` | a field in the same space |
| `composite(data, f, g, ...)` | scalar fields of one space as one vector field |
| `log_values(data, f, base, min_value)` | the logarithm, values below `min_value` (default the smallest normal float) taken as it |
| `point_elevation(data, low, high, range)` | position along a line, clamped and mapped to `range` (VTK's elevation filter) |
| `point_ids(data)`, `cell_ids(data)` | ids as implicit counting arrays: nothing stored |
| `warp(data, direction, scale, scale_by)` | the geometry moved along a vector field or a constant vector, optionally scaled by a scalar field |
| `transform(data, matrix)` | the geometry under an affine 4x4 or 3x4 matrix |
| `cylindrical(data, inverse)`, `spherical(data, inverse)` | the geometry in cylindrical `(r, theta, z)` or spherical `(r, theta, phi)` coordinates, or back |

Those that move the geometry return a dataset on the same topology with
every field and set kept; on an `L2` geometry each cell's own corners move.

## Refinement, statistics and sources

| | |
|---|---|
| `tetrahedralize(data)` | 3D cells as tetrahedra by Viskores' tables (hexahedron or voxel 5, wedge 3, pyramid 2), one wedge tetrahedron turned the right way out |
| `triangulate(data)` | 2D cells as triangles: quads and pixels in two, polygons in fans |
| `shrink(data, factor)` | each cell toward its centroid: an `L2` geometry on the same topology, point data as `L2` (Viskores and VTK give each cell its own points) |
| `statistics(data, f)` | a `Statistics` named tuple of Viskores' descriptive statistics: n, min, max, sum, mean, central moments, sample and population variance and deviation, skewness, kurtosis; of a `PartitionedDataSet` too (see *Partitioned datasets*) |
| `entropy(data, f, bins)` | Shannon entropy in bits of a `bins`-bin histogram over the values' range; refuses no values, fewer than one bin, and a range that is not finite (as `numpy.histogram` does); of a `PartitionedDataSet` too |
| `split(data, pieces)` | a structured grid cut into a `PartitionedDataSet` (see *Partitioned datasets*) |
| `sources.wavelet(extent, ...)` | VTK's and Viskores' wavelet (`RTData`) on a uniform grid |
| `sources.tangle(dims)` | Viskores' tangle cube (`tangle`) on [0, 1]^3 |

Tetrahedra and triangles are pieces of their cells (`carry`'s `Pieces`): cell
data goes to each, the points stay the same.

## Locating points, probing and advection

`CellLocator(data, density=1.0)` bins the cells' bounding boxes on a
uniform grid over the dataset, about `density` bins per cell.
`locator.find(points)` takes an `(n, 3)` array or a field of 3-vectors and
gives each point's cell (a field of the dataset's `id_dtype`, -1 where
none holds it) and its
parametric coordinates there. A point counts as in a cell within 0.1% of
the cell's bounding-box diagonal, as `vtkProbeFilter`'s computed tolerance
has it. Of the cells that hold the point the smallest id wins; a cell that
only comes within the tolerance wins only where none holds it, so a point
just inside one cell is not extrapolated in a lower-numbered neighbour.
Advection ranks its cells the same way.
The search runs from the cells: each cell tests the points in the bins its
box overlaps, so a cell is reached through its own view, never looked up by
id. The bins follow the geometry when the locator is built; build it again
after moving the points.

A polyhedron has no parametric coordinates. Its faces are fanned into
triangles from their face points (the mean of each face's points), and the
triangles into *sub-tetrahedra* from the cell point (the mean of the cell's
distinct points); point data is linear on each, taking the means of the
values at those two points, so a linear field is exact and a face's two
cells agree on it. On a polyhedral dataset `find` returns `(cells, tets,
pcs)`: each point's sub-tetrahedron in its cell (i32, numbered face by face
and edge by edge, -1 for none) and its coordinates there, the weights of
the face point and the edge's two points, which a field's view evaluates
with `tet_value(c, t, pc)`. The locator refuses a dataset with a cell that
is not star-shaped from its cell point, whose sub-tetrahedra would overlap.

`probe(data, where, fields=None, locator=None)` evaluates `data`'s fields
at the points of `where`, as Viskores' Probe and VTK's `vtkProbeFilter` do.
`where` is a dataset, which keeps its topology and fields, or an
`(n, 3)` array, which becomes vertex cells. Each field becomes point data
on `where`, alongside `valid` (1 where a cell holds the point, else 0 and
the values are 0):

- a field with a basis (`H1`, `L2`, `Constant`, values on points) is
  evaluated through it in the cell holding the point;
- values on cells come from that cell;
- anything else is left behind.

`locator=` reuses a locator built once for several probes.

Probing a dataset keeps its topology, geometry and id type, and samples its
point positions (the corner values of an order-2 geometry). Host point input
becomes vertex cells using the source's geometry precision and id width.
Advection likewise preserves that geometry precision and id width.

Each shape's `nearest_point(pts, x, pc)` gives the point of a cell nearest
`x`, as VTK's `EvaluatePosition` does: exactly for triangles and
tetrahedra, by clamping the parametric coordinates for the others.

`advect(data, field, seeds, step_size, steps, integrator="rk4",
locator=None)` moves each seed through a steady velocity field -- point
data, interpolated in the cell holding the particle, or cell data -- by
fixed RK4 (or Euler) steps, as Viskores' ParticleAdvection does. It returns
the particles where they stopped, one vertex cell each, with `steps`, `time`
and `status`. `status` holds Viskores' bits, `ParticleStatus.SUCCESS`,
`TERMINATE`, `SPATIAL_BOUNDS`, `TOOK_ANY_STEPS` and `ZERO_VELOCITY`.
`streamlines(...)` returns the paths, the seed first, as Viskores does: a
polyline per seed that moved (a `PolylineTopology`), with `seed` on the
cells and `time` on the points. A seed that took no step keeps its point
and has no line.

The rules are Viskores':

- a step is taken whole when every stage of it lies in the mesh;
- a step that would leave the mesh is bisected to the boundary, and the
  particle is then pushed out by the step still left, ending just outside
  with `SPATIAL_BOUNDS`;
- a particle stops after `steps` steps, outside the mesh, or at zero
  velocity;
- a seed outside the mesh does not move and loses `SUCCESS`.

Each particle's whole path is one thread of one launch. Its stages locate
it from the particle's side (`CellLocator.cell_bins()`, each bin's
candidate cells) and reach the cell by id through the topology's flat
arrays, the cell's type choosing its shape's inversion and interpolation.
The kernel compiles only the shapes the mesh has. At 1M cells, 10,000
seeds and 200 steps it takes 0.27 s on 8 CPU threads and 0.33 s on Metal
for hexahedra (0.34 s and 0.59 s for tetrahedra), against 2.1 s (1.6 s)
for VTK's `vtkStreamTracer` on 8 threads.

## Interoperability

| Function | |
|---|---|
| `tack.interop.vtk.vtk_to_dataset(grid, dtype=tack.f32, polyhedral=None, orient=False, id_dtype=None)` | a `vtkUnstructuredGrid`, rectilinear grid or image (`vtkImageData`, as `UniformCoordinates`). Point data becomes `H1`, cell data `Constant`. A grid with polyhedra becomes polyhedral (`polyhedral=True` forces it); `orient` repairs inconsistent winding |
| `tack.interop.vtk.dataset_to_vtk(data)` | back to VTK, polyhedra and polygons included |
| (both) | a `vtkPartitionedDataSet` and a `PartitionedDataSet` convert piece by piece (null pieces skipped); a `vtkPartitionedDataSetCollection` and a `dict` of names to `PartitionedDataSet`s convert by `vtkCompositeDataSet.NAME()` (`"block<i>"` without one, `"<name>#<i>"` for a name taken) |
| `tack.interop.mfem.mfem_to_dataset(mesh, fields=None, dtype=tack.f32, id_dtype=None)` | an MFEM mesh, curved at order 2 too, with its grid functions |
| `tack.interop.mfem.mfem_field(data, mesh, gf, dtype=tack.f32)` | one grid function: H1 orders 1 and 2, L2 orders 0 and 1, scalars and vectors |

Polyhedral formats such as CGNS arrive through VTK's readers. The CGNS
reader's polyhedra can be inconsistently wound, so pass `orient=True`.

## Settled, and open

**Settled by the prototype:**

- continuity follows from where values live;
- fields are separate kernel arguments, and a domain view is flat: geometry
  and incidence are mixins, not nested templates, since geometry needs the
  cell's connectivity (templates may hold templates, which the implicit
  arrays use, but not refer back to their holder);
- view methods follow one naming rule per kind of answer (ids, positions,
  corners, counts);
- moving a field between spaces is one function, `project`;
- sets know what they select, and filters carry them;
- shape, order and size groups are one mechanism;
- geometry is a field, kept apart from the named fields;
- a field's values may be any array, explicit or implicit;
- polyhedra are a separate path that shares the data model;
- face values have an orientation where it matters.

**Settled by the consolidation (2026-10-16):**

- one name per operation, a variant a parameter (`extract_surface`'s faces,
  `project`'s target space);
- what each kind of algorithm returns: a filter a `DataSet`, an analysis a
  `Field` or a named tuple of them, a summary a number or a named tuple, a
  selection a `Set`;
- one parameter name per meaning (`inside=`, `value`, `lower`/`upper`);
- a dataset computes on its own device, and its parts share one.

The algorithm surface is settled as the core's is: a change to it is a
decision, recorded in the CHANGELOG as breaking.

**Open:**

- coefficient layout (DOF-fastest or cell-fastest);
- filters on higher-order data (subdivide, rather than linearize);
- clip on polyhedra, polygons and polylines;
- locating points in polyhedra that are not star-shaped from their cell
  point (a solid-angle containment test);
- a polyline topology's edges (its segments as global entities), and
  stitching contour lines into polylines.

# Dataset API

`tack.dataset`, in the `tack-vis` package. This page describes the API as it
stands; the [Visualization Guide](../index.md)
develops the model, mesh construction, field spaces, filters, mesh kernels and
complete pipelines with checked examples. For why it is shaped this way and how it got here, see the
design records: [Dataset API](../design/dataset-api.md) and
[Polyhedra](../design/polyhedra.md). They are history; this page is the
current state. What is missing is listed with each part, and under
[Settled, and open](#settled-and-open). Each algorithm's and class's
signature and parameters are in [Algorithms](algorithms.md) and
[Data Model](data-model.md).

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
  It is kept apart from the named fields, as `data.geometry`. Its values are
  relative to the dataset's **origin**, three doubles on the host (*Origins
  and point clouds*).
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
| `StructuredTopology(point_dims, id_dtype=None, offset=None)`, usually through `rectilinear_grid(x, y, z)` | grid dimensions; cells addressed by (i, j, k); `offset`, the grid index of its first point in a larger grid (VTK's extent's start: `split`'s pieces, an image whose extent starts elsewhere) | shape-based |
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
can be too large for `i32` (a hexahedron tetrahedralizes into 24
connectivity entries), so a pipeline that starts in `i64` stays there.
Sort keys are tuples of ids, never ids packed into one integer, so 64-bit
ids sort as 32-bit ones do.

On GPUs without 64-bit atomics (Metal, Level Zero), `CellLocator`'s search
from the cells refuses a dataset of more than 2^30 − 1 cells (it ranks a
point's candidate cells up to twice the cell count in an atomic minimum); the
search from the points uses no atomics, and everything else works there.

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
| `topology.validate()` | checks the arrays a topology is built from, on their device, reading back a few integers: offsets start at 0, never run backwards and end at their members' length; every id is within its range (points; faces, for polyhedra; sides 0 or 1); an unstructured topology's types are linear shapes with their point counts. Raises `ValueError` saying what is wrong and how often, or returns the topology |

Winding comes from the input, not from geometry. Inconsistent input is
refused unless the caller asks for the repair.

A topology is not checked when made: a bad id is read, or written, out of
bounds by whatever uses it first. Call `validate()` on input you did not
make, or turn on debug mode (`tack.set_debug(True)`, `TACK_DEBUG=1`), which
checks every topology's ids when it is made -- before a polygonal or
polyline topology derives anything from them -- along with every kernel's
array accesses.

## Spaces

| Space | Values | Interpolated | Topologies |
|---|---|---|---|
| `H1(data, order=1)` | one per point, shared by the cells around it; order 2 adds one per edge and quad face; order `p` (`nodes=`) `p - 1` per edge, `(p - 1)^2` per quad face, `(p - 1)^d` per cell | yes, continuous | order 1 on all; order 2 on tetrahedra, hexahedra, voxels, wedges, triangles, quads and pixels; any order on segments, quads, pixels, hexahedra and voxels |
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
| `H1` | the points; at order 2 also edge middles, quad face centers and cell centers; at order `p` the products of the 1D nodes on each edge, quad face and cell |
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

## Units

`Field(space, values, unit=None)`: `unit` is the values' physical unit, a
string kept in unyt's canonical spelling (`"m/s"`, `"m*s**-1"` and `"m/s^1"`
are one unit; `"1"` is `"dimensionless"`), or `None` for unknown. The spatial
unit is the geometry's, `data.unit`, and `origin` is in it. Kernels never see
a unit. See [Physical Units](../12-units.md).

| | |
|---|---|
| `Field(..., unit=)`, `field.unit` | the values' unit, or `None` |
| `DataSet(..., unit=)`, `data.unit` | the spatial unit, the geometry field's; `uniform_grid`, `rectilinear_grid` and `point_cloud` take `unit=`, `point_cloud` also `units={name: unit}` |
| `OctreeDataSet(..., units={name: unit})`, `octree.unit`, `octree.units` | the fields' units, and the roots' spatial unit |
| `convert(data, unit, field=None)` | the geometry (origin scaled in double precision; images stay images) or a field in `unit`, scaled and offset on the device |
| `Statistics.unit`, `.weight_unit`, `.unit_of(name)` | the field's and the weights' units, and each statistic's (variances in the square, `sum` times the weights') |
| `interpolate(..., unit=)` | the function's values' unit, which its code does not say |
| quantities as parameters | lengths (`"10 cm"`, `(0.1, "m")`, a unyt or pint quantity) converted to the spatial unit, isovalues and thresholds to the field's, step sizes to the particles' time unit; a mass, density or velocity given as one quantity brings its unit (see [Physical Units](../12-units.md#quantities-as-parameters)) |
| `tack.dataset.units` | `canonical`, `conversion` (factor and offset), `same`, `compatible`, `common`, `derived` (products of powers), `delta` (degrees Celsius to kelvin for differences), `named` (`Pa*m**3` as `J`); the registry is unyt's plus `ftUS`, the US survey foot |

Filters that keep a quantity keep its unit (everything `carry` takes along,
`project`, `values_at_centers`, traces, `magnitude`, means, centres in the
spatial unit); those that compute one derive it (gradients, products,
measures, fluxes, statistics, deposits, SPH, particle times: the table in
[Physical Units](../12-units.md#derived-units)). Positions in two known spatial units do not meet:
probe's points, pieces, AMR boxes and octree probes refuse them. The VTK
interop maps each array's `UNITS_LABEL`.

## Time

Tack holds no series of steps: a simulation in situ, or a reader's loop,
hands filters one step at a time, and each step is a dataset at its time.

| | |
|---|---|
| `DataSet(..., time=, time_unit=)`, `data.time`, `data.time_unit` | the step's time, a double on the host or `None`, in a unit of time or `None` |
| kept by | every filter, as the origin and spatial unit are (`carry`, `keep_marks`, `drop_marks`, `to_device`); `probe` gives back its target's |
| refused | pieces of a `PartitionedDataSet` and boxes of an `AMRDataSet` at two times (compared in one unit); an octree's time is its roots' |
| VTK | `vtkDataObject::DATA_TIME_STEP` both ways; VTK has no time unit |
| `tack.dataset.temporal` | filters fed one step at a time (`update(data)`, which returns the step's output), keeping copies of what they need of earlier steps (`copy=False` keeps references); a `PartitionedDataSet` streams piece by piece |
| `RunningStatistics(fields)` | per point or cell over the steps so far: `<name>_mean`, `_variance` (population), `_min`, `_max` (Welford, f64 where the device has it); `count`; one mesh over the steps (remesh first) |
| `TimeDerivative(fields, copy=True)` | `<name>_dt`, the backward difference since the previous step, in the field's unit over the time unit; `None` on the first step; one mesh, increasing time |
| `Advection(field, seeds=None, integrator="rk4", step_size=None, copy=True)` | particles through the steps, the velocity linear in time between the kept previous step and this one (each probed in its own mesh, over ranks where pieces are); `update` returns them (`seed`, `source`, `release`, `time`, `status`); `inject(points, time=None)` releases more at any time |
| `extend_paths(paths, particles)` | joins a step's particles onto the paths so far, by `seed`: polylines through each particle's points in time |
| `Pathlines(field, seeds, ...)`, `Streaklines(field, seeds, releases=1, ...)` | the current pathlines every step; the line of each seed's released particles, newest first |

See [Streaming Steps](../13-streaming.md).

## Origins and point clouds

A position is `data.origin + value`: the origin is three floats kept in double
precision on the host (`(0.0, 0.0, 0.0)` unless set), the value the geometry's,
in `tack.f32` or, on a device that has it, `tack.f64`. Kernels see only the
values, so coordinates far from zero keep their precision: single precision
resolves 0.25 m at 4,000 km, a LiDAR survey's eastings, but a millimetre
within a few kilometres of an origin. LAS files store scaled integers and an
offset for the same reason.

| | |
|---|---|
| `DataSet(..., origin=)`, `data.origin` | set it; assigning it moves every point (the values stay) |
| `data.world_positions()` | `origin + positions()`, a float64 host array; `positions()` stays the values |
| `data.rebased(origin)` | the same world positions relative to another origin: each value moved by the difference of the origins, taken in double precision, added in the geometry's type. An image stays an image, its corner moved in double precision (a raster at survey coordinates, rebased near it, holds its points exactly); a rectilinear grid moves its axes; other implicit geometry becomes positions per point |
| `point_cloud(positions, fields=None, origin=None, dtype=tack.f32)` | a dataset of one implicit vertex cell per point (types, offsets and connectivity stored nowhere), `fields` as point data. Host positions are world positions, taken relative to `origin` in double precision; without one, float64 positions get the lower corner of their box rounded down, others zero. A Tack array of 3-vectors is relative to `origin` already. Floating-point fields take `dtype`, integer fields (a classification, intensities) keep their type; `tack.f64` is refused on a device without it |

**Filters keep it**: an output dataset has its input's origin (through `carry`,
`keep_marks` and `drop_marks`), as `to_device`, `split`, `ghost_cells` and
`merge_pieces` do. Lengths, areas, gradients and normals do not depend on it.
What takes or gives absolute positions converts on the host in double
precision:

- **Implicit functions** are given in world coordinates. `Plane`, `Sphere`,
  `Cylinder`, `Box` and `Planes` move their parameters by the origin exactly
  (`implicit.relative_to`); another object with a `value` is evaluated at the
  value plus the origin, added in the geometry's type.
- **`probe` points and `advect`/`streamlines` seeds** given as host arrays are
  world positions; a dataset's points (a target, `advect`'s particles) are its
  own, relative to its origin, and moved by the difference of the origins; a
  Tack array of seeds is relative to the data's origin.
- **`point_elevation`'s points** are world positions. **`interpolate`'s
  function** is called with world positions (the value plus the origin, in the
  geometry's type).
- **`transform`** moves the origin by the map, in double precision, and applies
  only its linear part to the values. **`cylindrical` and `spherical`** convert
  world positions, and give an origin of zero.
- **The VTK interop**: `vtk_to_dataset(..., origin=)` takes VTK's points, an
  image's origin and a rectilinear grid's lines relative to it in double
  precision; `dataset_to_vtk` writes world positions (double-precision points
  where the origin is not zero).
- **Position-valued fields** (`cell_geometry(...).centroid`, a field holding
  positions) are relative to the dataset's origin, as its geometry is.
- **`CellLocator.find`** takes positions relative to the dataset's origin.

Pieces whose positions are compared or moved between them must share one
origin: `ghost_cells`, `generate_global_ids`, `merge_pieces`,
`connected_components`, `probe` and advection over a `PartitionedDataSet` refuse
two (`partitioned.common_origin`, over every rank); `rebased` puts a piece on
another. An image's own origin (`UniformCoordinates`) is separate from the
dataset's and computed in the geometry's type.

### Finding points near points

`PointLocator(data, bin_size=None, points_per_bin=4)` puts the points in
uniform bins over their box (about `points_per_bin` a bin, cubes along the
axes they span) and sorts them by bin, each bin in index order
(`buckets.bucket_order`), as VTK's `vtkStaticPointLocator`. Nothing is
atomic, so every query visits the same points in the same order every time,
and distances are compared with each product rounded on its own
(`squared_length`, which no backend fuses), so the neighbours within a radius
and the k nearest are the same on every backend; reported distances agree to
rounding. Like `CellLocator`, it records its
geometry (`serves`) and is refused once Tack has written the positions; the
filters below use the one the topology keeps (`cached_point_locator`) unless
given one.

| | |
|---|---|
| `locator.neighbors(radius, points=None)` | every point within `radius` (inclusive) of each query, CSR `(offsets, ids)`; the queries are the points themselves, each leaving itself out, unless `points` (a host array of world positions, or a Tack array relative to the origin) are given |
| `locator.nearest(k, points=None)` | the `k` nearest (`k` up to `MAX_K`, 32), `(ids, squared_distances)`, each `(m, k)`, nearest first, ties by index; -1 where the points run out. Rings of bins outwards until the k-th distance lies within those searched |
| `locator.grid()` | the sorted points and the grid as a device class, for one's own kernels: `span(x, r)`, `cell(x)`, `bin(i, j, k)`, `first(b)`, `end(b)`, `point(s)`, `offset(p, x)` (point `p` less `x`, to its nearest image in a periodic box) |

`PointLocator(..., periodic=box)` takes a periodic box in world coordinates (a
side, the box from zero; three sides; or `(lower, upper)` corners), as a
cosmology simulation's: the bins tile the box exactly, points outside it wrap
in, and every distance is to a point's nearest image. A search radius must stay
under half the box (`check_radius`). The point filters below take `periodic=`
and pass it on; without one, nothing changes (it is compiled out).

### Point cloud filters

| | |
|---|---|
| `voxel_downsample(data, size, keep="centroid")` | one point per occupied voxel of side `size` (or three sides) anchored at the points' lower corner: the voxel's centroid (new points: floating-point fields averaged, integer fields its first point's; ghost marks dropped), its first point, or its point nearest the centroid (input points, in input order, fields and marks kept). Voxels by a stable sort, each reduced in index order: the same bits run to run, and the same voxels and kept points on every backend. VTK's `vtkVoxelGrid` fits a whole number of bins to the box, so its voxels are not exactly `size`; PDAL's `filters.voxeldownsize` is this |
| `outliers(data, radius=, min_neighbors=)` | marks (`outlier`, u8) points with fewer than `min_neighbors` others within `radius`, with each point's count (`neighbors`): VTK's `vtkRadiusOutlierRemoval`, the same points |
| `outliers(data, k=, deviations=1.0)` | marks points whose mean distance to their `k` nearest is more than `deviations` standard deviations above the mean of those means, with each one (`mean_distance`): VTK's `vtkStatisticalOutlierRemoval` and PCL, the same points, mean and deviation. Marked, not removed, as PDAL marks (class 7): `threshold_points(out, "outlier", 0, 0)` keeps the rest |
| `surface_normals(data, consistent=True)` | a surface's polygons wound alike (each region as its first cell) and its unit normals: `cell_normals` (Newell's) and `normals` at the points (the cells' summed, made unit), as VTK's `vtkPolyDataNormals` without splitting; a non-orientable surface is refused |
| `point_normals(data, k=None, radius=None, viewpoint=None, direction=(0, 0, 1))` | each point's unit normal from the covariance of its `k` nearest points (itself included) or those within `radius`: the eigenvector of the least eigenvalue, in closed form; signed toward `viewpoint` (a world position) or along `direction` (up). Zero with fewer than three neighbours, or on a line. VTK's `vtkPCANormalEstimation` agrees within 0.1° |
| `connected_components(data, radius=, min_size=None)` | with `radius`, points within it of each other connect too: on a point cloud, Euclidean clusters (VTK's `vtkEuclideanClusterExtraction`, one to one) and friends-of-friends halos with the linking length. The radius hook walks the locator's bins (nothing stored per pair, each pair once). `min_size` makes regions of fewer points -1 and numbers the rest again |
| `deposit(data, grid, field=None, reduce=None, method="nearest", per_volume=False)` | points onto an axis-aligned image: into the cell holding each (`"nearest"`, a field on the cells) or onto its corners by trilinear weights (`"cic"`, cloud in cell, on the points), as Viskores' particle density filters and yt. Counts without a field; with one, `"sum"`, `"mean"`, `"count"`, and for `"nearest"` `"min"` and `"max"` (a LiDAR surface model: the highest return per cell of a 2D image). Empty cells hold 0, or NaN for means and extrema; `per_volume` divides by the cell's volume. Points outside are left out. Each target reduced in point order, by a sort: the same bits run to run; counts, minima and maxima the same on every backend, sums and means to rounding |
| `sph_interpolate(particles, target, fields=None, kernel="cubic", smoothing_length="h", mass="mass", density="density", normalize=False, dimension=3)` | SPH values at a target's points (a dataset, or world positions as vertex cells): `sum_j (m_j / rho_j) A_j W(r, h_j)`, each particle reaching to its kernel's support (the scatter form), with `shepard`, the sum of `(m / rho) W`; `normalize` divides by it. Kernels `"cubic"`, `"quartic"`, `"quintic"`, `"wendland"`, VTK's, normalized in 1 to 3 dimensions (the Wendland kernel from 2); within float rounding of `vtkSPHInterpolator`. Smoothing length, mass and density are point fields or numbers; integer fields are not interpolated |
| `sph_density(particles, kernel="cubic", smoothing_length="h", mass="mass")` | each particle's density by the SPH sum with its own smoothing length (the gather form, SPLASH's), itself included |

### Cosmology halos

Dark matter halos are friends-of-friends groups,
`connected_components(particles, radius=linking_length, min_size=..., periodic=box)`,
and `halo_catalog` describes them. Ported from Viskores' cosmotools and
ParaView's LANL halo finder, with what they lacked: per-particle masses,
softening, periodic boxes, an applied minimum size (Viskores stores it and never
uses it), and the pruned most-bound-particle search on every large halo.

| | |
|---|---|
| `halo_catalog(particles, region="point_region", mass=1.0, velocity=None, center="mbp", softening=0.0, G=1.0, periodic=None)` | one point per halo (labels of `region` >= 0), at its most bound particle or centre of mass, with `halo`, `count`, `mass` (per-particle masses or one number), `com`, `mbp` (the index of the particle of least potential `-G sum m / sqrt(r^2 + softening^2)`), `potential`, `radius` (the farthest particle from the point), and with `velocity` its mass-weighted mean and one-dimensional `dispersion`. Each halo is unwrapped about its first particle in a periodic box, and its points put back in. The MBP is exact: halos up to 2048 particles sum every pair; larger ones use Viskores' MxN pruning, tightened by bounding each particle against each far bin's box rather than bin against bin (on a concentrated halo of 40,000, 3% are summed exactly instead of 69%). The catalogue shares the particles' origin; every sum runs in particle order |

### Point clouds in pieces

A point cloud in pieces (a `PartitionedDataSet` of `point_cloud`s, on one
origin) works as a mesh in pieces does: filters over neighbours run per piece
on pieces with copies of their neighbours' points, and the algorithms whose
answer needs every piece take the partitioned dataset.

| | |
|---|---|
| `ghost_points(pds, radius=None, k=None, periodic=None, around=None, fields=None)` | each piece with copies of the other pieces' points it needs appended, vertex cells marked `DUPLICATECELL` at points marked `DUPLICATEPOINT`, owned where they came from. With `radius`, every point within `radius` of the piece's box, found by the boxes in one exchange (`around`, a piece for each, widens a piece's box to its own piece's points: the targets `sph_interpolate` resamples onto there). With `k`, exactly the points among an owned point's `k` nearest: each point's `k` nearest in its own piece bound the true distance, a point whose ball reaches another piece's box asks that piece for its `k` nearest, and the best are fetched. In a `periodic` box a copy across a face is an image, moved by the box's side, so the pieces are ordinary clouds: run filters on them without `periodic`. Fields, sets and global ids come along, and `update_ghosts` refreshes copies from their owners |
| per piece, after `ghost_points` | `point_normals`, `outliers`' radius test, `sph_density` (then `update_ghosts(pds, "density")`), `sph_interpolate` onto `around`'s pieces: each owned point (and each target) gets the whole's answer |
| `connected_components(pds, radius=, min_size=None, periodic=None)` | each piece labelled with copies within `radius`, each copy's region joined to its owner's: the whole's clusters and friends-of-friends halos, numbered as the whole numbers them (by the lowest global id, else (piece, point)). Only regions that cross pieces are gathered; the rest are numbered by a sort over the pieces, so no rank holds every region. `min_size` counts each point once, where it is owned (meshes in pieces take it too) |
| `outliers(pds, ...)` | both tests over copies (`ghost_points` inside), the statistical test's mean and deviation over every piece's points; the pieces come back without the copies |
| `voxel_downsample(pds, size, keep=)` | the grid anchored at every piece's lower corner; each voxel's owned points brought to one piece by a sort over the pieces and reduced there in (piece, index) order. A centroid goes to the piece of its voxel's first point; `"first"` and `"nearest"` keep points on their own pieces |
| `deposit(pds, grid, field=None, ...)` | each piece deposits its owned points; the grids combine over every rank in piece order (sums and counts add, means divide them, extrema take the least or greatest). Every rank gets the whole grid's field; `field` is a name |
| `halo_catalog(pds, region="point_region", mass=, velocity=None, ...)` | each halo's owned particles sorted to one piece by label, in (piece, index) order, and catalogued there: a `PartitionedDataSet` of catalogues, halos in label order across them, `mbp` an index in piece `mbp_piece` |

Pieces cut from a cloud in index order give the whole's answers bit for bit
where the answer is reduced in point order (clusters' numbers, centroids,
halo catalogues, including each MBP); in a periodic box, values measured
through an image agree to a rounding, since an image's position is rounded
once more than the whole's minimum-image offset.

## Adaptive mesh refinement

An `AMRDataSet(levels, ratios=(), domain=None, world=None)` is a
block-structured AMR hierarchy, as VTK's `vtkOverlappingAMR` and AMReX's:
`levels`, each this rank's boxes of a level (a `PartitionedDataSet` or a list
of datasets), and `ratios[l]`, the refinement from level `l` to `l + 1`, a
whole number or one per axis. The hierarchy is in index space alone, as
AMReX's: a box is a dataset of a `StructuredTopology` whose `offset` is its
first cell's index in its level, and its geometry is anything structured --
an image, a rectilinear grid, a curvilinear grid -- since which cells a finer
level covers is a matter of integer boxes. VTK's AMR holds images and (since
9.6) rectilinear grids; curvilinear boxes are Tack's.

| | |
|---|---|
| `amr.levels`, `amr.ratios`, `amr.domain`, `amr.dimension`, `amr.num_levels` | the levels (`PartitionedDataSet`s), the ratios (three per level), level 0's cell-index box |
| `amr.boxes(level)` | every rank's boxes of a level, `(n, 2, 3)` cell-index boxes `[lo, hi)`, gathered over the world when the hierarchy is made, as VTK's metadata and AMReX's `BoxArray` are replicated |
| `for level, box in amr` | this rank's boxes, coarsest level first |
| marks | made with the hierarchy: `REFINEDCELL` on cells a finer box covers; `REFINEDPOINT` on points at whose location a finer level has a point, so a coarse point on the edge of a refined region is one; `DUPLICATEPOINT` on points a lower-numbered sibling also holds. Every location and region is then owned once, at the finest level holding it. Marks given are recomputed; other bits are kept |
| refused | overlapping boxes in a level; a fine box that, coarsened, is not within its coarser level's boxes (proper nesting); a level's images off one lattice (one spacing and origin, axis-aligned: VTK's model); boxes with ghost cells; boxes of mixed dimension |
| `statistics(amr, name)`, `entropy(amr, name, bins)` | over every box of every level, each region and location once |
| per box | every filter: contours, slices, clips and thresholds leave refined cells out (cracks at level boundaries remain); algorithms whose output is on the same cells (gradients, projections) compute there, refined data being valid |
| `amr_gaussian_pulse(dimension=3, ...)` | VTK's `vtkAMRGaussianPulseSource`: two levels, `Gaussian-Pulse` and `Centroid` on the cells |
| `ghost_cells(amr, layers=1, fields=None)` | each box grown by `layers` cells within its level's domain (`amr.level_domain(level)`), the ghosts marked `DUPLICATECELL` and `DUPLICATEPOINT`: a ghost a sibling box holds is its value, copied (AMReX's `FillBoundary`); one no sibling holds takes the coarser level's -- the coarse cell's value for a cell (injection), the coarse cell's corners by linear weights for a point (`FillPatchTwoLevels`) -- read from each value's owner; one neither holds is `HIDDENCELL`/`HIDDENPOINT`. A grid's ghost positions are filled the same way (an image's are its lattice), so rectilinear and curvilinear boxes grow too, a hidden point's position extrapolated along the grid. Ghosts a finer box covers are refined. `amr.layers` says how many |
| `update_ghosts(amr, *names)` | refreshes both kinds of ghost, and the points a box shares with a lower sibling, from their owners, after a filter: one `tack.communication.Plan` per kind, kept, and a kernel summing by weight |
| `probe(amr, where)` | each point evaluated in the finest level's cell holding it (refined cells hold nothing; on a level boundary the finer box wins), over every rank's boxes: a probe onto an image is the hierarchy resampled, as VTK's `vtkAMRResampleFilter` gives it (equal at every point VTK sets; VTK leaves points on the domain's upper faces unset) |
| `average_down(amr, name)` | in place, finest level first: each refined coarse cell the volume-weighted average of the fine cells over it, each refined coarse point the coinciding fine point's value (AMReX's `average_down`) |
| interop | `vtk_to_dataset(vtkOverlappingAMR)` gives an `AMRDataSet` (blocks placed by their `vtkAMRBox`); `dataset_to_vtk(amr)` a `vtkOverlappingAMR` when every box is an image or a rectilinear grid and each ratio is the same along every axis, else a `vtkPartitionedDataSetCollection` of a partitioned dataset per level. VTK's `BlankCells` marks the export as Tack does |

Not here: crack-free contouring across levels, periodic domains, tagging and
regridding, conservative interpolation (the vault's AMR proposal).

## Octrees

An `OctreeDataSet(roots, tree, level, key, fields=None, mask=None, world=None,
units=None)`
is a grid of octrees, VTK's `vtkHyperTreeGrid` with branching factor 2, stored
as a linear octree for GPUs: `roots` the grid of root cells (an image or a
rectilinear grid, 2D or 3D; root cell `t`, x fastest, is tree `t`), and every
node of every tree, refined ones included, as its tree, its level (0 at the
root) and its Morton key at that level (integer coordinates in the tree
interleaved, x lowest, so a child's key is its parent's shifted by the
dimension with its index, x fastest, below). Nodes are sorted in pre-order --
tree, key carried to the finest level, level -- so a subtree is one run. Each
node's ancestors must be given, and a refined node all its children.

| | |
|---|---|
| `octree.tree`, `.level`, `.key`, `.marks` | the nodes, device arrays; marks `REFINEDCELL` on non-leaves, `HIDDENCELL` where masked |
| `octree.fields` | cell data: one value per node (`dict` of arrays) |
| `octree.is_leaf`, `.centers()`, `.measures()`, `.geometry()` | from the keys: which nodes are leaves, centres, volumes (areas), lower corners and sides |
| `octree.leaves()` | the leaves as an unstructured `DataSet` of voxels (pixels in 2D), cell `c` the `c`-th leaf, fields as cell data, hidden leaves `HIDDENCELL`; coinciding corners one point (sorted on the finest level's lattice), hanging ones kept |
| `probe(octree, where)` | each point's leaf values (one binary search of the keys carried to the finest level), `valid` 0 outside, on hidden leaves and off a 2D plane; over ranks, each point to the rank whose trees hold it |
| `OctreeDataSet.from_cells(roots, level, position, fields)` | from cells' levels and positions (whole octs, as RAMSES writes them), ancestors added with NaN values |
| `statistics(octree, name, weights=None, level=None)`, `entropy(...)` | over the leaves (or one level's nodes), every rank's trees |
| interop | `vtk_to_dataset(vtkHyperTreeGrid)` (each tree's breadth-first descriptor read in one call, turned into keys) and `dataset_to_vtk(octree)`; the mask as `HIDDENCELL` |

**Weighted statistics**, for every kind of data: `statistics(data, name,
weights=...)` with `"volume"` (cell measures, cell fields only) or a scalar
field's name on the same entities. `mean` is `sum(w x) / W`, `sum` is `sum(w
x)`, the moments are `sum(w (x - mean)^k)`, `weight` is `W`, the population
variance `m2 / W`, the sample variance `m2 / (W - sum(w^2) / W)` (reliability
weights, NumPy's `aweights`), and skewness and kurtosis take `W` for `n`.
`entropy(..., weights=...)` shares each bin by the weights in it.

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

A `PartitionedDataSet(partitions, world=None)` is a dataset in pieces: an
ordered list of `DataSet`s, as VTK's `vtkPartitionedDataSet`. Each piece is
a whole dataset, with its own topology, numbering, geometry, fields, sets
and device, and may have no cells. Iterate over it, index it (`pds[i]`) or
take its length; `pds.partitions` is the tuple of pieces. Across MPI ranks
it holds this rank's pieces, and `world` (a `tack.communication.World`, this
process alone by default) is where the others are, for the algorithms that
combine pieces.

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
    histogram over it;
  - `connected_components`, `outliers`, `voxel_downsample`, `deposit` and
    `halo_catalog` on point clouds in pieces ("Point clouds in pieces").

  Pieces with no values are skipped. Each piece counts only what it owns
  (below), so where every shared cell and point has one owner, the result
  is the whole's. Across MPI ranks each rank passes its own pieces, and the
  pieces' moments and histograms are gathered over `pds.world` and combined
  in the pieces' global order: the same bits however they are shared out,
  and as in one process.
- **`split(data, pieces, ghost_layers=0)`** cuts a structured grid by index
  ranges, x fastest. `pieces` is a count, factored over the axes (each prime factor,
  largest first, cuts the axis whose pieces have most cells), or a tuple of
  counts, one per axis of the topology. Each axis's cells are shared out as
  evenly as they go, at least one per piece, and neighbouring pieces share
  the points between them. A piece keeps the geometry's kind (uniform
  coordinates with their origin moved, rectilinear axes cut, explicit
  points taken) and carries its block of every point, cell and `L2` field
  and every set, as filters carry them; fields on faces or edges are left
  behind. `ghost_layers` adds that many layers of each neighbour's cells
  around a piece, as far as the grid goes, marked duplicates, as VTK's
  `vtkGhostCellsGenerator` adds them; filters need one (below). A point of
  several pieces' own cells is the first such piece's, x fastest, and a
  duplicate in the others, as are the points of the ghost layers; the
  input's own ghost marks come along. An unstructured mesh is partitioned by
  VTK (`vtkRedistributeDataSetFilter`) or by its host application.
- **`ghost_cells(pds, layers=1, world=None, fields=None)`** gives each
  unstructured piece a layer of its neighbours' cells, as VTK's
  `vtkGhostCellsGenerator` does, in one process (pieces on any devices) or
  across MPI ranks. Pieces neighbour where they share points at exactly the
  same position; a piece gains the cells of others that use a point it
  shares with them (`DUPLICATECELL`) and the points of those it lacks
  (`DUPLICATEPOINT`); a shared point belongs to the lowest piece holding it,
  and is `DUPLICATEPOINT` in the others. That is VTK's rule too: the cells,
  points, marks and values equal `vtkGhostCellsGenerator`'s, and `split`'s
  ghost layer's. Point and cell fields come along (`fields` chooses), and
  point and cell sets keep their members. Boundary points are matched by a
  sort of their positions over every piece
  (`tack.communication.sort_by_key`), and each piece sends each neighbour
  its cells around their shared points. `layers=n` grows each further layer
  from the points of the layer before, cells of any piece included, so pieces
  thinner than `n` cells are fine; two and three layers equal `split`'s and
  `vtkGhostCellsGenerator`'s. **Structured pieces stay structured**: pieces
  that tile a box of cells (each placed by its topology's `offset`, as
  `split`'s and VTK's image pieces are) become their blocks grown by `n`
  cells, with their offsets, their values from the pieces that own them, and
  their coordinates grown with them (uniform, rectilinear, or explicit
  positions); the result is `split`'s ghost layers exactly.
- **Global ids.** `data.global_ids` is a `tack.dataset.GlobalIds` (or
  `None`), kept apart from the fields as the marks are: an `i64` per point
  (`points`) and per cell (`cells`), either of which may be `None`, the same
  on every piece holding the entity and unique otherwise. `vtk_to_dataset`
  takes VTK's `GlobalIds` attributes as them, not as fields, and
  `dataset_to_vtk` writes them back (`GlobalPointIds`, `GlobalCellIds`, the
  names `vtkGenerateGlobalIds` gives); `to_device` moves them. Filters carry
  them where an output entity is an input entity (`carry`'s `Same` and
  `Take`: kept points, kept or extracted cells, `remove_ghosts`), and what a
  filter makes (a contour's points, a clip's pieces, a surface's faces) has
  none, as in VTK.
- **`generate_global_ids(pds, world=None)`** numbers the pieces' points and
  cells, as VTK's `vtkGenerateGlobalIds`: each piece its owned entities in
  its own order, after the pieces before it, and copies their owner's ids.
  Owners come from the ghost marks where there are any; a point several
  pieces hold unmarked (pieces without ghost cells share their boundary
  points) belongs to the lowest of them and is marked `DUPLICATEPOINT` in
  the others. On such pieces the ids and marks equal VTK's. On pieces with
  ghost cells VTK gives a point to the lowest piece holding any copy, ghost
  copies too, and remarks them; Tack keeps the owners the marks say, so an
  owner still holds every cell around its points. Points are copies when
  their positions are exactly equal, cells when their points' ids are;
  pieces from `ghost_cells` reuse its exchanges, and `ghost_cells` carries
  global ids into the ghost layer, matching points by them, where every
  piece has them.
- **`probe(pds, where)`** with a partitioned source: each point goes to the
  lowest piece one of whose own cells (not a ghost) holds it, holding
  ahead of coming within the locator's tolerance, as one dataset ranks its
  cells; that piece evaluates it, and the values come back. The result
  equals probing the whole, points on faces pieces share included, with or
  without ghost cells (a point is evaluated in one cell). `where` is this
  rank's points (a dataset, an array, or a `PartitionedDataSet`, which gives
  one back); across MPI ranks every rank's points are probed over every
  rank's pieces. Built on `tack.communication.Routing`, which particles
  across pieces use too.
- **`merge_pieces(pds, root=0, fields=None, device=None)`** gives the
  pieces back as one dataset, every owned cell and every point once, as
  VTK's `vtkMergeBlocks` after `vtkRemoveGhosts`: for writing one file, or
  for code that knows no pieces. Points are matched and numbered as
  `generate_global_ids` does; ghost cells go, hidden cells stay marked, the
  pieces' own global ids come along. Structured pieces that tile a box come
  back as that box's structured grid (`split`'s input exactly, offset and
  coordinates included); other pieces as an unstructured dataset. Across
  MPI ranks the result is on `root` (`None` elsewhere).
- **`update_ghosts(pds, *fields)`** copies the named point and cell fields'
  values on ghost points and cells from their owners, in place, for pieces
  `ghost_cells` or `ghost_points` made (they keep the exchanges as
  `tack.communication.Plan`s). A filter that needs each entity's neighbours
  (`project` to points, point gradients) is right on what a piece owns and
  wrong on the ghost layer's rim; refreshing the ghosts in between makes a
  second such filter right too.

**Ghost marks.** Every cell and point that several pieces hold has exactly one
owner; every other copy is a ghost. `data.ghosts` is a
`tack.dataset.Ghosts` (or `None`), kept apart from the fields as the
geometry is: a `u8` per cell (`cells`) and per point (`points`), either of
which may be `None`, holding VTK's `vtkGhostType` bits (`Ghosts.DUPLICATECELL`
1, `REFINEDCELL` 8, `HIDDENCELL` 32, `DUPLICATEPOINT` 1, `HIDDENPOINT` 2, and
the rest, carried unread) and Tack's `REFINEDPOINT` 8, which VTK lacks. A
duplicate is a copy of what another piece owns; hidden is data not valid
there (a blanked cell); refined is data a finer AMR level supersedes (see
"Adaptive mesh refinement"). Filters that make cells leave hidden and refined
cells out (`Ghosts.MASKED_CELLS`); reductions leave out every skipped bit
(`SKIPPED_CELLS`, `SKIPPED_POINTS`).

- `vtk_to_dataset` takes a `vtkGhostType` array as the marks, not as a field,
  and `dataset_to_vtk` writes them back; `split` makes them;
  `data.to_device` moves them. Marks of the wrong length or type are refused
  when they are put in.
- `statistics` and `entropy` leave out duplicate and hidden cells and points,
  on one dataset too, so each value counts once across all pieces. An `L2`
  field goes by its cells' marks; a field on faces or edges, or of order 2,
  is refused on a dataset with marks until it is decided who owns those.
- **Filters process the ghost cells and carry the marks**, as VTK does:
  what a piece makes near its edge is what the next step there needs (point
  normals, smoothing, a second filter that looks at neighbours). What a
  filter makes takes the owner of what it came from:
  - an output cell (a contour's triangles, a clip's pieces, a kept cell)
    its source cell's;
  - a kept point its own; a point on an input edge (a contour's, a clip's)
    the edge's; a point inside a cell (a clip's centroids) the cell's;
  - a surface face its cell's; a face two cells share, an edge, and a cell
    with no source cell (a vertex) the owner of its smallest point,
    smallest by position, so every piece holding it decides alike.

  So over pieces with **at least one ghost layer**, the owned part of a
  filter's output is the whole's output, cell for cell and point for point,
  and `statistics` and `entropy` of the outputs are the whole's. Filters
  that keep the cells and points (`warp`, `transform`, `cylindrical`,
  `spherical`, `shrink`, `as_polyhedra`, `as_polygons`, `probe`'s target)
  keep the marks as they are.
- **Pieces without ghost cells** cannot tell what their neighbours make or
  keep from the cells, faces and edges they share: a contour point on a
  shared edge, a surface face on the cut, a shared point only a neighbour's
  kept cells use. Such a filter's output records that it lost the marks, and
  `statistics` and `entropy` refuse it with `NotImplementedError` naming the
  cause, rather than count shared values wrongly. Points chosen by their own
  values (`threshold_points`, `extract_points`, `mask_points`) keep them.
  Setting an output's `ghosts` clears the record.
- **Hidden cells are never processed:** filters that make new cells or
  points (`contour`, `slice`, `clip`, `threshold`, `extract_cells`,
  `extract_geometry`, `mask`, `extract_surface`, `extract_edges`,
  `triangulate`, `tetrahedralize`) and the source of `probe`, `advect` and
  `streamlines` leave them out, as if they were not there. Faces or edges
  chosen by id, and a reused `CellLocator`, are refused on data with hidden
  cells. Outputs on the same cells keep their values there, marked.
- **`remove_ghosts(data)`** takes out the duplicate and hidden cells and the
  points no remaining cell uses, as VTK's `vtkRemoveGhosts`: the last step
  before rendering, writing one file, or code that does not know ghost
  marks. The points kept keep their marks, and a point the piece owns stays
  even when only ghost cells used it, so reductions over the result are
  unchanged. A structured piece whose remaining cells are a block stays a
  structured grid.

```python
pieces = td.split(td.wavelet(20), 8, ghost_layers=1)    # 8 image-data pieces
for piece in pieces:
    piece.fields["RTData_cells"] = td.project(piece, "RTData", td.Constant(piece))
s = td.statistics(pieces, "RTData_cells")               # the whole grid's
iso = td.PartitionedDataSet(td.contour(piece, "RTData", 160.0) for piece in pieces)
s = td.statistics(iso, "RTData")                        # the whole grid's isosurface's
```

`packages/tack-vis/examples/44_partitioned_data.py` runs all of it: pieces
with ghost layers, their marks, work in a loop, reductions of the grid and
of the pieces' isosurfaces checked against the whole, pieces spread over
several GPUs where there are, and `.vtpd` files for ParaView.

**Next** (the partitioned-data proposal's later phases): more than one generated layer, and
the algorithms that need more than ghosts (particles across pieces, probe of
a partitioned source), on `tack.communication`.

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
- There is no `"points"` domain. Point data, like any field's values taken
  one by one, is read by an ordinary kernel over `field.values`.
- `domain` may also be a space: the kernel runs over its nodes, each once
  (`data.node_views(space)`), and takes arrays, not fields. A node view is an
  entity view -- points, edges, quad faces or cells -- whose entity `i` has
  `node_count(i)` nodes, each with `node_dof(i, j)` (where its value is in
  the space's values) and `node_position(i, j)` (where the geometry puts it):

  ```python
  @tack.kernel
  def sample(nodes, values):
      for i in nodes:
          for j in range(nodes.node_count(i)):
              x = nodes.node_position(i, j)
              values[nodes.node_dof(i, j)] = x[0] * x[1]

  space = td.H1(data, order=2)
  values = tack.zeros((space.size,), data.dtype)
  td.for_each(sample, data, space, values)
  ```

  `H1`'s nodes are its points and, at order 2, its edges' middles, its quad
  faces' centers and its hexahedra's and quads' interiors -- a shared node's
  position is the mean of its entity's points on a straight geometry, the
  geometry's own value on an order-2 one; `L2`'s are each cell's corners
  (order 1) or parametric center (order 0); `Constant`'s and values on
  cells' the parametric centers; values on points' the points. An `L2`
  geometry gives a shared node no single position and is refused for them;
  on polyhedra, polygons and polylines only point nodes exist.
  `interpolate` is one such kernel.

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
`tack.dataset` re-exports them all; [Algorithms](algorithms.md) lists them by
what they do, with their parameters. They share one set of conventions:

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
raise `KeyError`.

In the table,
"polyhedral" means `PolyhedralTopology`, "polygonal" `PolygonalTopology`
and "polyline" `PolylineTopology`.

| | Shape-based | Polyhedral | Polygonal | Polyline |
|---|---|---|---|---|
| `face_geometry` (normals, areas), `face_centers`, `cell_geometry` (volume, area or length; centroid) | yes | yes | yes | yes |
| `cell_quality` (scaled Jacobian, Jacobian ratio, condition, edge ratio, shortest and longest edge) | yes (order 2 along its curved edges; a pyramid by its four tetrahedra) | yes (by its fan tetrahedra) | yes (by its fan triangles) | edges only |
| `feature_edges` (boundary, feature, non-manifold, manifold edges of a surface) | yes (2D cells) | no (a volume: `extract_surface` first) | yes | no |
| `edge_lengths`, `extract_edges` | yes | yes | yes | no (no derived edges) |
| `boundary_faces`, `extract_surface` | yes (a 2D mesh's boundary: line cells; by default `extract_surface` passes a 2D mesh's cells through) | yes (the surface is polygonal) | yes (boundary: line cells; by default the polygons pass through) | yes (vertex cells) |
| `jump` and `upwind_flux` of cell data; `net_flux`, `divergence`, `perot` of face values | yes | yes | yes | yes |
| `threshold`, `extract_geometry`, `extract_cells`, `mask` | yes | yes | yes | yes |
| `contour`, `slice`, `slice_plane` | yes (`dimension=` 3, 2 or 1 picks the cells of a mixed mesh, by default its highest; a 2D mesh's: line cells; lines': vertex cells; an `L2` field or geometry gives each cell its own points; on segments, quads and hexahedra a field above order 1 or a curved geometry is cut on linear sub-cells, `subdivision=` steps along each axis or as many as `tolerance=` needs; elsewhere an order-2 `H1` field is refused, since its corners alone would be contoured) | yes (López, face-based) | yes (line cells) | yes (`contour`: vertex cells, of point data) |
| `clip` | yes (with an `H1` geometry) | yes (into polyhedra: faces clipped once, iso-polygons as caps) | not yet | not yet |
| `extract_points`, `threshold_points`, `mask_points` | yes | yes | yes | yes |
| `implicit_values` | yes | yes | yes | yes |
| `CellLocator`, `probe` | yes | yes (in sub-tetrahedra; cells star-shaped from their cell point) | yes (in triangles fanned from the cell point; star-shaped too) | yes (on segments) |
| `advect`, `streamlines` | yes (2D and 3D cells) | yes (in sub-tetrahedra; cells star-shaped from their cell point) | yes (in triangles fanned from the cell point) | no |
| `cell_centers`, `values_at_centers` | yes | yes (at the cell point, the mean of its points; point and cell data) | yes (likewise) | yes (likewise) |
| `gradients(at="cells" | "points")` of scalars or vectors, `flow_quantities` | yes | yes (Green–Gauss over each cell's faces, fanned from their points' means: exact for linear fields) | yes (in the polygon's plane, Green–Gauss over its edges; 2D cells on shapes in their plane too) | yes (zero) |
| `project` between cell and point values | yes | yes (over each cell's distinct points) | yes | yes |
| `project` into or from `L2`, `H1` above order 1, `SideTraces` | yes | no | no | no |
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
| point data (`H1`, values on points) | through the point map; interpolated only if floating point |
| `H1` above order 1, a curved geometry | at its order and nodes when whole cells are kept or taken (not pieces): each output cell's nodes are its source cell's; elsewhere its corner values, except on a solid's surface (`extract_surface`), whose nodes are the solid's, gathered at order 2 |
| cell data (`Constant`, values on cells) | through the cell map, keeping its kind |
| `L2` (DG) data | copied cell by cell, each cell keeping its order, when whole cells are kept (not pieces) and the output has reference cells |
| values on faces | through the face map, oriented values negated where a face is turned; or onto the output's cells when they are input faces (a surface) |
| values on edges | through the edge map, given only where the edges are the input's (`as_polyhedra`) |
| traces, anything without a map | left behind |

**Provenance** comes from the same rules: a cell field of the input's cell
ids, `data.fields["source"] = td.cell_ids(data)` (an implicit counting
array, nothing stored), reaches every output cell as the id of the input
cell it came from -- kept, a piece of, or a face of -- and `point_ids` does
the same for kept points. No filter needs an option for it.

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
filter states its maps and calls `td.carry(data, out, points=..., cells=...,
fields=fields)`. The maps, `carry`, `keep_marks`/`drop_marks`, `visible`,
`owned_values`, `Set.from_flags` and `td.units` are public, so a filter of
your own is written as the built-ins are
([Filters of your own](data-model.md#filters-of-your-own));
`test_public_filters.py` rewrites built-ins with public names only and holds
them to the originals.

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
(axis-aligned) and `Planes(origins, normals)` (the convex region below every
plane: a frustum, a box at any angle; its planes are evaluated at the data's
precision, as the others' numbers are). `value(p)` is negative inside, zero on
the surface, positive outside, in VTK's and Viskores' forms -- quadrics for
the sphere and cylinder, signed distances for the plane, planes and box -- so
a slice by one lands on the same points as theirs.

`Union(f, g, ...)`, `Intersection(f, g, ...)` and `Difference(f, g, ...)`
combine functions as VTK's `vtkImplicitBoolean` does: the smallest value, the
largest, and the first's or the rest's union's negated, whichever is larger.
They hold their operands as nested device classes, so a kernel inlines the
whole tree, and move them by an origin and match a `Planes` to the data as
each operand would be.

Each is a `@tack.device_class` object whose parameters are instance values,
so moving or resizing one compiles nothing anew; a kernel takes it as a
template and calls `f.value(p)`. Their parameters are world positions: on a
dataset with an origin, each is moved by it in double precision (*Origins and
point clouds*). Taking one:

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

`contour(data, field, value)` takes a sequence of values too, and
`slice` and `slice_plane` take one as `value` (the function's levels; for a
plane, signed distances along its unit normal, so parallel planes): every
value's contour in one dataset, made in one pass over the cells -- each cell
compared with every value, emitting for those it crosses -- with each output
cell's value in a cell field `contour_value` (of the field's type). Points
are not shared between values. One value, not in a sequence, adds no field.
For 10 values on 1M hexahedra the one pass took 38 ms against 80 for ten
contours on an M1 Max's CPU (a loaded host); merging the larger output is
most of what is left.

Where the isovalue equals a point's value, `contour` and `slice` take the
crossings of every edge at that point to be the point itself, one output
point shared by every cell there, with the point's own values (a value
equal to the isovalue counts as above it). A triangle, polygon or line with
two of its crossings at one point has no extent and is dropped (a polygon
keeps its distinct points if three are left), as warp-simdata drops it. VTK
and Viskores keep such cells and points, so their counts are higher where a
level runs through mesh points (a plane through grid points, an isovalue
some points have exactly); the areas agree.

`extract_extent(data, extent, rate=1)` keeps a box of a structured grid's points, every
`rate`-th, as a structured grid of the same kind (VTK's `vtkExtractGrid`).

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
| `convert(data, unit, field=None)` | the geometry, or a field, in another unit ([Units](#units)) |
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
| `tetrahedralize(data)` | 3D cells as tetrahedra that meet face to face, as VTK's `vtkDataSetTriangleFilter` makes them: a structured grid's hexahedra in 5, the two mirror splits alternating with the parity of (i, j, k) in the whole grid (the topology's `offset` added); any other cell the cone from its corner of smallest point id -- global id, where the dataset has global ids -- over the faces without it, each quadrilateral face split from its own smallest corner, so neighbours split a shared face alike (hexahedron or voxel 6, wedge 3, pyramid 2), across the pieces of a partitioned dataset too. Viskores' fixed tables, used before, did not conform |
| `triangulate(data)` | 2D cells as triangles: quads and pixels in two, polygons in fans |
| `shrink(data, factor)` | each cell toward its centroid: an `L2` geometry on the same topology, point data as `L2` (Viskores and VTK give each cell its own points) |
| `statistics(data, f)` | a `Statistics` named tuple of Viskores' descriptive statistics: n, min, max, sum, mean, central moments, sample and population variance and deviation, skewness, kurtosis; of a `PartitionedDataSet` too (see *Partitioned datasets*) |
| `entropy(data, f, bins)` | Shannon entropy in bits of a `bins`-bin histogram over the values' range; refuses no values, fewer than one bin, and a range that is not finite (as `numpy.histogram` does); of a `PartitionedDataSet` too |
| `split(data, pieces, ghost_layers=0)` | a structured grid cut into a `PartitionedDataSet`, with ghost layers (see *Partitioned datasets*) |
| `remove_ghosts(data)` | `data` without its duplicate and hidden cells, keeping the points it owns (see *Partitioned datasets*) |
| `connected_components(data)` | which cells are connected through shared points, as VTK's `vtkConnectivityFilter` (all regions, colored) and Viskores' `CellSetConnectivity`: `data` with `region` on the cells and `point_region` on the points, numbered from 0 in order of their lowest cell (VTK's numbering); -1 for hidden cells and unused points; union-find on the device. Of a `PartitionedDataSet` too: regions numbered across every piece and rank, in order of their lowest global cell id where pieces have global ids (the whole's numbering for the whole's ids) |
| `merge_points(data, tolerance=0)` | coincident points made one, as VTK's `vtkStaticCleanUnstructuredGrid` and Viskores' `CleanGrid` merge them: points at most `tolerance` apart join, chains of them too (union-find through the point locator's bins, each set's root its smallest point whatever the order), and each set becomes its smallest point, keeping its position, point fields, marks and global id; the cells are the same on the merged points. For meshes written without merging their points, which must be merged before their pieces can be matched: the 512 pieces of a simulation's output, each storing a point about 2.6 times, merged piece by piece then numbered by `generate_global_ids`, give the merged mesh's 6,310,285 points. Unstructured topologies |
| `ghost_cells(pds, layers=1, tolerance=0)` | unstructured pieces with a layer of their neighbours' cells, as `vtkGhostCellsGenerator` (see *Partitioned datasets*); with `tolerance`, pieces share points at most that far apart, for copies that differ in their last bits |
| `update_ghosts(pds, *fields)` | ghost values refreshed from their owners, in place, for pieces `ghost_cells` made |
| `merge_pieces(pds, root=0, tolerance=0)` | the pieces as one dataset, each cell and point once (see *Partitioned datasets*); `tolerance` as `generate_global_ids` takes it |
| `generate_global_ids(pds, tolerance=0)` | the pieces' points and cells numbered across all of them, as `vtkGenerateGlobalIds` (see *Partitioned datasets*); with `tolerance` (VTK's `Tolerance`), points of different pieces at most that far apart are one, and so are chains of them. Points a piece repeats within itself stay apart: `merge_points` merges them first |
| `sources.wavelet(extent, ...)` | VTK's and Viskores' wavelet (`RTData`) on a uniform grid |
| `sources.tangle(dims)` | Viskores' tangle cube (`tangle`) on [0, 1]^3 |

Tetrahedra and triangles are pieces of their cells (`carry`'s `Pieces`): cell
data goes to each, the points stay the same.

## Locating points, probing and advection

`voxelize(data, grid, fields=None, samples=4)` puts ``data`` on the voxels
(cells) of an image: `fraction`, the share of each voxel the data covers, and
each field's mean over it, as cell data, by `samples` probes a side per voxel
(the same bits every run; batched); voxels the data does not reach are
`HIDDENCELL`. It takes what `probe` takes (datasets, pieces, AMR hierarchies,
octrees); with one sample it is the probe at the voxel centres.

`CellLocator(data, density=1.0, levels=2)` bins the cells' bounding boxes on a
two-level grid over the dataset, as Viskores' `CellLocatorTwoLevel` does: a
coarse uniform grid of about 32 cells a bin, each bin split into a uniform
grid of leaves sized by its count, about 8 cells a leaf. A point is searched
for among its leaf's cells, so a mesh graded toward a body in a large far
field (the ONERA M6 wing's) is not all in a few bins. `CellLocator(data,
levels=1)` bins the cells on one uniform grid of about `density` bins a cell
instead, as Viskores' `CellLocatorUniformBins` and VTK's
`vtkStaticCellLocator` do: on a small, even mesh a search there scans fewer
candidates (6.6 a location against 12 on tack-examples' fire mesh), and
on a graded one a few bins hold most of the cells. Pass it as `locator=`; both levels find the same
cells. Where the search runs
from the cells instead (cells a by-id search cannot read, or a fresh
locator's first small query), the points are bucketed into a uniform grid of
about `density` bins per cell. It records
the topology, the geometry's values and their state when made;
`locator.serves(data)` says whether it still locates in `data`'s cells.
`probe`, `advect` and `streamlines` take a locator for any dataset of the
same topology and geometry, and refuse one whose positions Tack has written
since (a kernel, `fill` or `copyto` into them). Given none, they use the
locator the topology keeps (`tack.dataset.locator.cached_locator`), made
once and again when the geometry changes or is replaced: repeated probes of
a mesh build its bins once. Writes Tack does not make, through memory it
shares (DLPack, `export_memory`, `wrap_ptr`), need a new locator. The state
is a write count every array's storage keeps (`DeviceBuffer.version`,
counted by host writes and by every launch that may store to it).
`locator.find(points)` takes an `(n, 3)` array or a field of 3-vectors and
returns `Locations(cells, pcs, tets)`, a named tuple: each point's cell (a
field of the dataset's `id_dtype`, -1 where none holds it), its parametric
coordinates there, and `tets`, which is `None` except in polyhedra (below). A point counts as in a cell within 0.1% of
the cell's bounding-box diagonal, as `vtkProbeFilter`'s computed tolerance
has it. Of the cells that hold the point the smallest id wins; a cell that
only comes within the tolerance wins only where none holds it, so a point
just inside one cell is not extrapolated in a lower-numbered neighbour.
A cell holds a point within 1e-5 of its parametric domain
(`shapes.HOLD_TOLERANCE`; of a sub-triangle or sub-tetrahedron in a fanned
cell), so a point on a face, edge or vertex that cells share is held by
all of them and goes to the smallest id: without the margin which cells
held it was rounding noise, and the searches from the points and from the
cells disagreed there. Advection ranks its cells the same way.
The search runs from the points: one launch, each point testing the cells
its bin may hold (`cell_bins()`, cells reached by id), and `probe` reads point
and cell data the same way, one launch per field. Where cells cannot be read
by id -- an order-2 or `L2` geometry, line or vertex cells -- it runs from the
cells instead, each cell testing the points in the bins its box overlaps
through its own view; so does a fresh locator's first query of fewer points
than a quarter of its cells, sparing the cell bins until it is used again.
Both rank candidates alike and give the same answer. The bins follow the
geometry when the locator is built; build it again after moving the points.

A polyhedron has no parametric coordinates. Its faces are fanned into
triangles from their face points (the mean of each face's points), and the
triangles into *sub-tetrahedra* from the cell point (the mean of the cell's
distinct points); point data is linear on each, taking the means of the
values at those two points, so a linear field is exact and a face's two
cells agree on it. On a polyhedral dataset `find`'s `tets` is each point's
sub-tetrahedron in its cell (i32, numbered face by face and edge by edge,
-1 for none), and `pcs` its coordinates there, the weights of the face
point and the edge's two points, which a field's view evaluates with
`tet_value(c, t, pc)`. The locator refuses a dataset with a cell that
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
locator=None, tolerance=None, min_step=None, max_step=None, max_length=None)`
moves each seed through a steady velocity field -- point data, interpolated
in the cell holding the particle, or cell data -- by fixed RK4 (or Euler)
steps, as Viskores' ParticleAdvection does, or by RK45 steps that adapt
(`integrator="rk45"`: Cash-Karp, VTK's `vtkRungeKutta45`, with its rules: a
step whose error estimate passes `tolerance` is tried again smaller,
`0.9 h (err/tol)^-0.25`, keeping its first stage; a step taken proposes the
next, `0.9 h (err/tol)^-0.2`, within `[min_step, max_step]`; one at
`min_step` is taken whatever its error; a push out at a wall that moves the
particle less than `min_step` ends it, `SPATIAL_BOUNDS`, as VTK's RK45 ends it
out of its domain). A particle's step goes with it
between a GPU's launches and from piece to piece. `max_length` stops a
particle whose path has gone that far (in the length unit its last step
lands on it). It returns
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

**Going on.** `advect` takes its own output as `seeds`: the particles go on
from where they stopped, with their times and the steps they took (`steps`
counts those too), and a particle that stopped outside, at zero velocity
or after the new limit stays. Two calls of 20 and then 40 steps give 40
steps' particles exactly; a `seed` field comes along.

**Over pieces.** Given a `PartitionedDataSet` (and the field's name),
`advect` and `streamlines` let particles cross from piece to piece, in
rounds: each particle in flight goes to the lowest piece one of whose own
cells (not a ghost) holds it, found from the pieces' boxes and their
locators; each piece advances what it took; repeat until none is in
flight. On pieces with ghost cells a particle stops after a whole step
that ends in a ghost cell and the cell's owner goes on with whole steps,
so the paths are the whole mesh's, up to rounding, wherever a step
(`step_size` times the speed) fits in the ghost layer. Without ghost cells
a particle crosses by the short step to the boundary and the push out, so
it differs by about a step. `advect` returns a `PartitionedDataSet` of the
particles on the pieces they stopped in (or left the domain from), each
with a `seed` (its number among all the seeds); `streamlines` one of each
piece's stretches of the paths, with `seed` and `segment` on the cells,
each stretch after the first starting at a `DUPLICATEPOINT`, so every
path's points are owned once. Pieces may be on any devices, or on MPI
ranks, each rank passing its own seeds, numbered after the ranks before
it; one process gives the same particles.

Each particle's whole path is one thread of one launch. Its stages locate
it from the particle's side (`CellLocator.cell_bins()`, each bin's
candidate cells) and reach the cell by id through the topology's flat
arrays, the cell's type choosing its shape's inversion and interpolation.
The kernel compiles only the shapes the mesh has. At 1M cells, 10,000
seeds and 200 steps it takes 0.27 s on 8 CPU threads and 0.33 s on Metal
for hexahedra (0.34 s and 0.59 s for tetrahedra), against 2.1 s (1.6 s)
for VTK's `vtkStreamTracer` on 8 threads.

A stage first tries the cell the particle was last found in, taking it only
where it holds the point well inside (`shapes.WELL_INSIDE`, a thousandth of
the cell, so no other cell of the mesh holds it and the search would give the
same answer). About one location in seven misses, mostly next to a face.
`advect(..., neighbors=True)` and `streamlines(..., neighbors=True)` then try
that cell's face neighbours, by the same rule, before the bins: a point that
left its cell is mostly across a face (78% of the misses on the ONERA M6, 43-49%
on the aneurysm), which a few cells' tests find where the bins test 50 to 100.
Streamlines on the CPU take 1.5-2x less time there, the paths the same bit for
bit. The table of neighbours (`CellLocator.cell_neighbors()`, six ids a cell)
is built from the faces on first use and kept with the locator: 0.9 s and 210
MB for the aneurysm's 8.7M cells, 4.1 s and 1.1 GB for the M6's 46.9M. Only
solid cells walk: pyramids, polyhedra and surfaces search the bins as before.

## Interoperability

| Function | |
|---|---|
| `tack.interop.vtk.vtk_to_dataset(grid, dtype=tack.f32, polyhedral=None, orient=False, id_dtype=None)` | a `vtkUnstructuredGrid`, `vtkPolyData` (as the unstructured grid of its cells; strips as their triangles), rectilinear grid or image (`vtkImageData`, as `UniformCoordinates`). Point data becomes `H1`, cell data `Constant`. A grid with polyhedra becomes polyhedral (`polyhedral=True` forces it); `orient` repairs inconsistent winding |
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

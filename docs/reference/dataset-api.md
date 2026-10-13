# Dataset API

`tack.data`, in the `tack-vis` package. This page describes the API as it
stands; the [User's Guide chapter](../users-guide/09-visualization.md) is a
tour of it. For why it is shaped this way and how it got here, see the
design records: [Dataset API](../design/dataset-api.md) and
[Polyhedra](../design/polyhedra.md). They are history; this page is the
current state. What is missing is listed with each part, and under
[Settled, and open](#settled-and-open).

```python
import tack.data as td
from tack.data import algorithms as alg
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
`tack.field`.

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
dataset of more than 2^31 − 1 cells; everything else works there.

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
  `face_centers` its midpoint. `divergence`, `perot`, `jump` and
  `upwind_flux` then work on 2D meshes as on 3D ones.
- A mesh of line cells (a 1D grid's, or VTK lines) has no faces yet: a
  line's would be its end points, which are not derived, so `faces()` and
  everything built on it (`face_geometry`, `external_faces`, `traces`,
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
| `check_winding(data)` | of polyhedra or polygons, the ids of the cells whose faces do not wind consistently; empty when all is well (a shape's faces come from its table) |
| `orient(topology)` | turns around faces wound into their only cell |
| `from_cell_faces(..., orient=True)` | repairs input whose faces or cells are inconsistently wound, then turns each connected piece outward by the sign of its total volume |

Winding comes from the input, not from geometry. Inconsistent input is
refused unless the caller asks for the repair.

## Spaces

| Space | Values | Interpolated | Topologies |
|---|---|---|---|
| `H1(data, order=1)` | one per point, shared by the cells around it; order 2 adds one per edge and quad face | yes, continuous | order 1 on all; order 2 on tetrahedra, hexahedra, voxels and wedges |
| `L2(data, order=1)` | each cell's own copies: order 1 one per corner, order 0 one per cell; `order` may be an array, one order per cell | yes, discontinuous | shape-based |
| `Constant(data)` | one per cell | yes, constant in the cell | all |
| `Values(data, on, oriented=False)` | one per point, edge, face or cell: data about the entity, with no basis | no | all |
| `SideTraces(data)` | per face, per side, per face point; made by `traces` | through the face's shape | shape-based, 3D |

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
| `SideTraces` | each side's value from its cell (`traces`) | always |

Where cells disagree -- cell data, DG data -- the `H1` mean is VTK's
averaging (vtkCellDataToPointData). Sums run in a fixed order. Scalars or
vectors; floating point. Values on faces or edges have no basis and are
refused. On polyhedra, polygons and polylines only cell and point values
project, each into the other.

Unused points retain their values when the source is H1 (or values on points).
Cell or DG sources give unused points zero, since no cell contributes there.

The shortcuts are projections: `to_points(data, f)` is
`project(data, f, H1(data))`, `to_cells` projects into `Constant(data)`,
and `discontinuous` into `L2(data)`.

## Fields and arrays

`Field(space, values)`: the values are a `tack.field` (scalars) or a
`tack.Vector.field` (vectors), or an implicit array:

| Array | Value `k` | VTK | Viskores |
|---|---|---|---|
| `ConstantArray(value, size, dtype=None)` | `value`, a scalar or a vector | `vtkConstantArray` | `ArrayHandleConstant` |
| `CountingArray(size, start=0, step=1, dtype=None)` | `start + step * k` | `vtkAffineArray` | `ArrayHandleCounting` |
| `CartesianProduct(x, y=(0.0,), z=(0.0,))` | a rectilinear grid's point, from three axes | `vtkStructuredPointArray` | `ArrayHandleCartesianProduct` |
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
NumPy array is refused as values: put it in a `tack.field` first.

`uniform_grid(dims, origin, spacing, direction=None)` is a dataset on
`UniformCoordinates`, as `rectilinear_grid(x, y, z)` is on a
`CartesianProduct`; `sources.wavelet` and `sources.tangle` are uniform
grids. The VTK interop reads and writes `vtkImageData` as uniform grids,
direction matrix included.

**Geometry.** `DataSet(topology, geometry, fields=None, sets=None,
dtype=tack.f32)` takes the geometry as an `H1` or `L2` field of 3-vectors,
or as positions per point (any array of 3-vectors), which become an `H1`
field. It is kept apart from `fields`: `data.geometry` returns it, assigning
`data.geometry` replaces it (checked as the constructor checks it), and
`data.positions()` gives the positions on the host. Algorithms move it with
the cells; it is never among the fields they carry or select. An `L2` geometry pulls
cells apart without changing the topology. A geometry of order 2 is curved.

## Sets

A `Set(data, on, ids)` selects some of a topology's points, cells or faces
(`on` is `"points"`, `"cells"` or `"faces"`). Its `ids` are kept sorted and
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
- **Iteration.** A cell or face set's name is an iteration domain:
  `for_each(kernel, data, "inlet", ...)`. A cell set runs over the cell
  groups' subgroups of its members, so field views and varying spaces work
  as on all cells. A point set is not a domain (points have no view): loop
  over its `ids`.
- **Carried through filters**, by the maps that carry fields: an output
  entity is in a set when the input entity it came from is. Kept cells keep
  their sets (`threshold`, `extract_cells`), pieces of cells their cells'
  (`contour`, `clip`), and a face set becomes a cell set where faces become
  cells (`external_faces`, `extract_surface`). A set of entities the filter
  makes anew -- interpolated points, faces derived afresh -- is left
  behind. Filters that keep the topology keep every set.

**Open:** boundary entities from readers mapped onto sets (Exodus side and
node sets, MFEM boundary attributes).

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

- `domain` is `"cells"`, `"faces"`, `"edges"` or the name of a cell or face set.
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

Every algorithm is in `tack.data.algorithms`, one module each, and
`tack.data` re-exports them all. They share one set of conventions:

- the dataset comes first: `f(data, ...)`;
- a field argument is a `Field` or the name of one of `data`'s fields;
- an algorithm that makes a new dataset takes `fields=` (all by default, a
  name, a list of names, or `[]` for none) and returns a `DataSet`;
- an algorithm that computes a field returns a `Field`, and several fields a
  named tuple of them (`cell_geometry`, `face_geometry`, `flow_quantities`);
- summaries (`statistics`, `entropy`) return Python values;
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
| `edge_lengths` | yes | yes | yes | no (no derived edges) |
| `boundary_faces`, `extract_surface`, `external_faces` | yes (a 2D mesh's: line cells) | yes (the surface is polygonal) | yes (line cells) | yes (vertex cells) |
| `jump` and `upwind_flux` of cell data; `divergence`, `perot` of face values | yes | yes | yes | yes |
| `threshold`, `extract_geometry`, `extract_cells`, `mask` | yes | yes | yes | yes |
| `contour`, `slice`, `slice_plane` | yes (a 2D mesh's: line cells; an `L2` field or geometry gives each cell its own points) | yes (López, face-based) | yes (line cells) | no (would give points) |
| `clip` | yes (with an `H1` geometry) | not yet | not yet | not yet |
| `extract_points`, `threshold_points`, `mask_points` | yes | yes | yes | yes |
| `implicit_values` | yes | yes | yes | yes |
| `CellLocator`, `probe` | yes | not yet | not yet | not yet |
| `advect`, `streamlines` | yes (2D and 3D cells) | no (no parametric coordinates) | no | no |
| `cell_centers`, `values_at_centers` | yes | no | no | no |
| `gradients(at="cells" | "points")` of scalars or vectors, `flow_quantities` | yes | no | no | no |
| `project` (and `to_points`, `to_cells`) between cell and point values | yes | yes (over each cell's distinct points) | yes | yes |
| `project` into or from `L2`, order-2 `H1`, `SideTraces` (and `discontinuous`) | yes | no | no | no |
| `traces`, and `jump`/`upwind_flux` of point data | yes | no | no | no |

Where the table says "no", the algorithm needs a reference element and
raises `NotImplementedError` saying so.

**How fields carry through filters.** Every filter that makes a new dataset
(contour, slice, threshold, external faces and `extract_surface`,
`as_polyhedra`, `as_polygons`) carries fields the same way, through
`tack.data.carry`. The filter states how its output's entities come from the
input's -- the same entity (`Same`), input entity `ids[i]` (`Take`), a piece
of input cell `ids[i]` (`Pieces`), or a point between two input points
(`Interpolate`) -- and one set of rules applies to every field:

| Input field | On the output |
|---|---|
| point data (`H1`, values on points) | through the point map; an order-2 field brings its corner values; interpolated only if floating point |
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
| `external_faces`, `extract_surface` | the same points | pieces (each face's cell); face values become cell values | — | — |
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

`tack.data.Plane(origin, normal)`, `Sphere(center, radius)`,
`Cylinder(center, axis, radius)` (infinite), `Box(lower, upper)`
(axis-aligned) and `Planes(origins, normals)` (the convex region below every
plane: a frustum, a box at any angle). `value(p)` is negative inside, zero on
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

`clip(data, by, value=0.0, invert=False)` keeps the part of every cell where
`by` -- point data or an implicit function -- is at or above `value` (below,
with `invert`), cutting cells by Viskores' case tables (VisIt's): whole cells
keep their shape (pixels and voxels become quads and hexahedra, as in VTK),
cut cells become tetrahedra, pyramids, wedges and hexahedra (triangles and
quads in 2D). The tables number a wedge's corners as VisIt does, mirrored
from VTK's; Tack reads wedges through that order and writes wedge pieces back
in VTK's, which taken directly would be inside out. Edge points are shared
between cells; point data is interpolated onto them and onto the centroid
points some cases add.

Without a function: `extract_cells(data, ids)` (cells keep their order),
`mask(data, stride)`, `threshold_points(data, field, lower, upper)` and
`mask_points(data, stride)`, the last two as vertex cells.

## Field and geometry transforms

In `tack.data.algorithms` (and `tack.data`), in Viskores' conventions:

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
| `point_cloud(data)` | every point as a vertex cell, with the point data |
| `statistics(data, f)` | Viskores' descriptive statistics: n, min, max, sum, mean, central moments, sample and population variance and deviation, skewness, kurtosis |
| `entropy(data, f, bins)` | Shannon entropy in bits of a `bins`-bin histogram |
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
parametric coordinates there. Where cells share the point, the smallest
cell id wins. A point counts as in a cell within 0.1% of the cell's
bounding-box diagonal, as `vtkProbeFilter`'s computed tolerance has it.
The search runs from the cells: each cell tests the points in the bins its
box overlaps, so a cell is reached through its own view, never looked up by
id. The bins follow the geometry when the locator is built; build it again
after moving the points.

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

**Open:**

- coefficient layout (DOF-fastest or cell-fastest);
- filters on higher-order data (subdivide, rather than linearize);
- clip on polyhedra, polygons and polylines;
- locating points in polyhedra (no parametric coordinates; a containment
  test against the faces instead);
- a polyline topology's edges (its segments as global entities), and
  stitching contour lines into polylines.

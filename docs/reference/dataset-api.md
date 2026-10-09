# Dataset API (prototype)

`tack.data`, in the `tack-vis` package, on the `vis/dataset-api` branch.
This page describes the API as it stands. For why it is shaped this way and
how it got here, see the proposals:
[Dataset API](../design/dataset-api.md) and
[Polyhedra](../design/polyhedra.md). They are history; this page is the
current state. It is a prototype, built to get the API right rather than to
be complete: what is missing is listed with each part.

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
- The **geometry** is the field named `"shape"`, so point positions are
  data like any other.
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
| `UnstructuredTopology(types, offsets, connectivity, num_points=None)` | VTK cell types and point rows, of the linear shapes (vertex to pyramid) | shape-based |
| `StructuredTopology(point_dims)`, usually through `rectilinear_grid(x, y, z)` | grid dimensions; cells addressed by (i, j, k) | shape-based |
| `PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces, cell_face_sides, num_points=None)` | faces stored once, as point rings wound out of side 0; each cell a list of faces, with the side (u8, 0 or 1) it is on | polyhedral, 3D |
| `PolyhedralTopology.from_cell_faces(cells, num_points, positions=None, orient=False)` | each cell's own face rings, as VTK and most readers give them; copies of a face matched by point set | polyhedral, 3D |
| `PolygonalTopology(loop_offsets, loop_points, num_points=None)` | each polygon's loop of points | polyhedral, 2D |

`num_points` exists because a dataset may have points that no cell uses.

### Derived entities

- `topology.faces()` and `topology.edges()` are derived on the device by
  sorting, the first time they are asked for, and then kept.
- A face's points are wound out of its **side 0**. **Side 1** is the cell on
  the other side, or none (`-1`) on the boundary.
- Each cell knows, per local face, the face's id and which side of it the
  cell is on.
- On a polyhedral topology the faces are given, not derived, and keep their
  given numbering. A polygonal topology's faces are its edges.
- A polyhedral topology also derives each cell's distinct points
  (`cell_points()`), for point data.

**Open:** on a 2D shape-based mesh, `faces()` is empty. The polygonal
topology already treats a 2D mesh's faces as its edges, which is the
intended answer, but the shape path does not yet.

### Converting and checking

| Function | What it does |
|---|---|
| `as_polyhedra(data)` | a 3D shape-based dataset as polyhedra, carrying its fields |
| `as_polygons(data)` | a 2D shape-based dataset as polygons |
| `check_winding(data)` | the ids of the cells whose faces do not wind consistently; empty when all is well |
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

## Fields and arrays

`Field(space, values)`: the values are a `tack.field` (scalars) or a
`tack.Vector.field` (vectors), or an implicit array:

| Array | Value `k` |
|---|---|
| `CartesianProduct(x, y=(0.0,), z=(0.0,))` | the point of a rectilinear grid, from three axes; nothing stored per point |
| `ConstantArray(value, size)` | `value` |
| `CountingArray(size, start=0, step=1)` | `start + step * k` |

Every algorithm takes an implicit array unchanged; kernels read it through
the field's view. A NumPy array is refused: put it in a `tack.field` first.

**Geometry.** `DataSet(topology, geometry, fields=None, sets=None,
dtype=tack.f32)` takes the geometry as an `H1` or `L2` field of 3-vectors,
or as positions per point (any array of 3-vectors), which become an `H1`
field. It is kept as `fields["shape"]`. `data.geometry` returns it and
`data.positions()` gives the positions on the host. An `L2` geometry pulls
cells apart without changing the topology. A geometry of order 2 is curved.

## Sets

`data.sets` maps a name to an array of face ids. `alg.boundary_faces(data,
name="boundary")` makes the boundary set. Any set's name is an iteration
domain.

**Open:** cell sets, side sets given as (cell, local face), and boundary
entities from readers (Exodus side sets, MFEM boundary attributes).

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

- `domain` is `"cells"`, `"faces"`, `"edges"` or the name of a face set.
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

Every view supports `for i in view` and `entity_id(i)`.

**Cells.** These methods work on both kinds of topology, so face-based
algorithms are written once:

| Method | |
|---|---|
| `num_faces(c)` | the cell's faces (codimension-1 entities) |
| `face_id(c, k)`, `face_side(c, k)` | local face `k`'s global id, and the side the cell is on |
| `side_size(c, k)`, `side_point(c, k, j)` | local face `k`'s points, walked outward from this cell |
| `side_position(c, k, j)` | that point's position |

- Shape-based cells add: the class constants `NUM_POINTS`, `NUM_FACES`,
  `NUM_EDGES` and `DIMENSION`; `point_id(c, j)`; the geometry through
  `position(c, pc)`, `geometry_jacobian(c, pc)` and `point(c, j)`; face
  incidence (`face_orientation`, `face_corner`, `face_position`); and edges
  (`edge_id`, `edge_sign`).
- Polyhedral cells add `num_points(c)` and `point_id(c, j)` over the cell's
  distinct points, and `face_size(f)`.

**Faces.** `face_size(f)`, `point_id(f, j)`, `num_sides(f)`, and
`side_cell(f, s)` / `side_local(f, s)` give the cell on side `s` and its
local face number.

**Edges.** `point_id(e, j)`, with the lower point id first.

**Fields.**

| Field view | Methods |
|---|---|
| `H1`, `L2`, `Constant` | `value(c, pc)`, `parametric_gradient(c, pc)` at parametric coordinates; `dof(c, k)` |
| `Values` | `at(i)` |
| `SideTraces` | `trace(f, s, j)`, `value(f, s, pc)` |

## Algorithms and filters

Results are `Field`s, or `DataSet`s for filters. In the table,
"polyhedral" means `PolyhedralTopology` and "polygonal" means
`PolygonalTopology`.

| | Shape-based | Polyhedral | Polygonal |
|---|---|---|---|
| `face_geometry` (normals, areas), `face_centers`, `cell_geometry` (volume or area, centroid), `edge_lengths` | yes | yes | yes |
| `boundary_faces`, `extract_surface` | yes | yes (the surface is polygonal) | yes |
| `jump` and `upwind_flux` of cell data; `divergence`, `perot` of face values | yes | yes | yes |
| `threshold`, `extract_geometry`, `extract_cells`, `mask` | yes | yes | yes |
| `contour`, `slice`, `slice_plane` | yes | yes (López, face-based) | not yet |
| `clip` | yes | not yet | not yet |
| `extract_points`, `threshold_points`, `mask_points` | yes | yes | yes |
| `implicit_values` | yes | yes | yes |
| `cell_centers`, `values_at_centers` | yes | no | no |
| `gradients(at="cells" | "points")` of scalars or vectors, `flow_quantities` | yes | no | no |
| `to_points` of cell data, `to_cells` of point data | yes | yes (over each cell's distinct points) | yes |
| `discontinuous`, `to_points` of `L2` data | yes | no | no |
| `traces`, and `jump`/`upwind_flux` of point data | yes | no | no |
| `external_faces` | yes | yes (polygonal) | the boundary edges, as two-point polygons: there is no line topology yet |

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
row (`d(u_i)/d(x_j)` at `3i + j`). `algorithms.flow_quantities(gradient)`
gives divergence, vorticity and the Q-criterion from a 3-vector field's.

## Implicit functions

`tack.data.Plane(origin, normal)`, `Sphere(center, radius)`,
`Cylinder(center, axis, radius)` (infinite), `Box(lower, upper)`
(axis-aligned) and `Planes(origins, normals)` (the convex region below every
plane: a frustum, a box at any angle). `value(p)` is negative inside, zero on
the surface, positive outside, in VTK's and Viskores' forms -- quadrics for
the sphere and cylinder, signed distances for the plane, planes and box -- so
a slice by one lands on the same points as theirs.

Each is a `@tack.data_oriented` object whose parameters are instance values,
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

`tack.data.transforms`, in Viskores' conventions:

| Function | Result |
|---|---|
| `magnitude(f)`, `dot(a, b)`, `cross(a, b)` | a field in the same space |
| `composite(f, g, ...)` | scalar fields of one space as one vector field |
| `log_values(f, base, min_value)` | the logarithm, values below `min_value` (default the smallest normal float) taken as it |
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
| `algorithms.statistics(f)` | Viskores' descriptive statistics: n, min, max, sum, mean, central moments, sample and population variance and deviation, skewness, kurtosis |
| `algorithms.entropy(f, bins)` | Shannon entropy in bits of a `bins`-bin histogram |
| `sources.wavelet(extent, ...)` | VTK's and Viskores' wavelet (`RTData`) on a uniform grid |
| `sources.tangle(dims)` | Viskores' tangle cube (`tangle`) on [0, 1]^3 |

Tetrahedra and triangles are pieces of their cells (`carry`'s `Pieces`): cell
data goes to each, the points stay the same.

## Interoperability

| Function | |
|---|---|
| `tack.interop.vtk.vtk_to_dataset(grid, dtype=tack.f32, polyhedral=None, orient=False)` | a `vtkUnstructuredGrid` or rectilinear grid. Point data becomes `H1`, cell data `Constant`. A grid with polyhedra becomes polyhedral (`polyhedral=True` forces it); `orient` repairs inconsistent winding |
| `tack.interop.vtk.dataset_to_vtk(data)` | back to VTK, polyhedra and polygons included |
| `tack.interop.mfem.mfem_to_dataset(mesh, fields=None, dtype=tack.f32)` | an MFEM mesh, curved at order 2 too, with its grid functions |
| `tack.interop.mfem.mfem_field(data, mesh, gf, dtype=tack.f32)` | one grid function: H1 orders 1 and 2, L2 orders 0 and 1, scalars and vectors |

Polyhedral formats such as CGNS arrive through VTK's readers. The CGNS
reader's polyhedra can be inconsistently wound, so pass `orient=True`.

## Settled, and open

**Settled by the prototype:**

- continuity follows from where values live;
- fields are separate kernel arguments, so no nested templates;
- shape, order and size groups are one mechanism;
- geometry is a field;
- a field's values may be any array, explicit or implicit;
- polyhedra are a separate path that shares the data model;
- face values have an orientation where it matters.

**Open:**

- 2D faces on the shape path (above);
- sets beyond faces;
- coefficient layout (DOF-fastest or cell-fastest);
- a general `project(field, space)`, of which `to_points`, `to_cells` and
  `discontinuous` are special cases;
- whether views should keep one shared namespace;
- `"shape"` as a named field or a separate attribute;
- filters on higher-order data (subdivide, rather than linearize);
- clip on polyhedra;
- a topology of lines, for a 2D mesh's boundary and for contour lines of
  polygons.

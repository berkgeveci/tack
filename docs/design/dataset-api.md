# A common dataset API: design proposal

Status: proposal for discussion, 2026-10-08. A prototype of phases 1-3 at
linear order is in `tack.data` (section 9).
It builds on the prototype on `vis/data-model` (cell shapes, cell sets,
`for_each_shape`, six filters) and on two studies of prior art: MFEM 4.9's
mesh and finite-element spaces, and VTK's `vtkCellGrid` (master,
2026-09-25). The goal is the *shape* of the API every dataset presents --
so that discontinuous Galerkin fields, data on faces and edges, and
high-order fields fit without special cases -- not high-order cells
themselves.

## 1. What the first prototype (`vis/data-model`) gets right, and what it cannot express

The prototype separates a dataset into *topology* (a cell set), *geometry*
(point coordinates) and *data* (`point_data`, `cell_data`). Kernels
iterate one cell shape at a time (`for c in cells`) through a template
view that is both the shape and that shape's cells, so every compiled
kernel is specialized to one shape and structured cells iterate by
(i, j, k) without division. Six filters on that model match VTK.

What it cannot express:

1. **Data anywhere but points and cells.** No edges or faces as entities,
   so no edge data (fluxes, circulation), face data (normal fluxes,
   face-centered values), or data on the two sides of a face.
2. **Discontinuous fields.** A point-data field has one value per point.
   A DG field has one value per (cell, local node); today that requires
   exploding points, which is what MFEM's Catalyst "mesh" channel does
   (`num_vertices = NE x corners`).
3. **Fields with a basis.** Point data is implicitly linear and cell data
   implicitly constant. Neither order, nor node placement, nor H(curl) /
   H(div) mappings can be stated.
4. **Geometry as data.** Positions are a separate concept from fields, so
   curved geometry, or geometry discontinuous across cells (periodic
   meshes), has no place.
5. **Iteration over anything but cells.** Face kernels (DG fluxes,
   boundary integrals, external surfaces), edge kernels and
   quadrature-point kernels have no entry point.

## 2. What MFEM and vtkCellGrid teach

**MFEM** makes *where degrees of freedom live* the single source of truth.
A finite-element collection states how many DOFs each entity of each
geometry owns (`DofForGeometry`): H1 puts them on vertices, edges, faces
and interiors; L2 (DG) only in interiors; Nédélec (H(curl)) on edges and
interiors; Raviart-Thomas (H(div)) on faces and interiors; trace spaces
only on faces. Continuity is a *consequence*: DOFs on a shared entity are
shared. Edges and faces are numbered entities with CSR incidence tables
(`el_to_edge`, `el_to_face`, `face_info` with two sides, local face id and
orientation), built from vertex keys. Fields are coefficient vectors on a
space; geometry is itself a field (`Nodes`). For devices, MFEM's
vocabulary is the right one: L-vectors (one value per global DOF),
E-vectors (per-cell gathered DOFs, `ND x VDIM x NE`), face E-vectors
(`face_dofs x VDIM x {1|2} x NF`), Q-vectors (values at quadrature
points), and restriction operators that gather between them through
signed index maps.

**vtkCellGrid** makes the *cell type* the unit of dispatch. One metadata
object per cell type holds static, GPU-uploadable tables (reference
points, side connectivity, side offsets and shapes). A face or edge is a
*side*: a lightweight `(cell id, side index)` pair, so subsets (boundary
faces, external surfaces) never rewrite the parent arrays. Each attribute
states, per cell type, its function space (HGRAD, HCURL, HDIV, constant),
basis, order, and whether its DOFs are *shared* (gathered through a
connectivity) or *discontinuous* (one coefficient tuple per cell).
Algorithms are multi-pass queries -- count, prefix-sum, write by index --
dispatched to per-cell-type responders, and the GPU renderer compiles one
shader per (cell type, attribute) combination. What it lacks: data that
lives only on edges or faces with its own numbering. H(curl)/H(div)
coefficients are stored per cell, discontinuously, and an attribute
cannot be defined on a side set alone.

**For Tack:** take MFEM's model of *DOFs on entities* as the truth, which
covers H1, DG, edge, face and trace data uniformly; take vtkCellGrid's
*sides as (cell, side) pairs* and per-shape tables, which are already how
the prototype works; and take both systems' GPU idioms (gather maps,
E-vectors, count/scan/scatter, one specialization per shape and space),
expressed as Tack templates rather than runtime registries.

## 3. The model

A dataset is four layers, each of which can be absent or implicit:

```
DataSet
  topology   entities of each dimension, and their incidence
  spaces     where values live: DOF layout over the topology, plus a basis
  fields     a space and a coefficient array (geometry is one of them)
  sets       named subsets of entities (boundary faces, material blocks)
```

### 3.1 Topology: entities and incidence

Entities are the vertices (dimension 0), edges (1), faces (2) and cells
(3, or 2 for a surface mesh), each with its own index space. A cell has a
shape from `tack.data.shapes`, and its sides -- the faces, edges and
vertices of its reference element, in a fixed local order -- come from the
shape's tables.

| Incidence | Meaning | Source |
|---|---|---|
| cell -> vertex | connectivity | given (explicit) or implied (structured) |
| cell -> edge, cell -> face | global id of each local side, with orientation | derived by sorting side keys |
| face -> cells | the cells on each side of a face, with their local face index and orientation | derived with cell -> face |
| edge -> vertex, face -> vertex | the entity's own vertices | derived |

Only cell -> vertex is required. The rest is derived on demand by the
sort-based pattern `external_faces` already uses (key each local side by
its sorted vertex ids, sort, find runs) and cached on the topology, as
`point_links` is. Structured topology implies all of it arithmetically.

**Orientation** is stored where MFEM stores it: an edge's direction is
from its lower to its higher global vertex id, so each (cell, local edge)
carries a sign; a face's canonical vertex order is that of the first cell
that lists it, so each (cell, local face) carries an orientation index
into the face shape's permutation table. Fields with DOFs on edges and
faces need these to put coefficients in the right place.

**Sides.** A side is a `(cell, local side)` pair, as in vtkCellGrid. A
*face* is a global entity; a *side* is one cell's view of it. An interior
face has two sides, a boundary face one. Side sets (boundary conditions,
an external surface, a slice of selected faces) are arrays of sides, so
selecting faces never copies the volume.

### 3.2 Spaces: where values live

A space describes, for each cell shape, how many values each entity of
each dimension owns, and what basis turns them into a function:

```python
Space(
    topology,    # the mesh it lives on: a space owns the layout derived from it
    family,      # Constant, H1, L2, HCurl, HDiv, Trace(H1|L2|HCurl|HDiv), Values
    order,       # polynomial order; 0 for Constant
    variant,     # full tensor or serendipity (27 or 20 nodes on a quadratic hexahedron)
    nodes,       # node placement: GaussLobatto, GaussLegendre, Equispaced, Bernstein
    layout,      # Shared (one value per global DOF) or PerCell (one tuple per cell)
)
```

A space is constructed on its topology, as MFEM's `FiniteElementSpace` is
on a mesh, and owns the layout that follows: offsets for a PerCell
layout, and for a Shared one of order above 1 the (cell, local DOF) ->
DOF map built from the derived edges and faces, one per shape group.
Equal spaces on one topology are one object, so fields on the same space
share that layout. Components per value are the field's, not the
space's: a 3-vector field and a scalar one in H1 order 2 share one DOF
map (MFEM's `vdim`).

The DOF counts per (shape, entity dimension) follow from the family and
order, as in MFEM. That one rule expresses everything this proposal set
out to carry:

| Today's term | Space |
|---|---|
| point data | H1, order 1, Shared: one DOF per vertex |
| cell data | Constant (L2 order 0): one DOF per cell |
| linear DG field | L2, order 1, PerCell: one tuple of corner values per cell |
| high-order CG / DG | H1 or L2, order p |
| edge data, one value per edge | `Values` on edges (no basis), or HCurl order 1 (with one) |
| face data, one value per face | `Values` on faces, or HDiv order 0/1 |
| data on both sides of each face | PerSide layout of a face space (DG fluxes) |
| quadrature-point data | a `QuadratureSpace`: a rule per shape, no basis |

`Values` spaces carry one value per entity with no basis: data that is
*about* an edge or face (a flux, an ID, an error indicator), not a
function to interpolate. Interpolating spaces add a basis per shape: a
`@tack.func` that evaluates basis functions and gradients at parametric
coordinates, written once and compiled per shape and order, the way
vtkCellGrid's basis snippets are compiled for C++ and GLSL.

**Layouts** are MFEM's vectors, named for what kernels index:

- *Shared*: one array of global DOF values, plus a gather map per shape
  (cell, local DOF) -> signed global DOF, derived from the topology and
  the space's DOF counts. Continuous fields.
- *PerCell*: one array of shape `(cells of this shape, local DOFs, vdim)`
  per shape. DG fields, and any field after a gather (MFEM's E-vector).
- *PerSide*: per face, one tuple per side (MFEM's double-valued face
  E-vector). Face fluxes and jumps.

A space's layout is part of the kernel's specialization, as the cell
shape is; its DOF count per shape is a class constant, so local arrays
for coefficients have compile-time sizes.

### 3.3 Fields, and geometry as a field

A field is a space plus a coefficient array, with a name. Geometry is the
field named `"shape"` (vtkCellGrid's term): a vector field, H1 order 1 for
straight-sided cells, higher for curved ones, PerCell when geometry is
discontinuous. Rectilinear and uniform grids keep their coordinates
implicit (three axis arrays, or origin and spacing): their geometry field
evaluates procedurally and stores nothing per point, as
`RectilinearCoordinates` does in the prototype.

### 3.4 Sets

Named subsets of entities -- arrays of cell ids, face ids, or sides --
carry boundary conditions, material regions and filter outputs.
Threshold's output and external faces become sets over the input instead
of new datasets, where a consumer wants that.

## 4. Kernels: what a kernel sees

Kernels stay what they are in the prototype: specialized per shape,
iterating a view. What generalizes is *what* is iterated and *how fields
are read*.

**Iteration domains.** `tack.data.for_each(kernel, data, domain, *args)`
launches once per shape present in the domain:

| Domain | One iteration is | Launches |
|---|---|---|
| `cells` | a cell | per cell shape |
| `faces` | a face, with its one or two sides | per face shape (triangle, quad) |
| `edges` | an edge | one |
| `vertices` | a vertex | one |
| a side set | a side: (cell, local face) | per (cell shape, face shape) |
| `quadrature` | a (cell, point) pair | per cell shape |

The view for each domain is a template class combining, as now, the
shape with the entity kind's accessors. A face view offers `side(f, k)`
-> (cell, local face, orientation) for k = 0, 1 and `has_second_side(f)`;
a cell view adds `edge_id(c, e)`, `edge_sign(c, e)`, `face_id(c, f)`,
`face_orientation(c, f)` next to the existing `point_id(c, j)`.

**Field views.** A field is passed as its own template argument, built
per launch for the shape being iterated, so its methods compile to that
shape's basis and that space's layout:

```python
@tack.kernel
def flux(faces, u, out):
    for f in faces:
        c0, s0, o0 = faces.side(f, 0)
        u0 = u.value_on_side(c0, s0, o0, faces.center())   # the face's center, from side 0
        ...
```

- `u.dofs(c, local)` reads coefficients (gathering through the signed
  map for Shared, directly for PerCell),
- `u.value(c, pc)` and `u.gradient(c, pc)` evaluate at parametric
  coordinates through the basis,
- `u.value_on_side(c, side, orientation, face_pc)` maps a face point into
  the cell first.

H(curl)/H(div) values apply the Piola maps with the geometry field's
Jacobian, which the field view reaches through the cell view.

**One gap in the language.** A template cannot hold another template or
be passed to a device function, so a dataset cannot be one kernel
argument carrying its topology, geometry and fields. Fields are separate
arguments, which reads well and specializes cleanly. If bundling turns
out to matter -- `data.fields.u` inside a kernel -- nested templates are
a language feature to add first; nothing below depends on it.

## 5. Host API

```python
data = tack.data.DataSet(topology, geometry=positions_or_field)
data.fields["T"] = tack.data.Field(H1(order=2), coefficients)
data.fields["flux"] = tack.data.Field(Values(on="faces"), per_face)
data.sets["inlet"] = tack.data.SideSet(cells, local_faces)

data.topology.faces()        # derive faces and face -> cells, cached
data.fields["T"].at_vertices()   # resample to a linear point field
```

`point_data` and `cell_data` remain as views over fields in the two
classic spaces, so existing filters keep working while they are moved to
the general form. Interop maps each source onto the model directly:

| Source | Topology | Fields |
|---|---|---|
| vtkUnstructuredGrid | explicit cells | point data -> H1 1 Shared; cell data -> Constant |
| vtkRectilinearGrid, vtkImageData | structured, implicit | as above; geometry implicit |
| vtkCellGrid | per cell type | each attribute's CellTypeInfo -> a space per shape (shared -> Shared, discontinuous -> PerCell) |
| MFEM Mesh + GridFunction | entities and tables as given | FE collection + order + basis type -> family, order, nodes; GridFunction -> Shared coefficients with MFEM's signed element -> DOF table, or an E-vector as PerCell |

MFEM data arrives most faithfully as Shared coefficients plus MFEM's own
gather map: Tack does not need to rederive MFEM's DOF numbering, only to
accept a (cell, local DOF) -> signed global DOF table per shape.

## 6. Filters in the general form

Every filter becomes "per shape, per domain, with field views":

- **Cell centers** evaluate the geometry field at the parametric center.
- **Point/cell averaging** become *projections between spaces*: Constant
  <-> H1 1 is today's pair; L2 p -> H1 1 at vertices is how DG data is
  shown with continuous coloring.
- **Contour** evaluates the field at the cell's vertices for linear
  spaces; for order > 1 it subdivides each cell (or uses Bézier hulls, as
  vtkCellGrid's ChangeBasis prepares for) and contours the pieces. DG
  jumps are kept: each cell contours its own polynomial, and points merge
  only within a cell (built at order 1; section 9).
- **External faces** become "the faces with one side" from the derived
  face topology, as a side set over the input.
- **Threshold** becomes a cell set over the input, or a new dataset.

## 7. GPU considerations that shape the design

- **Closed enums, compile-time widths.** Shape, family, layout, node type
  and order are class constants of the views: the specialization key. DOF
  counts per shape are constants, so coefficient scratch is a fixed-size
  local array. vtkCellGrid's free-form tokens and runtime `std::function`
  registries become host-side choice of a specialized kernel.
- **Per-shape launches** throughout, as now; mixed meshes cost a launch
  per shape, never a per-cell branch.
- **Incidence by sorting**, not hashing: side keys, `argsort`,
  `unique`/run offsets -- the pattern `external_faces`, `point_links` and
  `contour`'s point merging already use.
- **Count, scan, scatter** for every variable-size output.
- **Coefficient layout.** MFEM's E-vector puts a cell's DOFs together
  (`ND x VDIM x NE`, DOF fastest), so one thread reads one contiguous
  run; a cell-fastest layout coalesces across threads instead. This is a
  measured decision per backend, and the field view hides it.

## 8. A phased path

1. **Entities.** Derive edges and faces (ids, orientations, face -> cells)
   from any cell set, cache them, and add `for_each` over faces, edges and
   side sets. Re-express `external_faces` on it. *No new data types.*
2. **Data on entities.** `Values` spaces on vertices, edges, faces, cells
   and sides; `point_data`/`cell_data` as views over them. Face and edge
   data round-trip from MFEM and through vtkCellGrid.
3. **Spaces with bases, low order.** Constant, H1 1, L2 1 (linear DG), with
   field views evaluating through `tack.data.shapes`' existing shape
   functions. Geometry becomes the `"shape"` field. Contour and the
   averaging filters move to field views; DG fields contour with their
   jumps intact.
4. **High order, H(curl), H(div),** quadrature spaces: new bases per shape
   and order, Piola maps, subdivision for contouring. This is where
   high-order cells start; the API should not change.

## 9. The prototype

`packages/tack-vis/src/tack/data/` implements phases 1 and 2 and the
linear part of phase 3, for an unstructured grid of any of the linear
shapes and a rectilinear grid:

| Module | What it holds |
|---|---|
| `topology.py` | `UnstructuredTopology`, `StructuredTopology`; cells as one `DomainGroup` per shape; `faces()` and `edges()` derived by sorting and cached |
| `views.py` | the template mixins: entity kinds (cells, structured cells, faces, edges), geometry (`position`, `geometry_jacobian`, read from the geometry field), cell incidence, one per space, one per storage (`get(k)`), and how a point's value is addressed |
| `arrays.py` | implicit arrays a field's values may be: `CartesianProduct` (a rectilinear grid's points), `ConstantArray`, `CountingArray`; helpers for any array |
| `spaces.py` | spaces on a topology, each owning its layout: `H1(data)`, `L2(data)` (holds its offsets), `Constant(data)`, `Values(data, on)`; one object per (kind, parameters, topology) |
| `dataset.py` | `Field` (a space and its values), `DataSet` (geometry is its `"shape"` field), `for_each` |
| `algorithms.py` | `cell_centers`, `values_at_centers`, `gradients`, `discontinuous`, `face_geometry`, `edge_lengths`, `boundary_faces`, `extract_surface`, `traces`, `jump`, `upwind_flux`, `divergence`, `to_points`, `to_cells` |
| `filters.py` | `contour`, `slice_plane`, `threshold`, `external_faces`, ported from `vis/data-model` onto fields and spaces |
| `interop/mfem.py` | `mfem_to_dataset`, `mfem_field`: MFEM meshes (curved of order 2 too) and grid functions (H1 orders 1 and 2, L2 orders 0 and 1, vectors), through PyMFEM |
| `interop/vtk.py` | `vtk_to_dataset` / `dataset_to_vtk`: point data as `H1`, cell data as `Constant` |

`packages/tack-vis/examples/42_dataset_api_tour.py` runs every algorithm
on both grids and writes VTK files, including the DG field with one copy
of each point per cell. `tests/test_dataset_api.py` checks faces, edges,
sides and orientations against VTK's cells for all five solid shapes, and
the algorithms against vtkCellCenters, vtkCellDataToPointData and
vtkGeometryFilter.

A kernel over faces, with a field on cells and one on faces:

```python
@tack.kernel
def _jump(faces, values, out):
    for f in faces:                                   # one launch per face shape
        c0 = faces.side_cell(f, 0)
        c1 = faces.side_cell(f, 1)                    # -1 on the boundary
        out[faces.entity_id(f)] = values[c1] - values[c0] if c1 >= 0 else values[c0] * 0.0

for_each(_jump, data, "faces", cell_values, out)      # or a face set: "boundary"
```

and over cells, through their faces:

```python
for c in cells:
    for f in range(cells.NUM_FACES):
        sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
        total += sign * flux[cells.face_id(c, f)]
```

What building it showed:

- **Field views as separate arguments work.** `for_each` turns each
  `Field` into its view for the group being launched, so `u.value(c, pc)`
  compiles to that shape's basis and that space's layout, with no
  nested templates. A field whose values do not live where the loop is
  (a face field in a cell loop) is refused on the host.
- **The groups are the layout.** Cell -> face and cell -> edge incidence is
  stored per (cell, local side) in group order, so a topology must keep
  its groups, and every view of the same cells must come from them.
- **An unstructured topology's L2 layout is its connectivity layout**:
  cell `c`'s corner values start at `offsets[c]`. Showing a DG field in
  VTK is then a gather (the tour's `explode_dg`).
- **A face loop cannot evaluate a cell's basis** when the two sides have
  different shapes (a hexahedron against a pyramid): one kernel cannot be
  specialized to both. The way through is MFEM's, and it is built: a
  *cell* loop over (cell, local face) evaluates each side's trace into a
  PerSide field, and the face loop reads that.
  - *Face orientation* is derived with the faces: per (cell, local face),
    `side_orientation = 2 * r + reflected`, where the cell's first point of
    the face is point `r` of the face's row and `reflected` is 1 when the
    cell goes round the face the other way. On a consistently oriented
    mesh that is exactly side 1. Cell views offer `face_orientation(c,
    f)`, `face_position(c, f, k)` (where the cell's point `k` of the face
    is in the face's row) and `face_corner(f, k)` (which corner it is; a
    voxel's pixel faces go round as quads). This is the index higher-order
    face DOFs will need, as MFEM's `Elem2Inf % 64`.
  - *`SideTraces(data)`* is the PerSide space: per (face, side, point),
    `(face * 2 + side) * 4 + point`, MFEM's double-valued face E-vector at
    order 1. `traces(data, field)` fills one from any field with a basis
    -- `H1`, `L2` (an order per cell too), `Constant` -- each cell writing
    its own side at the face's points in the face's order, so the sides
    line up point by point. A face view reads it with `trace(f, s, j)` and
    `value(f, s, pc)`, through the face shape's functions.
  - *Uses:* `jump` now takes any such field, from its traces at the face
    center, so a DG field's jumps are its own and a continuous field's are
    zero; `upwind_flux` is a DG advection flux, `(v . n) * area * u` with
    `u` from the side the flow leaves, whose `divergence` is each cell's
    net outflow.
- **Geometry is a field**, `fields["shape"]`: `H1` positions per point
  (explicit, or `RectilinearCoordinates`, which store only the axes), or
  `L2` positions per cell corner. Topology stays the corners, so an `L2`
  geometry -- cells pulled apart -- still has every face and neighbour.
  Cell views read the geometry through `position(c, pc)` and
  `geometry_jacobian(c, pc)`; `gradients` combines that Jacobian with a
  field's own `parametric_gradient`, so geometry and field each go
  through their own space. At order 1 the geometry's values are the
  corners, interpolated by the shape's functions; a higher-order geometry
  adds its own basis behind the same two methods. An `L2` geometry's faces
  are where their side 0 puts them: its traces, kept on the dataset, give
  face views their positions, so `face_geometry` works on it. Its edges
  are refused: the cells around an edge each have their own corners, and
  an edge has no sides to choose from.
- **A field's values are any array**, explicit or implicit, as Viskores'
  `ArrayHandle` storages are. A view is (space + storage): the space's
  methods read through the storage's `get(k)`, so a field can be a
  `CartesianProduct` of three axes, a `ConstantArray` or a
  `CountingArray` and every algorithm takes it unchanged. The rectilinear
  grid's geometry is just an `H1` field over a `CartesianProduct`; it lost
  its special-case mixins. For a structured grid's cells over a storage
  with `get_ijk`, the host picks an addressing mixin that reads each
  corner by (i, j, k), so no flat point id is ever split; elsewhere
  (faces, edges, an unstructured topology over the same points) the flat
  id is split as before.
- **Spaces own their layout.** A space is built on a topology --
  `H1(data)`, `L2(data, order=1)`, `Values(data, "faces")` -- and is
  interned there: equal spaces are one object, so fields on it share its
  layout, and a field is just a space and its values. `L2` holds its
  offsets (for an unstructured topology, the connectivity's own); every
  space knows its size, so a field of the wrong length is refused, and
  `for_each` refuses a field on another topology. An unstructured
  topology takes `num_points`, since a dataset may have points no cell
  uses, and `H1`'s size follows it. Order above 1 is accepted as a
  parameter and refused as not yet built; it is where a space's own DOF
  map will go.
- **Launch groups come from the fields' spaces** (section 11).
  `L2(data, order=orders)` takes an order per cell, 0 or 1 here, which is
  enough for a real p-adaptive DG field; its offsets come from a scan of
  per-cell value counts. A space that varies gives each cell a key, and
  `for_each` (through `DataSet.launch_groups`) splits each shape group by
  the combined keys of all such spaces among its arguments, on the device:
  a key per cell, a stable sort, runs. Each subgroup is launched with the
  field-view mixin for its keys (`L2` order 0 or 1), so the kernel is
  specialized as for a uniform space. Subgroups are slices of one gathered
  copy of the group's rows and ids, sorted by key, and keep each cell's
  position in its group, so the group's face and edge incidence still
  applies; they are cached on the topology. Algorithms that lay out one
  entry per (cell, corner), such as `to_points`, place a subgroup's cells
  by that position.
- **MFEM is the reference for real data and for order 2.** PyMFEM (`pip
  install mfem`) brings MFEM into the test process: meshes are built in it
  (Cartesian meshes of every element type, a mixed one, curved ones by
  `SetCurvature` and `Transform`), and its own counts, face -> element
  records, Jacobians, `GetValue` and `GetGradient` check the derived
  topology, orientation and every mapped space. `examples/43_mfem_poisson.py`
  solves MFEM's first example in H1 of order 2 on a curved mesh -- MFEM's
  `fichera-q2.mesh`, refined, has 4401 DOFs, the same count here -- and
  matches MFEM's values and gradients to float32 rounding before running
  the filters on it.
  - *Order 2 is built*: quadratic Lagrange bases on tetrahedra (10 nodes),
    hexahedra and voxels (27) and wedges (18), each node's function a
    product of 1D quadratic factors on its coordinates (barycentric ones
    on simplices), its nodes from the shape's own corner, edge and face
    tables. `H1(data, order=2)` owns the numbering -- points, then one per
    edge, one per quad face, one per hexahedron -- built from the derived
    edges and faces, and its views read a cell's values through its edge
    and face ids; no orientation is needed at order 2, each edge and face
    holding one value. A curved geometry is an order-2 H1 field whose
    positions and Jacobians go through the quadratic functions, so fields
    of either order sit on geometry of either order. MFEM's order-2 H1
    nodes are these same points, so its fields and curved meshes come in
    exactly, evaluated at the nodes. Pyramids, whose MFEM basis is a
    different (rational) one, are refused.
  - *The filters linearize*: they read each cell's corners, so contour and
    slice cut a curved, quadratic cell by its corner values and positions,
    and threshold and external faces keep an order-2 field's values at the
    points. An order-2 field is read cell by cell; a face loop takes its
    traces.
  - *MFEM's prisms are VTK's wedges, corner for corner.* MFEM's VTK writer
    swaps a prism's corners 1 and 2, and 4 and 5 (`PrismMap` in
    mesh/vtk.cpp); taken here, that turned every MFEM-positive prism
    inside out, against vtkWedge.h's convention and VTK's own wedges.
    Worth raising with MFEM.
- **The filters port cleanly, and gain from the spaces.** `contour`,
  `slice_plane`, `threshold` and `external_faces` from `vis/data-model`
  read fields through their views, and match VTK as before
  (vtkContourGrid for every case of every 3D shape and for real meshes,
  vtkCutter, vtkThreshold, vtkGeometryFilter). What is new: `contour`
  takes any field with a basis, evaluated at each cell's corners
  (`corner_value`, so the filters do not assume values sit at corners).
  A continuous field on a continuous geometry merges points by global
  edge, watertight; a DG field, or any field on an L2 geometry, merges
  them only within each cell, so the surface keeps the data's jumps.
  `slice_plane` computes its distance in the geometry's own space, so an
  L2 geometry's cells each slice their own corners. `threshold` tests
  corner or cell values through the field's view (an order per cell
  included) and carries every field it can -- points and cells gathered,
  L2 fields and an L2 geometry block by block with their orders.
  `external_faces` is the boundary face set through `extract_surface`,
  which now carries cell and point fields too. With `to_cells` -- any field
  with a basis averaged over each cell's corners, point data as
  vtkPointDataToCellData -- beside `to_points`, `vis/data-model` has
  nothing this branch lacks.
- **The mixins share one namespace.** A `ConstantArray` kept its number in
  `value`, which hid the spaces' `value(i, pc)` method; the kernel failed
  to compile with a message far from the cause. Building a view now
  refuses any attribute named like something the view class defines.
- **Not yet.** Faces and edges:
  - *Traces at order 1 only.* `SideTraces` holds a value per face point
    (at most four). Higher order needs more points per face -- a face
    quadrature rule or the face's own DOFs, ordered through
    `face_position` -- and the space would grow a per-face count.
  - *Faces of 3D cells only.* A 2D mesh's `faces()` is empty, though its
    cells' sides -- the entities DG fluxes and external boundaries need --
    are its edges. Whether `faces()` should mean the codimension-1
    entities (edges in 2D) or stay dimensional, with sides offered
    separately, is open.
  - *Face rows hold at most 4 points* (triangles and quads), enough for the
    linear shapes; polygon faces belong to the polyhedral path
    (section 10).
  - *No edge -> cells.* A face has at most two cells, so its record is
    fixed-size; an edge has any number, so that direction would be a CSR
    list. Nothing needs it yet.
  - *Structured faces and edges are derived by the general sort.* Section
    3.1 says a structured topology implies them arithmetically (face and
    edge ids from (i, j, k) and a direction); the prototype sorts keys as
    for an unstructured mesh, which is correct but stores and sorts what
    could be computed.
  - *A face set is re-gathered on every `for_each`*: `Faces.groups(subset)`
    splits the set by shape each call, where the full face list is
    cached.
  - *Boundary entities in the input* (Exodus side sets, MFEM's boundary
    elements) are not mapped onto derived faces; sets come only from
    algorithms such as `boundary_faces`.

  Beyond faces and edges: quadrature spaces, sets of anything but face
  ids, order above two (and L2 or quadratic face traces above one),
  H(curl) and H(div), and filters that cut curved cells by their
  quadratic functions.

## 10. Polyhedra: a separate path

Refined into a full proposal in [Polyhedral meshes](polyhedra.md).

Decided in principle (2026-10-08); not built. Build it when there is a
concrete polyhedral dataset and algorithm to aim at.

Everything the shape-based design rests on -- compile-time counts
(`NUM_POINTS`, `NUM_FACES`, edge and face tables), reference elements,
parametric coordinates, shape-function bases, DOF counts per entity of a
reference element -- is absent for polyhedra. Folding them into the same
views would leak a "no reference element" branch into every space and
every algorithm that touches a basis. MFEM and vtkCellGrid leave polyhedra
out, as Viskores does as far as we know (it has polygons only); VTK
supports them as a separate, slower path behind its common cell API. And
the use is narrower and different in kind: mostly finite-volume CFD
(OpenFOAM, Fluent, STAR-CCM+), with cell-centered and face data, whose
natural operations are face-based or work on sub-tetrahedra.

So polyhedra get **their own topology and their own algorithms**, and the
shape-based views keep their compile-time constants. Turning those
constants into methods so one kernel source could serve both is not
needed.

**The topology.** Unlike VTK, where only the polyhedra of a mixed mesh
carry cell -> face data (`SetPolyhedralCells`), a polyhedral topology gives
*every* cell, hexahedra included, by its faces; it is not mixed with
shape-based cells.

| | Stored | Derived on demand |
|---|---|---|
| faces | polygons: face -> points (CSR), ordered so the order defines the normal | |
| cells | cell -> faces (CSR), with a bit per entry: does the face's normal point out of this cell | |
| face -> cells | | by inverting cell -> face: the two-sided face record of section 3.1 |
| edges | | from the faces' point pairs, by the usual sort |
| cell -> points | | the unique points of a cell's faces, for point data |

This is the data the shape-based path's derived faces already hold
(`side_face` and `side_slot`), with offsets instead of a fixed stride, and
faces given rather than derived. It is also what polyhedral formats store:
Exodus NFACED/NSIDED, and OpenFOAM, whose owner and neighbour are side 0
and side 1, the normal pointing out of the owner.

**Shared with the shape-based path:** arrays and `Field`; the spaces with
no basis (`Constant`, `Values` on points, edges, faces and cells -- nearly
all finite-volume data); sets; `DataSet`; VTK interop; the `for_each`
machinery; and the face conventions. A cell or face field means the same
on either kind of mesh.

**Separate:**

- *Spaces with a basis*, if ever needed. H1 point data has no natural
  interpolant inside a polyhedron: generalized barycentric coordinates
  (mean value, as `vtkPolyhedron` uses; Wachspress for convex cells) are
  costly and need all of a cell's faces; piecewise linear on a
  sub-tetrahedralization is cheap. DG on polyhedra (and HHO, VEM) uses
  polynomials in physical coordinates about the cell center, whose DOF
  count depends only on the order, so those kernels stay fixed-size.
- *Geometry* is H1 order 1, point positions. Centers and volumes come from
  the face decomposition, gradients from the divergence theorem over faces
  (Green-Gauss) rather than a Jacobian.
- *Algorithms.* Some exist twice -- external faces, threshold,
  cell-to-point, contour -- since the polyhedral versions work
  differently. Those needing a reference element (contour, slice, probe,
  point location) iterate each cell's sub-tetrahedra in the kernel (cell
  center, face centers, each face triangulated about its center) without
  storing them, reusing the `Tetra` shape's tables.
- *Runtime counts.* Loops over a cell's faces and a face's points take
  runtime bounds. Algorithms stream rather than gather a cell into a local
  array; where they must gather, the host picks a padded cap (8, 16, 32,
  64) from the largest cell, and cells are grouped by size bucket as the
  shape-based path groups them by shape, which also keeps GPU threads from
  waiting on the largest cell in a warp.

**Bridges.** A shape-based mesh always converts to the polyhedral form,
every cell becoming its faces, so the polyhedral algorithms are a slow
fallback for any mesh. The reverse goes by tetrahedralizing, or by
recognising cells that are really hexahedra, tetrahedra, wedges or
pyramids. Polygons in 2D are the same story one dimension down, with edges
in the faces' role.

Open: which algorithms matter on polyhedra (visualization only, or solver
data too); whether cell -> faces (Exodus, VTK) or face -> owner/neighbour
(OpenFOAM) is primary -- leaning cell -> faces; and where the first
polyhedral dataset comes from.

## 11. Variable order (p-adaptivity)

Decided 2026-10-08. The grouping is built, with `L2` of orders 0 and 1
per cell (section 9); higher orders wait for their bases. A p-adaptive
field has a
polynomial order per cell (and possibly per direction). Unlike a
polyhedron, every cell still has a reference element and a fixed DOF count
*for its order*, so order is a second grouping key after shape.

**Grouping.** Launches are specialized per (shape, order): `NUM_DOFS` and
the basis are compile-time constants again, and kernel source does not
change. The cost is a variant per (shape, order) present, a handful for
orders 1-8. The consequence for the API: groups depend on the fields as
well as the topology. `for_each` iterates each shape group split by the
orders of the spaces among its arguments -- with one variable-order field,
that field's partition; with several whose orders differ, their common
refinement. Subgroups are not contiguous, so their views run over a list
of cell ids, as face sets do (or cells are renumbered so each subgroup is
contiguous).

**Storage: per cell, as DG is.** Each cell holds its own coefficients for
its own order, from `offsets[c]`, which a scan of DOF counts per (shape,
order) makes -- the CSR layout `L2` already has. Evaluation never needs
anything else. A continuous variable-order field from a solver is gathered
into this form once on the way in (MFEM's E-vector).

**The space keeps its family.** "H1, stored per cell" and "L2, stored per
cell" use the same evaluation kernels but are different data: a
continuous field's cells agree across shared faces (exactly, under MFEM's
minimum rule). Algorithms check the family where it matters:

- contour and slice output from a continuous field can be stitched
  watertight by merging points on shared edges and faces; a DG field's
  cut is genuinely discontinuous there;
- projecting a continuous field to points is exact at shared nodes; for DG
  it is a real smoothing;
- a continuous field exports to VTK's shared-point Lagrange cells or to an
  MFEM continuous grid function; a DG field needs points duplicated per
  cell.

**The space** carries the per-cell orders: `L2(data, order=orders)` or
`H1(data, order=orders)`, `orders` a per-cell integer array (VTK's
`HigherOrderDegrees` maps onto it, a tuple per cell for anisotropic
orders). It owns the offsets and the partition of each shape group by
order. A space with per-cell orders is interned by the identity of that
array, since an array is not a hashable parameter.

**Deferred: a shared variable-order layout.** Storing values on shared
entities once needs a rule for an entity between cells of different
orders. MFEM's (`FiniteElementSpace::SetElementOrder`) is the minimum
rule: an edge or face represents every order its cells need
(`CalcEdgeFaceVarOrders`), and the higher-order DOFs are constrained to
interpolate the lowest order's (`VariableOrderMinimumRule`), with the
constraint machinery of non-conforming refinement. VTK's per-cell-degree
Lagrange cells share points instead, so where orders differ the nodes on a
shared edge do not line up and cells are evaluated on their own. The
shared layout matters for solving or for memory: per-cell storage repeats
shared values in every cell, about 8 times the point data at order 1 on
hexahedra but about twice at order 4, where interiors dominate. Add it if
memory demands, starting from a mostly linear mesh with a few high-order
cells.

**Fallback: runtime order.** Compiling once for the highest order present,
with loops to each cell's own DOF count and local arrays of the maximum
size, avoids regrouping but wastes work on low-order cells and makes GPU
threads in a warp wait on the highest. Tensor-product bases make a runtime
order practical (a 1D Lagrange basis is loops). Keep it for orders spread
widely (say 1 to 12); group by order otherwise.

## 12. Questions to settle

1. **Truth for continuity.** This proposal follows MFEM -- continuity
   follows from which entities own DOFs -- and treats vtkCellGrid's
   shared/discontinuous as a layout choice. Agreed?
2. **Global edge and face ids**: always derived, or only when a field needs
   them? Deriving costs two sorts per entity kind; caching makes it once.
3. **`Values` spaces** (data about entities, no basis) as a family of their
   own, or as Constant on that entity dimension?
4. **Coefficient layout**: DOF-fastest like MFEM's E-vectors, or
   cell-fastest for coalescing -- or per backend behind the view?
5. **Nested templates**: worth adding to the language, so a dataset can be
   one kernel argument, or are separate field arguments the API we want?
6. **Face orientation conventions**: MFEM's (first cell's local order) or
   VTK's? They matter for face and edge DOFs and for interop in both
   directions.

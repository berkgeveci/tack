# Polyhedral meshes: design proposal

Status: proposal, 2026-10-08, on branch `vis/polyhedra`. Questions in
section 7 decided as recommended; phases 1 to 4 are built (sections 9
to 12). It refines section 10 of
[the dataset API design](dataset-api.md) -- polyhedra get their own
topology and algorithms, sharing everything above that -- with what three
studies found:

- **VTK master** (2026-09/10), including Jeff Lee's 2026 rewrite of
  `vtkPolyhedron`'s contour, clip, inside test and centroid on López's
  polygon-tracing algorithm (`vtkPolyhedronContour`), and his design note
  `PolyhedronAlgorithms.md`, whose last section proposes a face-based,
  GPU-ready reformulation;
- **`vtkExplicitUnstructuredGrid`** (VTK branch `explicit-ugrid`): faces
  stored once, cells as lists of faces with an orientation per reference,
  face data first-class, built for C-grid ocean and atmosphere models
  (MPAS, ICON) whose prognostic velocity lives on faces;
- **the half-face face map** (VTK branches `half-face-ugrid` and
  `unstructured-grid-face-data`): face data on an ordinary unstructured
  grid, faces derived from cells.

The two VTK designs correspond to Tack's two paths. The half-face map is
what Tack's shape-based topologies already have -- faces derived from
cells, `sides` per face, face data as `Values(data, "faces")`. The
explicit unstructured grid is the polyhedral topology this document
designs.

The aim, as for the rest of `tack.data`, is the API: what a polyhedral
topology is, what kernels see, and how much of the shape-based path's
API carries over. Algorithms are sketched only as far as they test that.

## 1. What VTK does today, and what to keep

**Storage.** `vtkUnstructuredGrid::SetPolyhedralCells(types, cells,
faceLocations, faces)`:

- `Faces` is a CSR of polygons in global point ids;
- `FaceLocations` is a CSR from each cell to face ids in `Faces`, empty
  for non-polyhedral cells;
- a polyhedron's own connectivity is its unique points.

Faces must be wound counter-clockwise seen from outside (a breaking change
in 2026). The layout allows sharing a face, but with no orientation bit an
interior face can be outward for only one of its cells. So in practice
every producer duplicates faces per cell: readers, `vtkExtractCells`, clip
and the Conduit importer. vtkHDF maps one-to-one
(`FaceConnectivity`/`FaceOffsets`, `PolyhedronToFaces`/`PolyhedronOffsets`).

**Per-cell processing.** `GetCell` deep-copies a cell's faces, builds a
`std::map` from global to local ids and lazily an edge table. Contour and
clip remap every cell into a local face stream through a hash map, with
`thread_local` scratch that grows to the largest cell. That is fine on a
CPU and is exactly what a GPU cannot do.

**López's algorithm** (`vtkPolyhedronContour`) is purely combinatorial:

1. Classify points against the isovalue.
2. Find crossings on face edges in each face's stored order: an
   outside→inside step marks the crossing's *key face*.
3. Trace: a crossing's successor is the next crossing on its key face.

With consistent winding every crossing has one successor and one
predecessor, so the crossings form cycles: one iso-polygon each,
disconnected ones separately. It works unchanged for non-convex and
non-planar faces and cells, and the saddle rule on a face with four
crossings is the same for both cells sharing it, so the surface is
watertight. Its scratch is bounded by the cell's size: iso-vertices at
most E, crossings per face at most the face's size, contour triangles
nIso − 2·polygons. Clip has a matched `CountClip`/`EmitClip` pair, which
must stay exactly in step.

**The face-based reformulation** (PolyhedronAlgorithms.md, "Future Work")
removes all cell-level bookkeeping:

- Phase 1 runs over (face, cell) pairs. It walks each face forward for its
  owner and reversed for its neighbour, and emits each key crossing paired
  with its neighbouring crossing as one edge of the cell's polygon graph.
- Phase 2 runs over cells, following each cell's degree-2 graph around
  its cycles.

It needs no local index remap, works on global ids throughout, and
matches the original "bit for bit" on 300 cases. Lee's suggested native
storage is the CFD one: faces stored once, wound for their owner, with
owner and neighbour cell ids. That is OpenFOAM's
`faces`/`owner`/`neighbour`, and Fluent's.

**`vtkExplicitUnstructuredGrid`** stores `CellFaceConnectivity` (cell →
faces), `FacePointConnectivity` (face → points), and `FaceOrientations`
(±1 per reference: the stored face runs the way this cell sees it). It
then *ignores* the orientation for polyhedra: its Perot reconstruction
recovers outward from geometry, `sign(S_f · (r_f − r_c))`. The ordering
notes measured that test failing on 15.7% of thin ocean cells.

**Kept here:**

- faces stored once;
- an authoritative per-reference orientation;
- López on the face-based formulation;
- count → scan → emit;
- incidence by sorting;
- face numbering as data when the producer authored it.

**Not kept:** duplicated faces, per-cell hash maps, geometric orientation
tests, and `vtkExplicitUnstructuredGrid`'s contract that typed cells list
their faces in a canonical, rotated order. Here every cell is a
polyhedron.

## 2. The topology

```python
td.PolyhedralTopology(
    face_offsets, face_points,      # face -> points (CSR), in the order that winds
                                    #   the face out of its first cell
    cell_offsets, cell_faces,       # cell -> faces (CSR)
    cell_face_sides,                # u8 per cell -> face entry: 0 if the face is wound
                                    #   out of this cell (its side 0), 1 if into it
    num_points=None,
)
```

**The orientation bit is the side.** It is the shape-based path's
`side_slot` under the same name and meaning:

- side 0 is the cell the stored winding points out of (OpenFOAM's owner);
- side 1 is its neighbour, which walks the face backwards.

A face with one side is on the boundary. A face referenced by more than
two cells, or twice by the same side, is refused, as the shape path
refuses three cells on a face. (VTK's half-face map allows non-manifold
faces; nothing here needs them yet.)

**Derived on demand and cached**, by the sorts the shape path already
uses:

| Derived | How | For |
|---|---|---|
| `faces().sides` | counting sort of cell → face entries by face | `[cell0, local0, cell1, local1]` per face, the same record as the shape path; boundary faces |
| cell → points | each cell's face points, sorted and made unique per cell | point fields, `to_cells`/`to_points`, export to VTK (whose connectivity is exactly this) |
| `edges()` | each face's consecutive point pairs as `(low, high)` u64 keys, radix-sorted into runs | edge fields; iso-vertex identity (section 4) |
| face → edges, edge sign | from the same sort, per (face, local edge) | walking a face's edges; edge-attached data |

**Numbering is data when authored.** Face ids come from the input and are
never re-derived. An MPAS edge id *is* the face id, so face data lines up
with the file. Derived numberings -- edges, and faces for inputs that do
not have them -- are deterministic. VTK's half-face map orders faces by
minimum point id; here the radix sort orders by `(low, high)` key.

**Orientation is established by readers, not geometry.** Formats either
carry it or let it be derived combinatorially:

| Format | Orientation |
|---|---|
| OpenFOAM, Fluent, CGNS NGON with ParentElements, CONVERGE | owner/neighbour: side 0 and side 1 directly |
| CGNS NFACE, MPAS (`edgeSignOnCell`), ICON (`orientation_of_normal`) | a sign per cell → face reference |
| VTK polyhedra (faces duplicated, each outward for its cell) | match the two copies, which run in opposite directions; the first copy found is side 0 |

`orient(topology)` repairs a consistent-up-to-sign input
combinatorially: within each cell, every edge must be walked once in each
direction (VTK's uneven-coedge test, and its CONVERGE fix). A geometric
fallback exists only on explicit request. `check_winding(data)` is the
diagnostic, as VTK's half-face branch has.

**Conversions:**

- *Any shape-based dataset → polyhedral.* `faces()` already holds every
  face in side 0's order with its sides, so the polyhedral topology is
  `rows` → `face_points` and `side_face`/`side_slot` → `cell_faces` and
  `cell_face_sides`. No reordering is needed.
- *Polyhedral → VTK.* `SetPolyhedralCells` with each face copied per cell,
  reversed for side 1, since VTK's layout has no orientation bit.
- *VTK → polyhedral.* Match the duplicated faces by sorted point keys.

## 3. What kernels see

**No shape, no parametric coordinates.** A polyhedral cell view has no
`NUM_POINTS`, no shape functions and no reference element. Its counts are
runtime:

```python
for c in cells:                               # one launch per size bucket (below)
    for k in range(cells.num_faces(c)):
        f = cells.face_id(c, k)
        side = cells.face_side(c, k)          # 0: wound out of c; 1: into c
        for j in range(cells.face_size(f)):
            p = cells.side_point(c, k, j)     # point j of face k in c's outward order:
                                              #   forward on side 0, reversed on side 1
```

`side_point` is the one abstraction every polyhedral kernel stands on.
However a face is stored, a cell walks it outward. That is the property
López, volumes and solid angles all need, and the one the face-based
formulation exploits by walking a face once per side.

**Faces look the same on both paths.** A polygon face view offers what
the shape path's face view does:

- `entity_id(f)`, `num_sides(f)`, `side_cell(f, k)`, `side_local(f, k)`;
- `point_id(f, j)` and `point(f, j)`;
- `face_size(f)` -- constant for a triangle or quad face, runtime for a
  polygon.

**Recommendation (section 7, question 1): give the shape path's views the
same entity methods** -- `face_size(f)` on face views, `num_faces(c)` on
cell views, each returning the shape's constant. Section 10 withdrew
"counts as methods" for the shape kernels themselves, and that stands:
shape kernels keep their compile-time constants. The point here is
narrower. Face-based finite-volume algorithms are the polyhedral path's
main use: `face_geometry`, `jump`, `upwind_flux`, `divergence`,
`external_faces`, and Perot reconstruction. Written against the entity
methods, each runs on both paths from one source.

**Size buckets.** Kernels that only stream -- sums over faces, flux
accumulation, volumes -- need no bound and run as one launch. Kernels
that keep per-cell scratch, such as López's crossings and cycles, need a
compile-time array size. The topology knows each cell's total face size
`S` (its count of face points). `for_each` groups cells by `S` into padded
buckets (32, 64, 128, 256), as it groups by (shape, order). The bucket's
cap is the view's class constant `MAX_FACE_POINTS`, so local arrays have a
fixed size. A kernel opts in by reading the constant; one that does not
get one group.

**Spaces on polyhedra:**

- `Constant` and `Values` on points, edges, faces and cells: as on the
  shape path. This is nearly all finite-volume data.
- `H1` order 1 (point data): values at points, read through
  `corner_value`-like access to a cell's points. There is no
  `value(c, pc)`, since there are no parametric coordinates.
  - Interpolating inside a cell, if wanted, is piecewise linear over the
    cell's sub-tetrahedra: cell center, face centers, each face fanned
    about its center. That is cheap and local.
  - Mean value coordinates (what `vtkPolyhedron` does) need all faces per
    point and are not offered.
- `L2` on polyhedra (DG, HHO and VEM style): polynomials in physical
  coordinates about the cell centroid, so the DOF count per cell depends
  only on the order. Deferred; the space has room for it.
- The geometry is `H1` order 1: positions per point. There are no curved
  polyhedra.
- **Face data's sign convention:** a value on a face (MPAS
  `normalVelocity`) is along the face's stored normal, out of side 0.
  Readers convert from the producer's convention.

## 4. Algorithms, as tests of the API

**Geometry**, streaming and one launch:

- face area vectors and centroids, fanned from the face's own centroid.
  The face is stored once, so both cells get the same triangulation, even
  on non-planar faces; VTK's duplicated copies need not;
- cell volume and centroid by the divergence theorem, signed by
  `face_side` and taken about the cell's first point, not the global
  origin (VTK loses precision far from it);
- point-in-cell by summed signed solid angles, fanned. Solid angles are
  additive, so no ear clipping is needed.

**Face-based finite volume:** `divergence`, `jump`, `upwind_flux` and
`external_faces` (faces with one side) run unchanged through the shared
entity methods. Perot reconstruction (from the explicit-ugrid branch) is
new: `u_c = (1/V) Σ_f A_f u_f (r_f − r_c)`, with the outward sign from
`face_side` rather than geometry.

**Contour (and slice), face-based López:**

1. **Per side** -- a cell → face entry, so a cell's sides are contiguous.
   Walk the face's points in the cell's outward order (`side_point`).
   Classify, and collect crossings. Each crossing is identified by its
   global edge, so both cells and both faces name it alike. Emit one
   successor pair per key crossing. Count, scan, emit.
2. **Per cell:** follow the cell's successor pairs around its cycles,
   within the bucket's scratch. Count polygons and their sizes; scan; emit.
3. **Iso-vertices:** one per crossing edge, positioned once. With
   `edges()` derived, the vertex id is the edge's compacted index (flag,
   scan), with no global merge pass. Without it, sort the `(low, high)`
   keys, as the shape path's contour merges.

Two choices need tests against VTK:

- **The saddle rule.** VTK's code pairs a key crossing with the *next*
  crossing on its key face; the design note's reformulation says
  *preceding*. They differ on faces with four crossings. Match the code,
  and test against `vtkContour3DLinearGrid` on saddle faces.
- **Triangulating output polygons.** VTK fans iso-polygons in
  `vtkContourHelper`, which is wrong for non-convex ones, and ear-clips in
  `ContourCell`. Proposed: emit polygons, and triangulate in a separate
  pass only when triangles are asked for -- a fan from the polygon's
  centroid (one extra point, branch-free), or a forward-walk ear clip
  capped at the bucket size, as VTK's WebGPU port does at 32.

**Threshold** keeps whole cells. The output is a polyhedral topology over
the kept cells' faces, renumbered; a face loses its side when that cell is
dropped, so a kept cell's face may become boundary. Faces stay shared.

**Clip** is deferred. Cut faces are new per cell, so the output either
duplicates them again or needs a matching pass. VTK's inside-out pairing
probably leaves inside-out clips open on saddle faces; choose that pairing
deliberately.

**Projections:** `to_cells` and `to_points` through cell → points; values
at sub-cells for slices.

## 5. API surface

```python
data = td.DataSet(td.PolyhedralTopology(...), positions)
data.topology.faces()       # sides, boundary(), groups(): one "Polygon" group
                            #   (or per size bucket)
data.topology.edges()       # as on the shape path
data.topology.cell_points() # derived cell -> points (CSR)
td.for_each(kernel, data, "cells" | "faces" | "edges" | set, *fields)

td.as_polyhedra(shape_dataset)     # any shape-based dataset, faces reused
td.interop.vtk.vtk_to_dataset(ug)  # polyhedral UGs: duplicated faces matched
td.interop.vtk.dataset_to_vtk(d)   # SetPolyhedralCells, side-1 copies reversed
```

`Values`, `Constant` and `H1` order-1 fields, sets, `traces` of cell
constants, and the face-based algorithms are shared with the shape path.
`L2` and `H1` order 2 are refused on a polyhedral topology, which has no
reference element.

## 6. Test data and oracles

- **Shape meshes converted to polyhedra.** Every face-based algorithm must
  reproduce its shape-path result exactly: face counts, sides, areas,
  volumes, divergence, external faces. A contour of a linear field must
  match the shape path's contour as a surface: the same area and the same
  points on cut edges. This is the strongest oracle: any mixed mesh, any
  backend.
- **VTK:**
  - `vtkPolyhedron`'s volume, centroid and `IsInside`;
  - `vtkContour3DLinearGrid` (López) on the same cells, saddles included;
  - `vtkGeometryFilter` for external faces;
  - VTK's own test meshes, copied with VTK's license into
    `packages/tack-vis/tests/data/vtk/`: `onePolyhedron.vtu`,
    `polyhedron2pieces.vtu`, `polyhedron_mesh.vtu` (wound inconsistently,
    and refused), `concavePolyhedron.vtu`, `sliceOfPolyhedron.vtu`,
    `vtkHDF/polyhedron.vtu` and `nonWatertightPolyhedron.vtu`;
    `largePolyhedral/` and `polyhedron.vtkhdf` are not used yet;
  - CGNS `Example_nface_n.cgns` (cell → faces with signs) and
    `Example_ngon_pe.cgns` (faces with owner and neighbour), the two
    orientation conventions, and `EngineSector.cgns`, from the same
    folder; these need a VTK built with its CGNS reader.
- **MPAS-like columns.** Voronoi cells of random points (SciPy's
  `Voronoi`) extruded into prisms with polygonal bases, thin vertically as
  ocean cells are. Exactly the case where geometric orientation failed.

## 7. Questions to settle

1. **Shared entity methods** (`face_size(f)`, `num_faces(c)`) on both
   paths' views, so face-based finite-volume algorithms are written once?
   Recommended: yes, for faces and sides. The shape path's own kernels
   keep their constants.
2. **Orientation storage:** a `u8` side per cell → face entry (as above),
   or the sign packed into the face id's high bit? `u8` matches the shape
   path's `side_slot` and the narrow-column lesson.
3. **Store cell → faces, or owner/neighbour?** Proposed: cell → faces plus
   sides, with owner/neighbour derived. Owner/neighbour inputs (OpenFOAM)
   convert with one counting sort.
4. **Non-manifold faces:** refuse, as proposed, or a CSR face → cells?
5. **The saddle rule:** VTK's code (*next* crossing) or the design note's
   (*preceding*)?
6. **Output triangulation:** polygons by default, triangulated on request
   -- centroid fan or capped ear clip?
7. **Interior interpolation of point data:** sub-tetrahedra, or none until
   an algorithm needs it?
8. **Bucket caps:** fixed (32/64/128/256), or derived from the mesh?

## 8. A phased path

1. **Topology and conversions.** `PolyhedralTopology`, derived sides,
   cell → points and edges, `as_polyhedra`, VTK interop, `check_winding`
   and `orient`. Test: shape meshes round-trip.
2. **Shared entity methods and the face-based algorithms.** Face
   geometry, volume and centroid, divergence, jump, upwind flux, external
   faces and Perot run on both paths. Test: converted meshes reproduce
   the shape path exactly; VTK's `vtkPolyhedron` geometry.
3. **Face-based López contour and slice**, with size buckets. Test:
   converted meshes match the shape path's surfaces; VTK on polyhedral
   data, saddles included.
4. **Threshold**, then readers for the real formats (CGNS NFACE and NGON,
   OpenFOAM, MPAS) as the data calls for them.

## 9. Phase 1 as built

`packages/tack-vis/src/tack/data/polyhedra.py`, with views in `views.py`
and VTK conversion in `interop/vtk.py`:

- **`PolyhedralTopology(face_offsets, face_points, cell_offsets, cell_faces,
  cell_face_sides, num_points=None)`.** Everything derived is computed on
  the device and cached:
  - `faces()` (`PolygonFaces`): the shape path's `sides` record and
    `boundary()`, by scattering each cell -> face entry to its (face,
    side). It refuses a side used twice, which covers more than two cells
    on a face, and a face wound into its only cell.
  - `cell_points()`: (cell, point) pairs as u64 keys, sorted once.
  - `edges()` (`PolygonEdges`): face edges as `(low, high)` keys, sorted
    into runs. This gives the same numbering the shape path gives the same
    edges, and per face-point entry an edge id and sign.
- **Kernels.** Cell views are `_PolyhedralCells`: `num_faces(c)`,
  `face_id(c, k)`, `face_side(c, k)`, `face_size(f)`,
  `side_point(c, k, j)`, and `point_id(c, j)` over the derived points.
  Face views are `_PolygonFaces`, with the shape path's side methods.
  `Polyhedron` and `Polygon` are deliberately not `Shape`s, whose methods
  assume fixed points and shape functions; geometry on them is
  `_PointGeometry`, positions only.
- **Winding.** `check_winding` sorts every cell's directed face edges by
  (cell, edge): a run must be two walks, one each way. `orient` turns
  around faces wound into their only cell. Repairing mixed winding inside
  a cell is not attempted.
- **Conversions.** `as_polyhedra` reuses the shape path's derived faces
  unchanged: face ids, face fields and edge numbering all carry over, and
  sides match exactly. VTK import matches duplicated faces by point set
  and refuses copies wound the same way; it reads on the host, cell by
  cell. VTK export writes a copy per cell, reversed on side 1.
- **Tests (`test_polyhedra.py`).**
  - Every shape mesh and VTK solid converts with identical sides, boundary
    and edges.
  - Cells walk their faces as the shape's own tables do.
  - `check_winding` finds a turned face; `orient` mends inward boundary
    faces.
  - Refusals.
  - Extruded Voronoi columns: thin, with polygonal faces of up to eight
    points; Euler's V − E + F − C = 1 holds.
  - VTK round trips with positive volumes (1, 1, 1/3 for the mixed mesh)
    and `vtkGeometryFilter`'s boundary.
  - Six VTK data files.
  - Mutations caught: ignoring the side in `side_point`, in the side
    scatter or in the winding check; non-unique cell points; exporting
    side-1 copies unreversed.

**Findings:**

- VTK's own `polyhedron_mesh.vtu` is wound inconsistently. In each cell
  the faces point both ways, and the face the two cells share runs the
  same way in both, so VTK reports a volume of 14.5 million for a
  41 × 41 × 20 box. The import refuses it rather than guess.
- `nonWatertightPolyhedron.vtu` is closed cell by cell; its "non-watertight"
  is between cells.
- A shape-based mesh's derived faces are already a polyhedral topology.
  The conversion is a reindexing, which confirms that the two paths share
  their face conventions exactly.

**Still to do:** size buckets and López (phase 3); an import that is not
a host loop; `orient` beyond inward boundary faces.

## 10. Phase 2 as built

**Shared entity methods.** Both paths' cell views offer the same methods:

| Method | Shape-based cell (`_FaceWalk`) | Polyhedral cell |
|---|---|---|
| `num_faces(c)` | `NUM_FACES`, a constant | from the CSR offsets |
| `face_id(c, k)`, `face_side(c, k)` | the face incidence | the stored entries |
| `side_size(c, k)` | the face table's count | `face_size(face_id(c, k))` |
| `side_point(c, k, j)`, `side_position(c, k, j)` | `face_corner` (a voxel's pixels as quads) | the stored order, reversed on side 1 |

Face views on both paths offer `face_size(f)`. On the shape path every
count is a compile-time constant, so a shape kernel compiles as before.

`_FaceWalk` is a mixin of its own, added only to domain views. An order-2
field view also carries the face incidence (for its DOF numbering), and
template classification checks every method of a view, so a
`side_position` that needs geometry cannot sit in a mixin a field view
includes.

**Algorithms written once** (`algorithms.py`):

- `face_geometry`: Newell's normal and area, fanned from the face's first
  point, for polygons of any size.
- `face_centers`: area centroids, fanned from the points' mean, so both
  sides of a face stored once see one fan.
- `cell_geometry`: volume and centroid by the divergence theorem over the
  outward-walked faces, taken from the cell's first point.
- `divergence`.
- `jump` and `upwind_flux` of cell data, read from each side's cell (data
  with a basis still goes through `traces`, on the shape path).
- `boundary_faces`.
- `perot`: Perot's reconstruction from face-normal components, signed by
  each cell's side of each face.

**Tests:**

- On the mixed mesh, a rectilinear grid and VTK's tetrahedra, wedges,
  pyramids, hexahedra and voxels, every one of these gives the same
  answers on the shape path and on the same mesh as polyhedra.
- Known volumes and centroids: the pyramid's is a quarter of the way up.
- Volumes agree with VTK's `vtkPolyhedron::ComputeVolume`.
- Perot gives back a uniform velocity to 2e-4 (float32) on every mesh,
  thin Voronoi columns included, and a uniform flow leaves every closed
  cell with no net outflow.
- Mutations caught: Perot ignoring the side; the shape walk skipping the
  voxel's reordering; polygon faces sized 4 (this one crashes, reading
  past the face); polyhedra missing a face.

**Not yet:**

- `extract_surface` of a polyhedral topology. Its surface is polygons, and
  no surface topology holds them (a 2D polygonal topology -- this one, one
  dimension down -- would). It refuses, pointing to `boundary_faces`.
- Traces of point data on polygon faces. `SideTraces` is four points a
  face.

## 11. Phase 3 as built

**Polygons are the same topology, one dimension down.**
`PolygonalTopology(loop_offsets, loop_points)` is a `PolyhedralTopology`
whose cells are polygons and whose facets are edges.

- It is built from point loops, as producers and isosurfaces give them.
  Edges are matched by sorting: the first polygon to use an edge is its
  side 0, and a second must walk it the other way. Inconsistent winding,
  or a non-orientable surface, is refused, as is an edge of three polygons.
- Sides, boundary, the shared entity methods and the face-based algorithms
  all carry over. `check_winding` checks that loops close; `cell_geometry`
  gives areas and centroids.
- `as_polygons` converts triangle, quad and pixel meshes.
- `extract_surface` of a polyhedral mesh is now its polygonal surface.
  The tests check it is closed: no boundary, Euler number 2.
- This also answers the dataset design's question 7: a 2D mesh's facets
  are its codimension-1 entities, its edges.
- One fix it forced: side 1 walks a facet `n - 1 - j`. Reversing "from the
  same first point" is no reversal at all for a 2-point edge.

**Size buckets** (`SizeBuckets(topology, caps)`) group cells by size
through the same subgroup machinery that splits launches by order.

- A bucket is any object that gives each cell a key. `launch_groups`
  takes such keys beside fields.
- A polyhedral subgroup selects cell ids (`_SelectedPolyhedra`, through
  `cell(c)`) instead of gathering fixed rows.
- The key's `domain_mixin` gives each launch's view `MAX_SCRATCH`, so
  `tack.local_array(..., cells.MAX_SCRATCH)` has a compile-time size.
- Grouping by shape, by order and by size is now one mechanism.

**Contour (and slice) of polyhedra**, face-based López, in four passes,
each count → scan → emit:

1. Per side, count the key crossings: an outward walk stepping from below
   the isovalue to at or above it.
2. Per side, pair each with the next crossing round that face, both named
   by global edge id. On side 1, walk edge `j` is stored edge `n - 2 - j`.
3. Per cell, in its size bucket, follow the pairs round their cycles into
   polygons, with a `used` array of `MAX_SCRATCH`.
4. One iso-vertex per crossing edge, interpolated from the lower point
   id, as the shape path's contour does.

The output is a `PolygonalTopology`, and its constructor is a free
watertightness check: two polygons walking a shared edge the same way are
refused. Point fields are interpolated onto the surface; cell fields come
per polygon from the cell it lies in. `slice_plane` gets this through
`contour` unchanged.

**Tests:**

- Contours of polyhedra converted from every shape mesh (mixed,
  tetrahedra, hexahedra, wedges, pyramids, voxels) have exactly the shape
  path's points. For a linear field, whose iso-polygons are planar, the
  polygons' area equals the triangles'.
- VTK's López (`vtkContour3DLinearGrid` with `GenerateTrianglesOff`) on the
  same polyhedra gives the same polygons, wound the same way: on every
  solid, the Voronoi columns, and two cubes sharing a saddle face.
- Cells land in the bucket their size needs, and a cell too large is
  refused.
- Slice points lie on the plane, with fields carried.
- Mutations:
  - mapping side-1 edges wrongly fails 15 tests;
  - pairing with the *previous* crossing -- the design note's wording,
    question 5 -- fails exactly the saddle comparison, confirming VTK's
    code uses *next*;
  - counting inside → outside crossings instead changes nothing, rightly:
    round a closed face they are equal in number.

**Not yet:**

- Triangulating output polygons (on request).
- Contour lines of polygonal topologies.
- Threshold and clip (phase 4).
- Iso-polygons are left as polygons. They come out of the trace in a
  per-cell order, so their numbering differs from VTK's though the
  polygons are the same.

## 12. Phase 4 as built

**Oriented face values.** Threshold forced the question. It keeps whole
cells, so a face whose side-0 cell is dropped must be turned around for
its remaining cell. Turning a face negates a quantity measured along its
normal -- a normal flux, MPAS's `normalVelocity`, `upwind_flux`'s output
-- but not its area or an id. `Values(data, "faces", oriented=True)` says
which:

- it is a space of its own, interned apart from plain face values, and
  only faces may be oriented;
- `upwind_flux` produces it; `divergence` and `perot` read either kind;
- conversions carry the flag.

**Threshold of polyhedral and polygonal topologies.** Whole cells are kept
by cell values, or by point values (all or any), each step a count, scan
and compaction on the device:

- faces used by kept cells, stored once and numbered in their old order,
  so face values come along;
- a face whose side-0 cell is gone is turned around and its references
  made side 0, with its oriented values negated;
- points compacted.

A polygonal topology keeps its polygons' loops; its edge values, whose
numbering is derived afresh, are left behind.

**Repairing the winding of real data.**
`PolyhedralTopology.from_cell_faces(cells, num_points, positions=None,
orient=False)` is now public: per-cell face lists, as VTK and readers
give them, with copies matched by point set. VTK import goes through it.
With `orient`, on the host:

1. Within each cell, faces walk shared edges oppositely: breadth-first
   from face to face, reversing as needed.
2. Across cells, a shared face's two copies run opposite ways: whole
   cells reversed as needed.
3. Each connected component is turned outward by the sign of its total
   volume. This is the one geometric test, made once on a sum over many
   cells, which thin cells cannot fool the way they fool per-cell tests.

`vtk_to_dataset(grid, orient=True)` uses it. Without it, inconsistent
input is refused, and the message names the option.

**Readers through VTK.** VTK's CGNS reader brings the two polyhedral
conventions:

- `Example_nface_n.cgns`: NFACE_n, a sign per reference;
- `Example_ngon_pe.cgns`: NGON_n with ParentElements, owner and
  neighbour.

They give the same mesh, consistently wound. `EngineSector.cgns`, a real
CFD mesh of 1956 polyhedra, arrives wound inward in 1891 cells and
inconsistently within 65. It is refused as is; with `orient` it imports
in under a second, every cell positive, VTK's own volumes agreeing to
5e-7, and the boundary matching `vtkGeometryFilter`'s 6209 faces. Whether
the file or VTK's reader is at fault is worth a look in VTK.

**Tests.**

- Threshold of polyhedra converted from every shape mesh, by cells and by
  points (all, any), keeps exactly the shape path's cells and points, with
  the same volumes and faces.
- `vtkThreshold` agrees.
- Keeping the Voronoi columns' upper layers turns the faces between
  layers. A uniform flow's oriented flux, negated with them, still gives
  Perot's exact answer; the same numbers as plain face values do not.
- Threshold of polygons.
- `orient` undoes random face and cell reversals.
- Both CGNS conventions agree, and EngineSector is refused, then repaired.
- Mutations caught: oriented values left un-negated, faces left unturned,
  turned faces keeping their side, `orient` skipping the volume sign.

**Not yet:**

- clip, whose cut faces are new per cell;
- readers beyond VTK's: OpenFOAM's owner/neighbour and MPAS's
  `edgeSignOnCell` map onto `from_cell_faces` or the constructor
  directly, when there is data;
- carrying edge values through threshold.

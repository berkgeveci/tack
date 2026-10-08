# Polyhedral meshes: design proposal

Status: proposal for discussion, 2026-10-08, on branch `vis/polyhedra`.
Nothing here is implemented. It refines section 10 of
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
  - the data in `~/Data/VTK/Data`: `polyhedron.vtu`,
    `concavePolyhedron.vtu`, `largePolyhedral/`, `polyhedron.vtkhdf`;
  - CGNS `Example_nface_n.cgns` (cell → faces with signs) and
    `Example_ngon_pe.cgns` (faces with owner and neighbour), the two
    orientation conventions.
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

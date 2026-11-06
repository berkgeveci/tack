# Changelog

All notable changes to Tack are recorded here. Rules cited by name live in
[`docs/reference/language-contract.md`](docs/reference/language-contract.md).

## 0.7.0 — 2026-11-06

The headline: Tack's filters run inside VTK. Nearly every `tack.dataset`
filter is a VTK algorithm (`tack.interop.vtk.algorithms`), its
arguments the algorithm's parameters, taking trees of datasets, AMR
hierarchies and pieces over MPI ranks, and running on the device its
`device` names. Datasets cross between VTK and Tack without a copy
(`copy=False`): on the CPU, on Metal, and on CUDA, HIP and Level Zero with a
VTK that holds device arrays. Examples 47-53 are VTK pipelines with Tack's
filters in them, on real Exodus, OpenFOAM and AMR files. Higher orders:
bases of any order on every shape from segments to wedges, Bernstein and
serendipity spaces, `HDiv` and `HCurl` at order 1, curved geometries, and
contour, slice, clip, threshold, advection and integration of them, cut at
the field's roots; VTK's Lagrange, Bezier and quadratic cells, its
`vtkCellGrid` and more of MFEM, both ways, with galleries in the
Visualization Guide. Advection gains an adaptive RK45, a path length, a
terminal speed and a walk through face neighbours; `tack.algorithms` gains
segmented reductions and scans, the same bits on every backend. The
documentation site is rewritten around Tack's four layers, the repository
is now `github.com/Kitware/tack`, and tack-core is on PyPI
(`pip install 'tack-core[cpu]'`), with tack-vis installed from this
repository at the release tag.

Nine changes break code written for 0.6; they lead the Changed section. The
ones most programs meet: `data.fields` is a `Fields` mapping that refuses an
ambiguous name, `project` into `Constant` averages over each cell rather
than its corners, `extract_surface` keeps only the points its faces use, the
VTK algorithms of points, lines and surfaces give `vtkPolyData`, and a
kernel over cells that reads their faces or edges asks for them with
`incidence=`.

Validated on Linux (CPU; CUDA on an RTX 6000 Ada; HIP on two MI210s; Level
Zero on an Intel Max 1100) and macOS (CPU and Metal on an M1 Max), with
debug mode, both id widths of the dataset tests and MPI, against VTK 9.7.1,
VTK master and VTK builds holding CUDA, HIP and Level Zero arrays, together
with tack-examples' ports of production datasets. A targeted round after
the last fixes reran what they touched on every machine, and the disk cache
now settles: a second and third run of the quick tests add no entries.

### Added

- Segmented operations in `tack.algorithms`: `segment_reduce` (sum, min, max, mean),
  `segment_scan` (inclusive and exclusive sums, running min and max),
  `segment_broadcast` and `segment_ids`, over flat values and CSR offsets,
  empty segments kept -- each cell's points, each point's cells, sparse rows.
  A segment of up to 256 elements is one thread's; a longer one goes by
  chunks of 256, a segment over several chunks combined a level up, so one
  enormous segment is a few launches, not one thread's walk. On an M1 Max's
  GPU a 10M-element segment reduces in 4 ms (a thread per segment: 1.2 s), a
  skewed mix of 1M short segments and one of 5M in 3.7 ms (590 ms), and 2.5M
  segments of about 4 in 0.66 ms (0.59). No atomics in the arithmetic, so
  float sums are the same bits on every backend; `ordered=True` sums each
  segment in element order, as `reduce_by_key` does. Empty segments take
  `empty=` (0, NaN for a mean, or the type's extremes for min and max), or
  keep their output with `empty="keep"`; vectors reduce componentwise; debug
  mode checks the offsets.
- `advect(..., neighbors=True)` and `streamlines(..., neighbors=True)`: a stage
  whose particle has left the cell it was last found in tries that cell's face
  neighbours before the locator's bins, taking one, as the last cell is taken,
  only where it holds the point well inside, so the paths are the bins' bit
  for bit. On the CPU, streamlines on the aneurysm (8.7M tetrahedra) took 1.45
  to 2.04x less time (20,000 seeds: 1116 to 546 ms) and on the ONERA M6 (46.9M
  cells) 1.63 to 1.90x (16,384 seeds: 2067 to 1086 ms). About one location in
  seven missed the last cell, and 43-49% (aneurysm) and 78% (M6) of the misses
  were in a face neighbour. The neighbours are a table of six ids a cell
  (`CellLocator.cell_neighbors()`), built from the faces on first use and kept
  with the locator: 0.9 s and 210 MB on the aneurysm, 4.1 s and 1.1 GB on the
  M6, so the walk is off by default. Pyramids, polyhedra and surfaces search
  the bins as before.
- Tack's filters as VTK algorithms, `tack.interop.vtk.algorithms`:
  `VTKPythonAlgorithmBase` subclasses that sit in a VTK pipeline beside VTK's
  own, `(reader >> Contour(field="pressure", value=0.5))()`. `Contour`,
  `SlicePlane`, `Clip`, `Threshold`, `ExtractSurface`, `Tetrahedralize`,
  `SurfaceNormals`, `ConnectedComponents` and `Streamlines` make a dataset;
  `Gradients`, `Elevation` and `CellQuality` add arrays to a shallow copy of
  their input; `GhostCells`, `GenerateGlobalIds` and `MergePieces` take every
  piece of a `vtkPartitionedDataSet` together, and the others run each piece.
  Parameters are properties; an exception is VTK's error and is kept as the
  algorithm's `error`. CPU only for now.
- `vtk_to_dataset(..., copy=False)` and `dataset_to_vtk(..., copy=False)` share
  memory on the CPU instead of copying: points, point and cell data, cell
  types, offsets and connectivity (ids in VTK's width), ghost marks and global
  ids, wherever the values are one contiguous block of the type Tack keeps
  (anything that must change, such as a type, an origin, strips or polyhedra,
  is copied). Shared VTK arrays become read-only Tack arrays that keep VTK's
  alive, and the other way. `dtype=None` keeps each floating-point array in
  its own type. The algorithms convert this way, so a pipeline copies nothing.
- Examples 47-49: a VTK pipeline with Tack's filters in it, from a VTK reader
  to VTK rendering; pieces of a `vtkPartitionedDataSet` through ghost cells,
  contours and `MergePieces`; streamlines.
- `integrator="rk45"` on `advect` and `streamlines`: steps that adapt to the
  flow, as VTK's `vtkStreamTracer` with its RK45 (Cash-Karp, `vtkRungeKutta45`)
  and its rules. `step_size` is the first step; a step whose error estimate
  passes `tolerance` (a distance, 1e-6 by default) is tried again smaller, its
  first stage kept; each step taken proposes the next, between `min_step` and
  `max_step` (a fiftieth of `step_size` and twice it by default, VTK's
  ratios), in either step unit. A particle's step goes with it between a GPU's
  launches (bit for bit with one launch) and from piece to piece. On a swirl
  at VTK's settings, Tack's RK45 takes about half of fixed RK4's points (42-44
  a path, VTK 42) and lies within 2-3e-5 of the path, VTK's within 1-2e-5.
- `max_length` on `advect` and `streamlines`, with any integrator: a path that
  has gone that far stops (`TERMINATE`); in the length unit its last step
  lands on it.
- The VTK comparison harness (`benchmarks/vtk_compare`) times point gradients
  against `vtkGradientFilter`, and the performance page says what its dashes
  are: comparisons not made, not failures (VTK's Viskores filters have no cell
  gradient, volume or centers; the Linux run timed VTK against Tack's CPU
  alone).
- Bases of any order on segments, quads, pixels, hexahedra and voxels
  (`tack.dataset.bases`): `H1(data, order=p, basis=...)` from order 3 and
  `L2(data, order=p, basis=...)` from order 2 (or on Gauss-Legendre nodes from
  order 1), on Gauss-Lobatto nodes (the default, MFEM's for `H1`),
  equispaced ones (VTK's Lagrange cells) or, for `L2`, Gauss-Legendre ones
  (MFEM's default for `L2`). `H1` holds `p - 1` values per edge, `(p - 1)^2`
  per quad face and `(p - 1)^d` per cell after the points, each cell reading
  a shared edge in order from its lower point id and a face in the frame of
  its own row, so cells turned every way share them; `L2` keeps each cell's
  `(p + 1)^d` values. `interpolate`, `project`, `probe`, `gradients` and
  curved geometry of any order work on them; polynomials of degree `p` come
  back exactly.
- `subdivide(data, level)`: curved or high-order cells as linear sub-cells on
  each cell's lattice of `level` steps, fields at the lattice points (VTK's
  `vtkCellGridToUnstructuredGrid`). A lattice point shared by cells is one
  point, evaluated once, by the first cell holding it.
- The MFEM interop takes `H1` from order 3 and `L2` from order 2 on quads and
  hexahedra, and curved meshes of any order on them: on Gauss-Lobatto,
  Gauss-Legendre and closed uniform nodes the values are copied in Tack's
  numbering, on any other basis (Bernstein) evaluated at Tack's nodes,
  exactly either way. `mfem_grid_function(data, field, mesh)` gives a field
  back as an MFEM `GridFunction`, the same DOFs in MFEM's numbering.
- VTK's Lagrange cells: `vtk_to_dataset` reads a grid of
  `VTK_LAGRANGE_CURVE`, `_QUADRILATERAL` or `_HEXAHEDRON` cells of one order
  `p` as linear cells on their corners with the geometry and point data in
  `H1(order=p, basis="equispaced")`, VTK's nodes; `dataset_to_vtk(...,
  higher_order=True)` writes a high-order or curved dataset of those shapes as
  Lagrange cells of its largest order, each shared node one point, which VTK
  evaluates as Tack does.
- `contour`, `slice` and `slice_plane` take `subdivision=`: on the tensor
  shapes, a field above order 1 or a curved geometry is cut on linear
  sub-cells, by default as many lattice steps as the largest order of the
  field and geometry, and only the cells whose field's Bernstein
  coefficients can hold the value are refined. The surface has no cracks;
  its error falls as the square of the step. `subdivision=1` cuts the cells
  linearized at their corners; `tolerance=` (in the field's unit, a distance
  for slices) chooses the fewest steps whose estimated error is within it,
  from the Bernstein coefficients' distance from the corners' interpolation.
- `subdivide`, and `contour` and `slice` on sub-cells, keep ghost marks: a
  sub-cell takes its parent's, and a lattice point is marked as `carry` marks
  the points it makes, so each piece's cut of a high-order field is its share
  of the whole's. A shared lattice point has the same bits in every piece
  holding it: an edge or face point is computed from that entity's own
  values in a frame its corners' positions fix, and anything else in the cell
  every piece chooses alike (the lowest global id, or the first centroid by
  position). Pieces' surfaces merge with `merge_pieces` into the whole's.
- `advect` and `streamlines` through curved cells (an `H1` geometry above
  order 1) and with a velocity of any order, or in `L2`, on hexahedra,
  voxels, quads, pixels and the order-2 tetrahedra, wedges and triangles.
  The search from the points reads a curved cell and a high-order field
  through its shape group's own view, so it inverts a curved cell by its own
  map and evaluates the field as `probe` does: the paths are an RK4 over
  probed velocities to 1e-12. `probe` on a curved geometry searches from the
  points too, with the same answers as from the cells. Advection refused
  both before. A mesh of one shape compiles its kernels in seconds; a curved
  mesh of several shapes takes minutes on its first call.
- `L2(..., layout=)`: how a tensor basis' values are stored, `"blocks"` (each
  cell's together: MFEM's and VTK cell grids' layout) or `"cells"`
  (cell-fastest, so neighbouring GPU threads read neighbouring values).
  `"auto"`, the default, is `"cells"` for a tensor basis of order 3 or more on
  a GPU and `"blocks"` elsewhere: on an RTX 4060 Ti cell-fastest reads 2.3x and
  writes 3.6x faster at order 6, on the CPU blocks reads 1.5-1.8x and writes
  about 2x faster. `to_device` transposes when the layout changes; the MFEM
  interop reads and writes either; `space.blocks(values)` and
  `space.indices()` read a field per cell whatever its layout.
  `TACK_L2_LAYOUT=cells` (or `blocks`) overrides `"auto"` for a whole run.
- `H1` and `L2` of any order on triangles and tetrahedra, with equispaced nodes
  (`basis="equispaced"`, VTK's and gmsh's Lagrange simplices): values shared by
  entity as on the tensor shapes (`p - 1` per edge, `(p - 1)(p - 2) / 2` per
  triangle face, `(p - 1)(p - 2)(p - 3) / 6` per tetrahedron), a triangle
  face's placed through its orientation in each cell. `interpolate`, `probe`,
  `gradients` and `carry` take them; polynomials of the order are reproduced
  exactly.
- `contour`, `slice` and `subdivide` of fields above order 1 on triangles and
  tetrahedra, on Freudenthal's sub-simplices of the cells' lattice: a lattice
  point shared by cells is computed once, and in every piece holding it to the
  same bits, so surfaces close and pieces agree.
- Curved triangles and tetrahedra of any order: a geometry in `H1` of order 3
  and up (equispaced) on simplices, located as it is (its Bernstein control
  points bound it, Newton inverts its map) by `probe`, the cell locator and
  advection, and subdivided by `subdivide`, `contour` and `slice` (order 2's
  curved simplices too), each shared lattice point from its edge's or face's
  own nodes.
- VTK's Lagrange triangles and tetrahedra, both ways
  (`dataset_to_vtk(..., higher_order=True)`, `vtk_to_dataset`), and MFEM's
  triangles and tetrahedra of any order: closed-uniform values copied, MFEM's
  default bases evaluated at Tack's equispaced nodes, exactly; curved MFEM
  simplex meshes come across as curved geometry.
- Wedges of any order (equispaced): `H1` and `L2`, curved wedge geometry,
  subdivision for `contour`, `slice` and `subdivide` (Freudenthal's triangles
  in layers), VTK's Lagrange wedges both ways, and MFEM's wedges.
- `clip` and `threshold` of a curved geometry or a field above order 1, on
  every shape `contour` subdivides: the cells whose field's Bernstein bound
  lies wholly on the kept side stay as they are (curved, their fields of any
  order), and only the others are cut, on linear sub-cells of their lattice
  (`subdivision=`, `tolerance=`, as `contour` takes them). The whole part owns
  the corners on the seam and the cut part's copies are `DUPLICATEPOINT`, so
  reductions over the two count each point once, with or without ghost layers.
- Exact crossings: `contour`, `slice`, `slice_plane` and `clip` of a field above
  order 1 or a curved geometry put each crossing of a sub-cell's edge where the
  field itself takes the value along it (a fixed number of Illinois steps from
  the lattice values), and carry the geometry and every field there, so the
  vertices lie on the true surface whatever the subdivision. A crossing on an
  edge or face comes from that entity's own values in a frame its corners fix,
  so neighbouring cells and pieces put it at the same bits. `exact=False` keeps
  the crossing between the lattice values.
- Bernstein spaces: `H1(data, order=p, basis="bernstein")` and `L2(...,
  basis="bernstein")` on segments, quads, pixels, hexahedra, voxels, triangles,
  tetrahedra and wedges, and a curved geometry in Bernstein form. The values are
  the Bernstein polynomials' coefficients (control points, for a geometry),
  which bound the cell's field or geometry. `interpolate` and `project` fill
  them from values at the equispaced nodes. Each edge's and face's coefficients
  are formed in a frame its corners' positions fix, so pieces sharing it agree
  to the bit. Every filter sees the same polynomials as the equispaced Lagrange
  space of the order.
- VTK's Bezier cells (curves, quadrilaterals, hexahedra, triangles, tetrahedra,
  wedges), both ways, as Bernstein spaces: `vtk_to_dataset` reads them,
  refusing rational weights other than 1, and `dataset_to_vtk(...,
  higher_order="bezier")` writes them. `higher_order=True` chooses Bezier cells
  for a dataset with Bernstein spaces.
- `H1` of order 2 on segments.
- Element-to-DOF tables: `H1(data, order=p, dofs=table)` numbers a space's values
  by an explicit table, a row per cell giving each node's value index, as MFEM
  and VTK's cell grids store them, instead of by entity. A cell's corners must be
  its points; the other nodes may be numbered freely (VTK's layout, every node a
  point, included). Every filter reads it, a curved geometry may be numbered so,
  and filters' outputs number by entity again.
- Serendipity: `H1(data, order=2, basis="serendipity")` on quads, pixels,
  hexahedra, voxels and wedges (the full order-2 space on triangles and
  tetrahedra): the corners' and edges' values alone, the quad faces' centres and
  the cells' insides determined from them. Every filter reads it as the full
  order-2 space, with no kernels of its own; a geometry may be serendipity too.
- VTK's classic quadratic cells, both ways: `vtk_to_dataset` reads
  `VTK_QUADRATIC_QUAD`, `_HEXAHEDRON` and `_WEDGE` (quad8, hex20, wedge15) into
  serendipity, `VTK_QUADRATIC_TRIANGLE` and `_TETRA` into the full order-2 space;
  `dataset_to_vtk(..., higher_order="quadratic")` writes them (`True` chooses them
  for a dataset with serendipity fields).
- `interpolate` (and `node_views`) on a curved geometry of another order, on
  hexahedra, quads and the other tensor cells: an `H1` field of order 3 on an
  order-2 geometry, say. Each edge's and quad face's nodes are on the
  geometry's trace there, read from that entity's own geometry values in a frame
  its corners fix, so every piece holding it gets the same bits. It was
  refused; curved simplices and wedges of another order still are.
- `mfem_to_dataset(..., dofs="table")` and `mfem_field(..., dofs="table")` keep
  a curved geometry and `H1` fields of order 2 and up numbered as MFEM numbers
  them: the values as they are, in a space numbered by MFEM's element-to-DOF
  table (`H1(..., dofs=table)`), on every basis whose values are copied.
- A gallery in the Visualization Guide: contour, slice, clip, threshold,
  surfaces with feature edges, cell quality, gradients, probe, subdivide,
  pieces with ghost cells, streamlines, glyphs and pathlines on polyhedra,
  tetrahedra, an order-4 field on straight hexahedra, an order-3 field on
  curved ones and a mixed mesh, each picture a script in `docs/gallery/` (Tack makes what is
  drawn, VTK draws linear polygons and lines), rebuilt by
  `docs/gallery/build.py`.
- `extract_edges(..., subdivision=n)`: each edge as `n` line cells through points
  on a curved geometry's trace, so a mesh's outline shows its cells as they are.
- `tubes(lines, radius, sides=12, radius_by=None)`: polylines or line cells as
  tubes of quads with outward normals, the frame carried along each line so it
  does not twist (VTK's vtkTubeFilter); `glyphs(data, vectors, scale=1.0,
  scale_by=None, resolution=8)`: an arrow at each point along a vector field, as
  closed triangle surfaces wound outward (vtkGlyph3D with its arrow). Both carry
  the input's fields.
- `gradients(data, field, at=L2(...))`: the gradient at each cell's nodes of an
  `L2` space, exact there, so a high-order field's gradient keeps its order's
  detail, which averaging to the points loses.
- `surface_normals(..., weights="area")`: point normals with each cell's normal
  weighted by its area, so a cut surface's slivers do not swing them.
- `streamlines` and `advect` take `terminal_speed` (in the velocity's unit, or a
  quantity), `vtkStreamTracer`'s TerminalSpeed: a path ends at its first point
  where the speed is at or below it, the seed included, and the particle is
  `TERMINATE` and `SLOW_SPEED` (a new `ParticleStatus` bit, 256). It holds over
  pieces and launches as `max_length` does. Without it a path in a field that
  is near zero but not zero (the carotid's outside its vessel) wandered on to
  its length or step limit. VTK also drops a step whose two ends' average
  speed is at or below the limit, so it may end a path one point sooner.
- `integrate(data, fields=None)`: a dataset's measure and its fields' integrals
  (`Integrals`), as VTK's vtkIntegrateAttributes, by Gauss quadrature in each
  cell's reference element through its own map (collapsed onto simplices and
  wedges; `integrate.quadrature`), so curved cells and fields of any order
  integrate exactly to rounding; owned cells only, and over a
  `PartitionedDataSet` every piece on every rank; the cells of the mesh's
  dimension only, as VTK's (a volume leaves out its surfaces, lines and
  vertices). Polyhedra take cell data.
- `warp` by a displacement of higher order than the geometry (a finite-element
  solution, as MFEM's examples displace their meshes) lifts the geometry into
  its space, so the output is curved; a 2-vector displacement (a 2D problem's)
  moves x and y.
- The MFEM interop reads meshes whose nodes are in MFEM's older collections
  ("Linear", "Quadratic", "QuadraticPos", "Cubic", as `escher-p3.mesh`), and
  periodic meshes, whose nodes MFEM keeps per element, with an `L2` geometry.
- An MFEM gallery in the Visualization Guide: seven of MFEM's examples (Poisson
  on the curved Escher knot and Fichera corner, elasticity warped and in DG,
  adaptive refinement, the curved sphere, DG advection on a periodic mesh),
  solved with PyMFEM, imported with `tack.interop.mfem` and drawn from Tack's
  filters (`docs/gallery/mfem_examples.py`).
- `vtk_to_dataset` reads a `vtkCellGrid` (VTK's discontinuous-Galerkin cells, as
  `vtkCellGridReader` reads `.dg` files): its cells on the shape attribute's
  corners, `C1` attributes copied (shared as `H1`, per cell as `L2`), `constant`
  ones as `Constant`, and every other `HGRAD` or Bezier basis resampled by VTK's
  evaluator onto Tack's equispaced space of its order, exact where the spaces
  agree. `HCURL`, `HDIV` and pyramids' rational bases are left out with a
  warning, curved pyramids refused. It needs a VTK newer than 9.7.1, with
  `vtkCellAttribute`'s cell-type accessors and the evaluator's interpolate-only
  mode. `test_vtk_cell_grid.py` holds every `.dg` file of VTK's test data to
  VTK's evaluator, cell centres, probe, ranges, linear cells and transform (it
  skips without them).
- `dataset_to_vtk(data, cell_grid=True)` writes a `vtkCellGrid`: each shape's
  cells a DG cell type (`vtkDGCell.SetCellSpec`), the geometry its shape
  attribute, order 1 in VTK's complete basis `C` (shared for `H1`, per cell for
  `L2`), higher orders in its arbitrary-order basis `A` (VTK's `A` nodes are
  Tack's equispaced nodes in Tack's order, which the tests show on hexahedra,
  quads, tetrahedra and wedges), `Constant` fields as `constant`. Reading takes
  each type's corners from its cell spec and orders types by the specs'
  offsets, so a grid round-trips.
- `vtk_to_dataset` reads every VTK higher-order cell Tack has a shape for, of any
  shapes, dimensions and orders in one grid (Lagrange, Bezier, quadratic,
  biquadratic, triquadratic, quadratic-linear, the 7-node triangle and cubic
  lines, with linear cells beside them): each type's nodes matched to Tack's by
  VTK's parametric coordinates and copied, and where Tack's cell has nodes a
  VTK cell lacks, VTK's own interpolation functions give them, exactly.
  `test_vtk_higher_order.py` holds VTK's higher-order test files and one cell of
  each type to VTK's interpolation, derivatives, contours, slices, clips,
  boundaries and edges (it skips without VTK's test data).
- A field's nodes on a curved simplex or wedge geometry of another order (an
  order-4 field on quadratic tetrahedra, VTK's quadratic cells among them) are
  placed where the geometry puts them: on edges and quad faces its trace there,
  as on hexahedra since #217, and on triangle faces its trace in the face's
  canonical frame (its corners by position) evaluated in the geometry's
  Lagrange or Bernstein basis (`views._CurvedTriangleNodePosition`), so every
  piece holding a face gets the same bits. They were refused.
- `dataset_to_vtk(copy=False)` and `vtk_to_dataset(copy=False)` share Metal arrays
  with VTK, with any VTK: on Apple silicon a Metal buffer is host memory, so VTK
  holds views of the buffers themselves (after one wait), and with Metal the
  default device they come back as the same buffers. VTK's own memory is wrapped
  in place where it lies on whole pages and copied to Metal otherwise, so the
  dataset is on Metal throughout. Metal's `wrap_ptr` (and
  `tack.array_from_ptr(..., device="metal")`) takes a NumPy array of host memory
  for this.
- `convert` takes a `Field` and gives it in another unit, a new field in the
  same space, scaled and offset on the device, as a dataset's named field
  converts; before, a field converted only inside a dataset.
- `tack.set_default_float(tack.f64)` and `tack.get_default_float()`: the
  float type of arrays nothing else types (constructors without a `dtype`,
  Python floats, `vec`/`mat` types without a component type, implicit
  arrays) and of `tack.dataset`'s constructors, sources and readers. f32
  unless set; kernels' precision still follows their arrays.
- Every `H1`, `L2` and `Constant` field view on cells with a reference element
  numbers its own nodes: `NODES` per cell, node `k` at `node_parametric(k)`,
  its value `dof(c, k)` stored at `dof_index(c, k)`. One kernel computes a
  field at another's nodes on every shape, space, order and basis
  ([Work at a field's nodes](docs/visualization/write-a-mesh-algorithm.md#work-at-a-fields-nodes)).
  Order 1 had no `dof_index` and no node count of its own; a loop over the
  shape's `NUM_QUADRATIC`, the quadratic element's nodes, wrote nothing on
  segments, whose order-2 view numbers its nodes in a tensor basis.
- A dataset's fields may share a name in different spaces, as VTK's point and
  cell arrays do: `data.fields["p", td.Constant]` (a kind of space, or the
  space itself) picks one; algorithms take `("p", td.H1)` wherever they take a
  field's name, and `fields="p"` on a filter keeps every field of that name.
  `data.fields.named(name)` and `data.fields.entries()` list them.
- `vtk_to_dataset(grid, polyhedral="split")`: a VTK grid mixing polyhedra with
  shaped cells as a `PartitionedDataSet` of two pieces over the same points --
  the shaped cells an `UnstructuredTopology`, the polyhedra a
  `PolyhedralTopology` -- sharing positions and point fields, each point owned
  by one piece (the other marks it `DUPLICATEPOINT`), global ids the grid's
  point and cell ids. A CFD mesh of 20.9M hexahedra, 67K wedges and 508K
  polyhedra reads in 4.6 s and 14.7 GB (`copy=False`), where reading every cell
  as a polyhedron did not finish in 64 GB.
- `vtk_to_dataset` finds tetrahedra, hexahedra, wedges and pyramids that are
  inside out (corners in the reverse of VTK's order, as some converters write
  prisms: negative volumes, inward faces) on every read -- a kernel screens
  every cell, the volume decides -- and warns, naming `orient=True`, which now
  turns them outward too. A tangled cell, negative at some corners but not
  inside out, is left. Polyhedra are checked by their volumes (`cell_geometry`)
  once their faces are matched: a region of them wound inward together passes
  the matching, and, in a split read, nothing compares them with the shaped
  piece; they are named in a warning, and `orient=True` turns them outward.
  VTK's own `concavePolyhedron.vtu` is such a polyhedron (exact signed volume
  -11). A CFD mesh's 66,606 inside-out wedges took 0.3 s to find
  among 21M cells. Wedges are the common case: VTK's documentation gave their
  corner order the reverse of its code until March 2026, and writers that
  followed it wrote them inside out.
- Implicit arrays read one value from the host, as a stored `Array` does:
  `values[k]` (a Python number, or a NumPy array of a vector's components;
  negative `k` from the end) and `len(values)`, so `data.geometry.values[0]` works
  on images and rectilinear grids. Each read is one launch; iterating raises, as
  on an `Array`. Underneath, a `@tack.func` may now have a host implementation
  (`@name.host`), which Python calls while kernels inline the device body.
- A finite-volume guide covering cell/face associations, orientation, integrated
  flux balances and Perot velocity reconstruction, with a checked example on
  shaped and polyhedral meshes in the documentation validation suite.
- `project(..., average="nodes")`: into `Constant` (or order-0 `L2`), the mean
  of the source's own values in each cell (its `NODES`), as
  vtkPointDataToCellData and Viskores' CellAverage take it, for every VTK
  cell type and order.
- Write a Filter (Visualization Guide) rewrites a sixth built-in, `contour` of
  a triangle mesh: points made on edges (`Interpolate`), lines inside cells
  (`Pieces`), held to `td.contour` every part. Its section takes `carry` apart:
  each map, each kind of entity's keyword, and what goes with each (point and
  cell fields, the geometry, sets, ghost marks, global ids, the frame), which
  the guide covered only in a table.
- `merge_points` takes polyhedral, polygonal and polyline topologies. A
  polyhedral mesh's faces are matched again on the merged points, so the copies
  of a face that pieces each stored once, with one cell, become one face with
  both: OpenFOAM's processor pieces put together without merging (an AhmedML
  volume: 1,561,917 copied points of 24,358,265, 1,494,132 faces stored twice)
  merge in 13 s into the mesh their solver ran on, whose boundary is the walls
  and the far field (40,700 faces not at rest, not 3,028,964) and whose body is
  one piece, not 116. Copies wound the same way, or a face with two cells on a
  side, are refused.
- `CellLocator` locates in polyhedra that are not star-shaped from the mean of
  their points, as snappyHexMesh's concave cells at walls are: each one's cell
  point moves into its kernel, where it sees all of its faces, given as
  weights on its points so linear fields stay exact
  (`CellLocator.moved_cell_points`). `probe`, `advect` and `streamlines` use
  it. On an AhmedML volume (21.5M cells read as polyhedra), 119 of the 131
  cells the locator refused have a kernel.
- `vtk_to_dataset(grid, merge_points=True)` (or a tolerance) merges an
  unstructured grid's coincident points before reading it, as a mesh written
  per processor needs: the polyhedra's faces are matched once, and with
  `polyhedral="split"` every point is owned by one piece. Merging the pieces
  afterwards could leave a point owned twice or by no piece.
- `generate_global_ids` of a filter's outputs from pieces without ghost
  cells (which lost their marks, so reductions refuse them) matches every
  point and cell and marks the owners, so `statistics` and `entropy` then give
  the whole's answer; the refusal names it. Such outputs keep their cells'
  marks from their source cells.
- `world_gradient(cells, u, c, pc)`, the device function `gradients` uses:
  a field's gradient in the world at a parametric point, for kernels that
  need it where `gradients` does not put it.
- `world_hessian(cells, u, c, pc)`: a field's second derivatives in the world
  at a parametric point of a solid cell, for kernels. By the chain rule
  through the geometry: the field view's new `parametric_hessian` less the
  geometry's new `geometry_curvature` along the world gradient (zero where the
  map is affine), taken to world axes by the inverse Jacobian. Analytic from
  the shapes' new `shape_hessian` and `quadratic_hessian` at orders 1 and 2,
  and from the bases at any order on tensor shapes, simplices and wedges
  (Lagrange and Bernstein), a wedge's from its triangle factor's second
  derivatives and its segment factor's.
- `world_jet(cells, u, c, pc)`: a scalar field's value, world gradient and
  world Hessian together (on surfaces, `world_gradient`'s in-plane gradient
  and a zero Hessian), from the field views' new `parametric_jet`, which
  reads each of the cell's values once. Each read of an `H1` value of order 3
  or more resolves where its node is stored, about two thousand lines of
  generated CUDA, so a kernel that needs several of the three compiles that
  once.
- `HDiv` (Raviart-Thomas) and `HCurl` (Nedelec, first kind): at order 1 on
  hexahedra, voxels and tetrahedra, and of any order on hexahedra and voxels,
  where the values on shared faces and edges are kept in the face's or edge's
  own frame, so cells that number their corners differently agree.
  `HDiv` holds one value per face, its integrated flux along the face's
  stored normal (the convention of oriented face values, `upwind_flux` and
  `net_flux`); `HCurl` one value per edge, its tangential integral from the
  low point id to the high one. A field view's `parametric_value(c, pc)` is
  the field in the reference cell, and `hdiv_value` and `hcurl_value` map it
  into the world through the contravariant and covariant Piola maps (the
  contravariant by |det J|, so inside-out cells agree with their neighbors);
  `hcurl_curl` gives an `HCurl` field's curl and `hdiv_div` an `HDiv` field's
  divergence.
- `H1.quadratic_count(shape)`: the nodes a view of an `H1` space of order 1
  or 2 reads on a shape's cells as the shape's own functions, for kernels
  that loop over them.
- `td.edge_crossing(level, a, b, va, vb, xa, xb)`, the rule every built-in cut
  places a crossing by, for filters of one's own: the ends in the order to
  interpolate from (the end first by position) and the weight, or the end
  itself when the crossing is that point.
- `advect(..., step_unit="length")`, as `streamlines` takes it: `step_size` an
  arc length along the path rather than a time, over pieces too.
- `UnstructuredTopology.cell_points()`, each cell's points sorted and unique, as
  polyhedral and polyline topologies give them.
- `ghost_cells` takes polyhedral pieces (OpenFOAM's processor pieces read as
  polyhedra): each ghost is sent with its faces wound out of it and the
  piece's faces are matched again, so a face a ghost shares with the piece's
  own cell is one face; `update_ghosts` and `generate_global_ids` run on
  them. The split reading's two pieces are ghosted together once the shaped
  piece is made polyhedral (`as_polyhedra`), and then reduce exactly.
- `extract_surface` of a `PartitionedDataSet` gives the whole's surface, a
  piece each: pieces without ghost cells take the faces between them for
  boundary, so the faces of every piece's surface are matched over the pieces
  by their points and those two pieces hold are left out; the surface's points
  are owned once. On pieces with ghost cells it is each piece's own surface.
- `tack.interop.vtk.algorithms.Probe`, `td.probe` as a VTK algorithm in the
  manner of `vtkProbeFilter`: the source's fields (input port 1,
  `SetSourceConnection`/`SetSourceData`) at the input's points, on the input's
  structure, with `valid`.
- A `device` parameter on every Tack VTK algorithm: the filter runs there, its
  input moved there once (and kept, so the filters after it find it there).
  By default a filter runs where its input's memory is, which for a reader's
  output is the host.
- `vtk_to_dataset(..., copy=False)` and `dataset_to_vtk(..., copy=False)`
  share GPU memory too. A VTK grid whose points, cells or data arrays are
  Viskores arrays on a GPU (as `array_to_vtk` makes, in a VTK with DLPack and
  Viskores' arrays) becomes a dataset on that device sharing them, whatever
  the default device, its other arrays copied there; a dataset on a GPU
  becomes a VTK grid of Viskores arrays over its points, field values,
  offsets and connectivity. They were copied to the host before, so a
  pipeline of VTK and Tack filters on a GPU now hands its data across
  without leaving the device.
- Tack's VTK algorithms (`tack.interop.vtk.algorithms`) take trees of datasets,
  a `vtkMultiBlockDataSet` or a `vtkPartitionedDataSetCollection`, as Exodus,
  IOSS, CGNS, EnSight and OpenFOAM readers give, and give a tree of the same
  shape with its blocks' names and data assembly. Each partitioned dataset in
  the tree, a `vtkMultiPieceDataSet` block too, is filtered whole, so
  `GhostCells` and `GenerateGlobalIds` see all its pieces.
- Every argument of a wrapped Tack filter is a parameter of its VTK algorithm,
  with the filter's default (`Contour(merge_points=False)`, `Streamlines(integrator="rk45",
  max_length=...)`); a required one left unset is named. A class wraps a
  `tack.dataset` function by naming it (`FUNCTION`).
- Fields are chosen as VTK's filters choose them: by name, by name and where VTK
  keeps it (`("p", "cells")`, for a point and a cell array of one name), or with
  `SetInputArrayToProcess`. VTK's implicit functions (`vtkPlane`, `vtkSphere`,
  `vtkBox`, `vtkCylinder`, `vtkPlanes`, `vtkImplicitBoolean`) work where Tack's
  do, such as `Clip`'s `by`.
- Tack's VTK algorithms work over MPI ranks: the filters whose answer needs every
  piece (`GhostCells`, `GenerateGlobalIds`, `MergePieces`, and the reductions)
  take the other ranks' pieces, the ranks taken from VTK's global controller as
  VTK's parallel filters take them (its communicator made mpi4py's by
  `vtkMPI4PyCommunicator`; `pvbatch` sets one), or given as `world=` (an mpi4py
  communicator or a `World`), as with VTK's wheels, which have no MPI.
  `tack.interop.vtk.world()` gives VTK's processes as a `tack.communication.World`.
- A VTK algorithm (`tack.interop.vtk.algorithms`) for every `tack.dataset` filter
  but the summaries and the finite-volume operators that take values on faces,
  named as the filter in CamelCase: `ExtractCells`, `ExtractGeometry`,
  `ExtractExtent`, `Mask`, `Slice`, `BoundaryFaces`, `ExtractEdges`,
  `FeatureEdges`, `Tubes`, `Glyphs`, `CheckWinding`, `MergePoints`, `Triangulate`,
  `AsPolyhedra`, `AsPolygons`, `Shrink`, `Subdivide`, `Transform` (also of a
  `vtkTransform` or `vtkMatrix4x4`), `Warp`, `Cylindrical`, `Spherical`,
  `CellGeometry`, `FaceGeometry`, `CellCenters`, `FaceCenters`, `EdgeLengths`,
  `ImplicitValues`, `CellIds`, `PointIds`, `FlowQuantities`, `Project`,
  `ValuesAtCenters`, `Interpolate`, `Magnitude`, `Dot`, `Cross`, `Composite`,
  `LogValues`, `Convert`, `Jump`, `UpwindFlux`, `Voxelize`, `Advect`,
  `ExtractPoints`, `ThresholdPoints`, `MaskPoints`, `PointNormals`, `Outliers`,
  `VoxelDownsample`, `Deposit`, `SphDensity`, `SphInterpolate`, `HaloCatalog`,
  `Split`, `GhostPoints` and `RemoveGhosts`. Values on faces or edges come out as
  the faces' polygons or the edges' lines with the values as cell data; a grid a
  filter samples onto (`Voxelize`, `Deposit`) may be a `vtkImageData`.
- `ExtractSurface`, `ConnectedComponents`, `Streamlines` and `Advect` take the
  pieces of a `vtkPartitionedDataSet` together, as their Tack filters do: the
  surface leaves out faces the pieces share, regions join across pieces, and
  particles go from piece to piece (and rank to rank). They ran piece by piece.
- `Streamlines` and `Advect` take their seeds from the points of the data on
  input port 1 (`SetSourceConnection`), as `vtkStreamTracer` does, when `seeds`
  is not given; `SphInterpolate` takes its particles there, as
  `vtkSPHInterpolator` does.
- `Statistics`, `Entropy` and `Integrate`, VTK algorithms of `td.statistics`,
  `td.entropy` and `td.integrate` that reduce all their input (every dataset of a
  partitioned dataset or a tree, on every MPI rank) to one answer, as VTK's
  `vtkDescriptiveStatistics` and `vtkIntegrateAttributes` do. `Statistics` and
  `Entropy` give a `vtkTable`, a row per field and a vector field's components one
  by one; `Integrate` gives `vtkIntegrateAttributes`' one vertex at the centroid,
  each field's integral on it and the measure as `Length`, `Area` or `Volume`.
- The VTK algorithms take an AMR hierarchy (`vtkOverlappingAMR`) whole, as one
  `AMRDataSet`, where the Tack filter takes one: `Probe` (as its source),
  `Statistics`, `Entropy`, `Integrate`, `Voxelize` and `GhostCells`, and the new
  `AverageDown` (`td.average_down`, its input left as it was). A region that a
  finer box covers, and a point that a sibling box holds, counts once.
  `GhostCells` and `AverageDown` give a `vtkOverlappingAMR` of the input's boxes,
  and the next AMR filter takes the hierarchy as it was made, ghost layers too.
  The others run box by box, as Tack's filters do on AMR, each level's boxes as
  pieces and the cells finer boxes cover left out: a field filter gives the
  hierarchy with its fields, one that makes a dataset a
  `vtkPartitionedDataSetCollection` with a partitioned dataset per level.
  `GenerateGlobalIds`, `Streamlines` and `Advect` refuse a hierarchy. Over MPI
  ranks each rank holds its own blocks, as VTK spreads a hierarchy.
- `vtk_to_dataset(..., world=)`: the ranks a `vtkOverlappingAMR`'s blocks are
  spread over, a `World` or an mpi4py communicator, by default VTK's (`world()`).
- `RunningStatistics`, `TimeDerivative`, `Advection`, `Pathlines` and
  `Streaklines`, VTK algorithms of Tack's temporal filters, in two modes. In
  situ, each execution takes the step the input holds and gives the output at
  it, stamped with its time; the same step again is not taken twice, and a step
  back in time, a parameter changed or VTK's `NO_PRIOR_TEMPORAL_ACCESS` reset
  starts over. Driving, the filter asks its input for the steps it needs, one
  per pass and never one it has taken: the statistics, particles and paths over
  every step (or those in `time_range`) in one output with no time (time in the
  data: `step` and `time` on the particles, `time` on the paths' points), and
  streaklines and derivatives at the time requested, going on from the steps
  taken. A filter drives where its input has time steps and the pipeline can go
  back in time; `mode=` chooses.
- `Contour` and `Threshold` take a vector field's `component` (the first by
  default, as VTK's filters; `Threshold` also `"magnitude"`), as a view of the
  vector's values: they refused a vector field. Found running VTK's Python tests
  on Tack (`multipleComponentContour.py`).
- `transform(..., vectors=, normals=)`: fields that turn with the map, its linear
  part applied to a vector (a velocity, a displacement) and its inverse transpose
  to a normal, made unit length again so it stays normal to the moved surface.
- `Field(space, values, unit, gradient=)`: a field's world gradient where
  something other than its basis gives it, such as a solver's own
  reconstruction, as a field on the same topology (3 values each for a scalar,
  3k row by row for a vector of k). `world_gradient`, `world_jet` and
  `gradients` read it, so `flow_quantities` and every kernel built on them use
  the solver's derivatives; the Hessian it does not determine is NaN.
- Examples 50-53 in `packages/tack-vis/examples/`: VTK pipelines on real files
  from VTK's test data, each from a reader to a rendered picture (`--png`) and
  checked against VTK's filters. 50 is an Exodus file's collection
  (`disk_out_ref.ex2`, IOSS): an isosurface, streamlines as tubes, the surface.
  51 is an OpenFOAM case's multiblock (`squareBend`): gradients, vorticity, a
  slice, integrals over the inlet and outlet. 52 is an OpenFOAM time series (the
  cavity) through `Pathlines`, driving and in situ. 53 is an Enzo AMR file:
  statistics and the mass over the hierarchy, a slice of each level. The files
  are fetched once by their SHA-512 (`_vtk_data.py`, `vtk_data.sha512`) or read
  from `TACK_VTK_DATA`; each takes `--arch` for the device. The VTK examples
  (47-53) draw their pictures with VTK's views and representations
  (`vtkViewsScivis`) where the VTK has them, VTK master, and with actors and
  mappers on VTK 9.7.1, chosen at run time (`_scene.py`).
- The VTK guide lists every algorithm with VTK's filter that does the same work,
  and has a section each on pieces and trees, MPI ranks, GPUs, summaries, AMR
  hierarchies and time.
- A field argument may be an object whose `on(data)` gives the `Field` on `data`,
  as a field expression is. Over a `PartitionedDataSet`, `statistics`, `entropy`
  and `integrate` call it once a piece, so an expression reduces over pieces as a
  name does.
- Fields evaluated where they are read: a space may declare itself `evaluated`
  (its view computes the field by `value` and `corner_value`; it stores no values),
  `continuous` (its crossings are shared between cells, as `H1`'s are), its
  `default_subdivision` and its `integration_order`, and give `rebind` and
  `stored`. `contour`, `subdivide`, `clip` and `threshold` read such a field
  through its view (`threshold` by its values at the lattice points, not as cell
  data), filters make it again on their output, `integrate` integrates
  it by quadrature to its order, and `statistics`, `entropy`, `owned_values` and
  `dataset_to_vtk` read its stored values. `Space` declares each, with its
  default.
- A publish workflow for tack-core (`.github/workflows/publish.yml`). On a
  release tag it checks the tag is the version, builds the wheel and the sdist,
  `twine check`s them, installs tack-core alone from each, and uploads to PyPI
  with trusted publishing after the `pypi` environment's approval. Run by hand,
  it uploads to TestPyPI, the dry run. `RELEASING.md` has the one-time setup and
  a release's steps. tack-vis is not published; it is installed from the
  repository at the release tag.

### Changed

- **Breaking:** `clip` and `threshold` of a curved geometry or a field above
  order 1 return `PartitionedDataSet([whole, part])`, the cells kept whole and
  the cut part, where both went by the cells' corner values alone (and `clip`
  made curved cells straight). Straight cells with order-1 fields, and
  `threshold` by a cell field, still give one `DataSet`.
- **Breaking:** the cuts above (with `exact=True`, the default) move their
  vertices onto the field's roots: positions and carried values differ from the
  lattice values' interpolation (by about the square of the steps); the cells are
  the same.
- **Breaking:** an array is made only of a type its device supports:
  `tack.empty` and every constructor, copy and `to_device` on it,
  `array_from_ptr` and DLPack imports raise `TypeError` for an `f64` array on
  Metal (or on a Level Zero device without `f64`), where before the array was
  made and the first kernel given it raised.
- **Breaking:** `data.fields` is a `Fields` mapping, not a `dict`. A name two
  fields share is refused rather than read as one of them (`KeyError`, from
  `get` and `pop` too); its keys are the names, and `(name, space)` for a
  shared name, so `dict(data.fields)` keeps every field. A field of another
  topology is refused as it is put in, not when an algorithm reads it.
- **Breaking:** `vtk_to_dataset` reads a point array and a cell array of one
  name as two fields of that name, no longer renaming the cell one to
  `"<name> (cells)"` with a warning; `dataset_to_vtk` writes both under the
  name. Writing two point (or two cell) fields of one name is refused.
- **Breaking:** `project` into `Constant` (or order-0 `L2`) gives each cell
  the source's average over it, its integral divided by the cell's measure,
  by quadrature through the cell's own map, where it took the mean of the
  cell's corner values. The two agree for a linear field on a tetrahedron, a
  triangle or an image's (or rectilinear grid's) boxes, which still take the
  corners' mean; they differ on pyramids (the apex weighs a quarter of a
  linear field's average, not a fifth), on distorted cells, and for fields of
  higher order, whose nodes other than the corners the corner mean never
  read. Polyhedra, polygons and polylines average a point field over the
  pieces they are cut into. `average="nodes"` gives VTK's mean.
- **Breaking:** `extract_surface` puts a surface on the points its faces use,
  renumbered in order, as VTK's geometry filter does, not on all the input's
  points. A surface's point statistics are now the surface's, and the surface
  of pieces matches 1.24M points on the AhmedML mesh, not 45.6M. A 2D mesh
  passed through keeps its points if its cells use them all. The fresh
  boundary's renumbering is kept with it, so a repeated call still costs only
  the carry.
- **Breaking:** the VTK algorithms of points, lines and surfaces give `vtkPolyData`,
  as VTK's filters do, where they gave a `vtkUnstructuredGrid`: `Contour`,
  `Slice`, `SlicePlane`, `ExtractSurface`, `BoundaryFaces`, `ExtractEdges`,
  `FeatureEdges`, `SurfaceNormals`, `Tubes`, `Glyphs`, `Triangulate`,
  `AsPolygons`, the face filters, `Streamlines`, `Advect`, the temporal particle
  filters and the point-cloud filters. They now feed `vtkPolyDataMapper` and VTK's
  polygonal filters (`vtkCleanPolyData`, `vtkStripper`, `vtkDecimatePro`), which
  refused them. `Clip` and `MergePoints` give `vtkPolyData` for a `vtkPolyData`
  input, as `vtkClipPolyData` and `vtkCleanPolyData`, and the filters that keep
  their input's kind (`Warp`, `Transform`, `Probe`, ...) keep a polygonal one. The
  cells go into the polydata's vertices, lines and polygons, shared where they are
  all of one kind. Found running VTK's Python tests on Tack.
- **Breaking:** a kernel over shape-based cells that reads their faces or
  edges (`face_id`, `face_side`, `face_orientation`, `face_position`,
  `edge_id`, `edge_sign`) asks for them:
  `for_each(kernel, data, "cells", ..., incidence="faces")`, or `"edges"`, or
  both, and `domain_view(domain, group, incidence=...)` the same. They are
  derived if they are not yet. A cell view had them whenever the faces or
  edges had been derived, by this filter or any before it, so the same kernel
  over the same cells compiled again once another filter had derived them,
  and the disk cache kept gaining entries from one run of a program to the
  next as the order of its filters changed (about 25 a pass of the quick
  tests, from `contour`, `project` and `subdivide`). A view now has only what
  it asks for; a curved geometry still reads the incidence it needs.
- Averages over each point's cells -- `gradients(at="points")`, the
  cell-to-point `project` and `surface_normals` -- are segmented means over
  the point links (`segment_reduce(..., op="mean", empty="keep")`), the values
  read through the links, not gathered. A point of up to 256 cells sums them
  in the links' order as before, so ordinary meshes give the same bits; a
  point of more is summed by chunks, where one thread walked its links. At
  the apex of 403,200 tetrahedra on an M1 Max's GPU: point gradients 294 to
  5.7 ms, cell to point 271 to 3.4. On the CPU, where one core walks such a
  point quickly, about the same (5M tetrahedra 14.8 to 14.1 ms; the apex 1.7
  to 2.4).
- `tack.interop.vtk` is a package (its names unchanged), with `algorithms`
  beside it.
- A device without f64 (Metal, a Level Zero device without fp64) refuses a
  kernel that computes anything in f64 -- a `tack.f64` cast, an f64 constant,
  local or parameter, from the kernel or a function it inlines -- with a
  `TypeError` naming the cause, once its types are known and before code is
  generated (`refuse_f64`). Metal's code generator refused only f64 casts,
  with a `NotImplementedError`. A CPU posing as such a device, as the dataset
  device tests make one, refuses them too, so those tests find on the CPU what
  only Metal's compiler found before (subdivision's `tack.f64` casts, on the
  higher-order branch).
- Contours and slices of order-2 fields on hexahedra and quads, which were
  refused, are cut on sub-cells; of a curved geometry on them, which were
  linearized at the corners, likewise (`subdivision=1` for the corners).
- Subdividing a scalar field in a tensor basis (contour and slice above order
  2, `subdivide`) reads each cell's values once, not once a lattice point: an
  order-6 `H1` field's `dof_index` for every node at every point was most of
  the time. 32^3 hexahedra, an order-6 contour: Metal 457 to 122 ms, the CPU
  278 to 166 ms; order 3, Metal 31 to 27 ms. The values are the same, to the
  bit.
- Locating a point in a curved cell (the cell locator, `probe`, advection on a
  curved geometry) reads the cell's nodes once, not twice at each Newton
  step through `dof_index`: an order-3 curved probe of 100k points on 16^3
  hexahedra, Metal 177 to 73 ms, the CPU 52 to 28.5 ms. The same cells and
  coordinates, to the bit.
- The MFEM interop copies a positive basis' coefficients (`H1Pos`, and `L2`
  with `BasisType.Positive`) into Tack's Bernstein space of that order, on
  quads, hexahedra, triangles, tetrahedra and wedges, permuted as nodal values
  are; they were evaluated at Gauss-Lobatto (or equispaced) nodes.
  `mfem_grid_function` gives a Bernstein field back in MFEM's positive basis.
- The CPU backend keeps far less memory per compiled kernel: compiled code shares
  JIT engines (32 modules to one) instead of an engine each, the LLVM module is
  dropped once its code is made, and a compiled variant keeps only the launch
  bounds of its IR, not the body. A small kernel's variant went from about
  965 KB to 75 KB, a high-order contour and clip from 51 MB to about 21 MB, and
  the quick test tier's workers peak at 1.5 GB instead of 4.9.
- `transform` keeps an image an image: the affine map is folded into its origin
  and direction (`UniformCoordinates`, any invertible direction, a shear too),
  in double precision, and no position is stored; VTK receives a `vtkImageData`
  with that direction matrix. A singular map still gives stored positions.
- `vtk_to_dataset` converting to `f32` without an `origin` chooses one where
  the data are more than 100 times farther from zero than they are wide: the
  lower corner of VTK's bounds, rounded down, the same for a composite's
  pieces. Far points kept 0.2 m steps on a survey's coordinates; they keep
  their precision. `point_cloud` chooses its origin for `f64` positions by the
  same rule (`automatic_origin`): a small cloud about zero got (-1, -1, -1),
  coarsening its values.
- `contour` gives no surface, lines or vertices in a cell with a NaN among its
  values: the surface stops at the cells holding numbers. A NaN was below
  every level, and a crossing on an edge to one lay at the edge's finite end,
  a point not in the data; VTK's lies at NaN.
- `clip` scans its five kinds of counts one at a time through one buffer and
  lets the cells' counts and offsets go before merging points; `contour` lets
  its cells' counts and offsets, and the per-corner arrays it copied when
  dropping repeated triangles, go before its merge. Clip's peak on 5M
  tetrahedra fell from 609 to 304 MiB of arrays, contour's on Damavand from
  3.47 to 3.19 GiB; the outputs are the same.
- `extract_extent` of a flat image or rectilinear grid (a 2D grid in Tack)
  takes VTK's extent and rates in three axes, as VTK gives a flat image's
  (`(i0, i1, 0, 0, k0, k1)`), the flat axis's ignored; it took one pair per
  axis the grid has only.
- `PointLocator.nearest`, and with it `outliers`, `point_normals` and
  `ghost_points` by `k`, take any number of neighbours: up to 32 are kept in
  private arrays as before, more in the output's rows. `k` was at most 32
  (PyVista's `point_cloud_neighbors.py` asks for 40).
- `H1` and `L2` without a `basis` take the mesh's own family above order 2:
  equispaced where the mesh has triangles, tetrahedra or wedges (the one family
  they have), Gauss-Lobatto otherwise. The default was refused on simplices.
- `extract_edges` of an `L2` geometry draws each cell's own edges on points of
  their own along the cell's map (`subdivision` segments each): its cells need not
  meet. It was refused.
- `integrate` takes point data on polyhedra, polygons and polylines, exactly:
  the values are linear on their sub-tetrahedra, triangles and segments. It
  took cell data alone.
- A curved `L2` geometry on triangles, tetrahedra and wedges (a mesh displaced
  by a DG field, a periodic mesh of order 2 or more) maps, measures and inverts
  its cells as its `H1` twin (`_L2SimplexGeometry`, `_L2WedgeGeometry`, over
  the `_SimplexGeometry` and `_WedgeGeometry` the `H1` ones now share). It was
  refused.
- `subdivide`, and the cuts that subdivide (`contour`, `slice`, `clip`,
  `threshold`), take meshes of several families of one dimension (hexahedra,
  wedges and tetrahedra, as MFEM's mixed Fichera corner): each family is refined
  alone and the parts joined at the lattice points they share. Their cuts'
  crossings are placed linearly on the sub-cells (exact crossings stay a
  one-family feature). They were refused.
- `vtk_to_dataset(dtype=None)` with a default device without `f64` (Metal)
  reads floating-point arrays as `f32`, which such a device's arrays take,
  where it kept VTK's doubles there before.
- Reorganize the Visualization Guide around a gradual data-model explanation
  and complete workflows, with representation chapters for point clouds,
  pieces, AMR and octrees, and optional advanced theory, layouts and references.
  Preserve runnable examples and existing page addresses.
  Open the model with an overview relating mesh and particle representations
  to FE/DG, FV, SPH and planned native FD support.
- The CUDA, HIP, OpenCL and Metal code generators find the types of the
  variables an `if` declares in one walk of its branches. Kernels with many
  nested branches, which inlined device functions produce, generated their
  source in time that grew with the number of variables times the branches'
  size at every level: a spread of tack-vis algorithms' 105 kernels now
  generate in a third of the time, to the same source.
- A polyhedron with no kernel (tangled or inside out, a face whose
  fan folds, no volume) no longer makes `CellLocator`, and with it `probe` and
  advection, refuse the whole mesh: it holds no points, with a warning. A point
  inside it is found in a neighbour it overlaps, or nowhere; a particle reaching
  it stops. The AhmedML volume has 12, wall cells at the nose's corners.
- `contour` and `slice` make a crossing that lies within half an ulp of a
  point (relative to its edge's largest coordinate) that point, as they did
  when the point's value is the level, so every edge's crossing there is one
  point. A plane a hair from mesh points, such as a symmetry plane's points
  at y = 1e-45, had put several points at one position, and on pieces with
  ghost cells `generate_global_ids` of such slices failed. Slivers whose
  points all lie that close to one point have no area and are dropped:
  Viskores' wedge gyroid gives 49 triangles, where Viskores gives 52.
- `threshold` and `threshold_points` keep a value exactly when it lies in
  `[lower, upper]` as given: on an f32 field the bounds are rounded inward to
  f32, since the nearest f32 of a bound may lie just outside it (f32(-0.1) is
  -0.10000000149), which kept values VTK, comparing in double, leaves out (one
  cell of the AhmedML mesh's 18,027,253). f64 and integer fields are unchanged.
- A `PolygonalTopology` matches its edges when something first needs them
  (`faces()`, the facet arrays, `num_faces`), not when it is made, and refuses
  an edge of three or more polygons, or an inconsistent winding, there. Its
  polygons work without the matching: areas, centres, integrals, probes,
  quality and writing take any loops.
- Advection tries the polyhedron a particle was last found in before the
  bins, as it tries solid cells with a reference element: taken where one of
  its sub-tetrahedra holds the point well inside, so the paths are the same
  bit for bit. On the AhmedML mesh's 21.5M polyhedra, 2,000 streamlines take
  0.66 s on the CPU (0.76 before) and 3.0 s on Metal (6.0).
- The documentation site is rewritten and reorganized by Tack's four layers,
  built bottom-up: Core (GPU computing in Python), Communication (distributed
  parallelism), Data (the data model) and Visualization (analysis and
  algorithms), after a front page (What Tack is, use cases, install). Each
  layer opens with a start page and builds without depending on the layers
  above. The reference is gathered by layer with a new glossary that gives VTK,
  ParaView and Viskores counterparts, and the design notes form an addendum.
  Every example on the new pages runs in the docs check. The writing kit
  (style guide, term list, page checker) is in `docs/_writing/`.
- The documentation site's navigation has five tabs, as NumPy's has a few: Get
  started (what Tack is, use cases, install), Guide (Core, Communication, Data
  and Visualization as groups of its sidebar), Reference, Developer's guide and
  Design notes. Pages have breadcrumbs and previous and next links; the header is
  plain, the headings dark, the fonts the system's.
- The documentation's first pages, after an editorial review: the home page
  starts with four paths (write GPU code, analyze a dataset, connect a
  simulation's data, scale across devices or ranks) and is a third shorter;
  the Data, Communication and Visualization introductions are short, their
  longer examples left to the chapters that teach them; Visualization starts
  with a Tack-only example; pages open in their own words, not a set formula
  (the style guide's first rule, which `check_page.py` now checks); and the
  Visualization pages show their results, with a new picture of contour, clip
  and threshold of one ball (`docs/gallery/extract.py`) and the gallery's
  images beside probing, streamlines, pathlines, pieces and higher-order cuts.
- A topology's cell locator serves a geometry of the same positions: where it
  does not serve a dataset, `probe` and advection compare the positions on the
  device and, if they are equal, keep its bins (the search made again from the
  new positions). `Pathlines`, `Streaklines` and `Advection` keep a copy of the
  previous step, whose positions are the step's, and built every mesh's
  locator twice: half of a step's time on the dam break's adaptive meshes
  (1.36 s to about 0.6 for 100 particles over 30 meshes). A simulation that
  hands over new arrays of unchanged positions each step no longer rebuilds
  them either.
- The repository moved to Kitware's GitHub organization: it is
  `https://github.com/Kitware/tack`, and the documentation is at
  `https://kitware.github.io/tack/`. The packages' metadata, the READMEs and
  the documentation link there; GitHub redirects the repository's old
  address, but not the documentation's.
- Tack's VTK algorithms (`tack.interop.vtk.algorithms`) keep one Tack dataset
  per VTK dataset, by its address, modification time and time step, dropped
  when the VTK dataset is deleted. Filters that read one input, or one filter
  run again after a parameter changed, share it and what it derives from its
  topology (locators, faces, point links), which every run rebuilt before: a
  second `Streamlines` on 2.6M tetrahedra took 13-86 ms against 380-540. A
  field filter's (`Gradients`, `Elevation`, `CellQuality`) output keeps its
  input's topology, so the filter after it does too.
- The VTK interop shares GPU memory with any VTK that holds device arrays, not
  only one built with Viskores: VTK's `vtkDeviceDataArray` (CUDA and Level Zero,
  needing only the driver) serves as well. Nothing in the interop changed; its
  device tests now run wherever such a VTK is found.
- `tack.interop.vtk.init_level_zero()` works with a VTK that has its own Level
  Zero context, without Viskores; its error message and docs no longer say
  Viskores on Kokkos' SYCL backend is required.
- Handing an unstructured grid between VTK and Tack costs less:
  `vtk_to_dataset` finds the cell types a grid holds without sorting them
  (`np.unique` was 8 ms for 1.3M cells), and `dataset_to_vtk` sets the cells
  without VTK's scan of every type for a polyhedron (4 ms). Together most of
  what a GPU pipeline spent between its filters.
- Every source file opens with an SPDX copyright and license header, as in
  ParaView and VTK (`tools/spdx_headers.py`, checked in CI's lint job).
  `tack.dataset`'s clip case tables now credit their authors: they are VisIt's
  (Lawrence Livermore National Security, BSD-3-Clause), which VTK keeps in
  `vtkTableBasedClipCases.h` and Viskores copied, not Viskores' own.
- CUDA writes the multiply-add contractions the language contract permits as
  explicit fused multiply-adds and compiles with `--fmad=false`. NVRTC's own
  fusion depended on the code around an expression, so two kernels inlining
  the same function could differ in the last bit (streamlines with and without
  the locator's previous-cell try, on curved wedges).
- Template methods take only the synthetic parameters (a template's arrays and
  runtime scalars) they read, themselves or through the methods they call. Every
  method took its whole tree's, and each inlined call copied them all: a view
  holding other views carried theirs into every call.
- A template's methods are resolved when the kernel reaches them, not all of
  them for every kernel. A method the kernel never calls costs nothing and is
  not required to resolve.
- `world_gradient` and `world_jet` ask the field's view, whose default
  (on every field view whose basis gives the derivatives) differentiates its basis as
  before. A space that computes its values can give its own.
- The temporal VTK algorithms (`RunningStatistics`, `TimeDerivative`,
  `Advection`, `Pathlines`, `Streaklines`) take a `vtkMultiBlockDataSet` or a
  `vtkPartitionedDataSetCollection`, which they refused, in both modes: each
  partitioned dataset and each block that is one dataset has a Tack filter of its
  own, and the output is a tree of the same shape, at the step's time (the
  tree's, or its datasets'). The OpenFOAM, IOSS and CGNS readers give trees.
- `integrate` takes a mapping of names to fields (names or `Field`s), each
  integral under its name, as `statistics` and `entropy` take a `Field`.
- Filters carry a field's given gradient with it, so `gradients` of a filter's
  output reads the gradient the field was given, not one from its basis.
  A field whose gradient a filter cannot carry is left out with it.
- `cell_geometry`, `face_geometry` and `surface_normals` keep what they compute
  with the mesh: computed once a mesh, shared by every dataset on the same
  geometry, and made again when the geometry changes (another geometry, or
  positions Tack has written). `extract_surface` of the same mesh gives the same
  surface geometry, so the surface's normals are computed once over many steps
  that change only fields.
- CI's wheel job also builds tack-core's sdist, checks both with `twine check`,
  and installs tack-core alone, from the wheel and from the sdist
  (`tools/check_core_install.py`). Asking for `tack.dataset` without tack-vis
  names the command that installs it from the repository.
- tack-core is on PyPI: the install docs, the READMEs (tack-core's is its PyPI
  page) and the messages for a missing backend give `pip install
  "tack-core[...]"` instead of an install from a clone, and tack-vis's install
  from the repository at the release tag (`docs/install.md`). A clone's `uv
  sync` stays, for working on Tack.

### Fixed

- `contour`, `slice` and `clip` interpolate each crossing from its edge's end
  first by position (x, then y, then z; the lower id only between coincident
  points), not from its lower point id. Ids are a piece's own, so two pieces
  holding an edge could interpolate its crossing from opposite ends: with an
  f32 field the copies differed in their last bits (up to 4e-8 on a wavelet),
  and `merge_pieces` of ghosted unstructured pieces' isosurfaces failed
  ("marked duplicates but no piece holds them unmarked") for want of a
  tolerance. Every piece now computes a shared crossing, and the values
  carried there, to the same bits, from the same end ghost marks already
  give it its owner by; within one mesh, as before, every cell sharing an
  edge agrees. Output points and their order are unchanged; their last bits
  may differ from before.
- Gradients in f32 on thin cells. A cell's Jacobian summed its corners'
  absolute positions, and its field's parametric derivative its values: on a
  cell 250 times longer than thick, near coordinates of 1 and a pressure of
  1e5, those sums lost the short edge and the small differences to rounding.
  The ONERA M6's boundary-layer wedges' f32 pressure gradients were off by
  more than 0.1% in 6,569,170 cells (14%), by more than 100% in 380, the worst
  93 times. Both are now formed from differences from the first corner and
  value (the shapes' gradients sum to zero, so the result is the same),
  exact in f32: 31 cells off by more than 0.1%, the worst 0.32%, as in f64
  (1 cell, 0.33%), at the same cost; on Metal, which has no f64, 26 cells,
  0.37%. Cell quality's Jacobian measures and order-2 geometry follow the
  same rule; locating points already did.
- Points in curved cells are located in the cells as they are. The cell
  locator bounded and inverted every cell by its corners, whatever the
  geometry's order, so a point where a curved cell bulges past its corners
  (at order 2 as well) was missed or found in the wrong cell. A cell is now
  bounded by the box of its Bernstein control points, which hold it, and
  inverted by Newton's method on its own map.
- Filters that keep whole cells (`extract_cells`, `threshold`, `split`,
  `remove_ghosts`, and any filter written with `carry` and a `Same` or
  `Take` cell map) keep an `H1` field above order 1, and a curved
  geometry, at its order and nodes. They carried the corner values only, so
  a filter's piece of a curved or high-order mesh came out straight and
  linear. Each output cell's nodes are its source cell's, to the bit.
- `CellLocator.cell_bins()`, `leaves()` and `cell_neighbors()` build their
  tables on the dataset's device. Called directly on a dataset on a GPU (not
  through `find`, `probe` or advection, which set the device), they allocated
  on the default device and their first kernel refused arrays on two devices.
- An RK45 particle at a wall the flow turns back from ends there
  (`SPATIAL_BOUNDS`), as `vtkStreamTracer`'s RK45 ends it out of its domain.
  Its stage points swung past the wall at every step size, or its stages'
  directions cancelled, so each step's bisection pushed it out by about an
  ulp, still inside, and it crawled until its steps ran out: 86 of 20,000
  aneurysm paths, on every device, and on a GPU those few threads set every
  launch's length (20,000 seeds took 6.4 s on CUDA, 14 s on HIP and 44 s on
  Level Zero, against 1.4-2.9 s on the CPU). A push out that moves the
  particle less than `min_step` now ends it; on the aneurysm 2 paths still
  run to the step limit, and the CPU run takes 1.4 s instead of 2.3 s. Euler
  and RK4 are unchanged.
- `interpolate` places a shared node on a quad face, and an order-3 or higher
  node on an edge, in a frame the entity's corners' positions fix (an edge from
  its first point by position, a face from its first corner toward the first
  of that corner's neighbours), not in its stored point order, which follows a
  piece's point ids. Pieces holding a face computed its nodes from different
  corners, so a curved geometry or field interpolated on each piece differed
  from the whole's in the last bits there (up to 2e-7 in f32). Values may
  differ from before in their last bits.
- A filter carrying an `L2` field of a tensor basis kept its order but not its
  nodes, so a Gauss-Legendre field came out labelled Gauss-Lobatto, another
  basis, and read wrongly.
- `interpolate` on a curved geometry numbered otherwise than the target space
  (by a table) placed nodes at the wrong positions.
- Order-2 Bezier cells read from VTK took the order-2 Lagrange view's node
  numbering for a Bernstein space's.
- An order-2 `H1` field on a geometry curved to order 3 or more had its edge and
  face nodes at the mean of their corners, off the curved cells; they are now
  on the geometry's traces. Node views of a curved geometry and an `L2` field of
  another order no longer fail to build (both defined `ORDER`).
- `contour` with `subdivision=` on polyhedra, polygons or polylines sent them to
  the sub-cell path, which failed; they are cut as they are, as `clip` and
  `threshold` cut them.
- `as_polygons` of a polygonal dataset failed; it comes back as it is.
- `ghost_cells` and `merge_pieces` made a curved geometry straight with no word
  (they move positions at the points); they refuse it.
- `cell_quality` scores a degenerate cell (an edge of no length, or no area or
  volume) 0 in its scaled Jacobian and Jacobian ratio, as Verdict does for
  triangles, quads and tetrahedra; it gave the largest float, of either sign.
- `cell_quality` measures a polygonal topology's triangles and quads as those
  shapes, not by fans from their centers (an equilateral triangle scored 0.577
  beside a pentagon), and a polygon's fan triangles' condition number is
  Verdict's triangle's; it was twice that.
- `contour` and `slice` of a 2D mesh match its cells' sides by their points
  instead of through the topology's faces, so a surface with an edge of three
  or more cells (PyVista's action figure, room and sextant) gets its lines,
  meeting there, where they refused the whole mesh. The topology keeps the
  matched sides for the next call.
- `contour` of a 2D mesh finds each side's crossings from its two points, not
  from the side's first cell, so a crossing every line needs is always made.
- `face_geometry`, `upwind_flux`, `net_flux`, `divergence`, `perot` and
  `surface_normals` take a surface with lines among its polygons (EnSight's
  and VTK's surfaces carry them), the lines having no edges of the
  surface's: zero flux and a zero normal. `net_flux`, `divergence` and
  `perot` of a mesh of vertices are zero. Both failed inside the kernels
  (`no @tack.func method 'side_position'`, `'num_faces'`).
- `outliers`' docstring no longer says its statistical test is VTK's: it marks
  points far above the mean distance only, as PCL; `vtkStatisticalOutlierRemoval`
  also removes those far below it.
- `probe` rounds an integer field evaluated in a cell to the nearest integer,
  halves away from zero, as VTK's probe does; it truncated (a `u8` field of 0
  and 1 probed three quarters of the way came out 0).
- `shrink` keeps integer point fields, each cell's corners copied as they are;
  it failed on them. `project` from order-1 `H1` into order-1 `L2`, a copy,
  takes integers.
- `dataset_to_vtk` makes normals VTK's active normals, which its renderers and
  filters read: a 3-vector point field named `Normals` or `normals`, and a
  3-vector cell field named `Normals`, `Normals (cells)` or `cell_normals`. A
  surface's normals carried through a Tack filter (a clip of a VTK surface
  with normals) came back as a plain array, and VTK drew the surface flat.
- `vtk_to_dataset` reads a `vtkBitArray` as one u8 value (0 or 1) per point or
  cell; it came as its packed bytes, eight values to one, and conversion
  failed (`kitchen.vtk`'s `hpor`).
- `vtk_to_dataset` reads hexahedra with corners repeated as finite-element
  files write other shapes (Tecplot's bricks, degenerate Exodus and Patran
  hexahedra) as those shapes: wedges, pyramids and tetrahedra. As hexahedra,
  each one's collapsed face, an edge, was claimed by every cell around it, and
  every face-based algorithm refused the mesh.
- `vtk_to_dataset` reads a polydata's poly vertices (a vert cell of several
  points, as `vtkTableToPolyData` and `vtkParticleReader` make) as one vertex
  per point, each taking the poly vertex's cell data, as strips become
  triangles; they were refused. A VTK cell type Tack has no shape for is
  refused as such, naming the higher-order cells that convert only when the
  type is one.
- `vtk_to_dataset` refuses VTK cells it reads neither as shapes nor as
  higher-order spaces (biquadratic and triquadratic cells, quadratic
  pyramids), naming them by VTK's class names; they came across as cells no
  algorithm took, every one failing with "are not linear shapes".
- `warp` takes a direction or `scale_by` given as values on points when the
  geometry is order-1 `H1`, as `surface_normals` gives its normals; it
  refused them as "not in the geometry's space".
- `cell_geometry` of a curved geometry (`H1` above order 1, a curved `L2` one)
  measured each cell by its corners; it measures it by quadrature through its
  map (a sphere's area was 3-6% short).
- `jump` of a field of order 2 or more took each side's value from the face's
  corner values (`traces`) and their linear mean; it evaluates each side's own
  basis at the face's center (DG jumps against MFEM's to 1e-16).
- `mfem_to_dataset` of a 2D curved mesh refused its two-component nodes; they
  get z = 0.
- An `L2` geometry above order 1 (a mesh displaced by a DG field, a periodic
  mesh of order 2) was read as order-1 corners, each cell's first values, so
  probes, centres and everything else were wrong. On tensor shapes it is now a
  curved geometry (`views._L2TensorGeometry`, either layout); on simplices and
  wedges it is refused.
- `subdivide` (and the cuts that subdivide) of an `L2` geometry shared lattice
  points through the topology, so on a periodic mesh, whose seam's cells share
  points but sit apart, sub-cells stretched across the mesh; each cell's lattice
  points are now its own.
- `contour` of a high-order field on a mesh subdivision does not take (cells of
  two families, as MFEM's mixed Fichera corner; pyramids) is refused with that
  reason; the message spoke of corner values.
- `vtk_to_dataset` of a grid of VTK cell types Tack has no shape for (the
  biquadratic quad, among others) built a topology it could not read, with no
  word; they are read (above) or refused, naming the types. Pyramids above
  order 1 and quadratic polygons are refused.
- A serendipity field of integers truncated its face centres (computed in its
  own type); they are computed in floating point, as its basis is evaluated.
- A curved cell is inverted to rounding: two Newton steps past VTK's stopping
  rule (`CURVED_NEWTON_POLISH`). Probes of MFEM's curved meshes agreed with MFEM
  to 3e-9; now to 1e-15.
- `extract_edges(..., subdivision=)` placed a curved edge's points at f32
  fractions of it (`(k + 0.0) / level`), about 1e-8 off the curve in f64.
- Filters no longer store an image's or a rectilinear grid's positions, or
  an implicit field's values, to read them once: `point_elevation`,
  `implicit_values` (so `slice`, `clip`, `extract_geometry` and
  `extract_points` by an implicit function), `carry` of taken and
  interpolated points (`threshold`, `clip`, `extract_cells`, `mask`, ...),
  `warp`, `transform`, `cylindrical`/`spherical`, `glyphs`, `tubes`,
  `feature_edges`, `composite`, the 1D and 2D contours, clipping polyhedra,
  advection and `voxelize` pass them to their kernels as they are. A 256³
  image's positions are 200 MB; `warp` without `scale_by` also no longer
  stores an array of ones. On that image `point_elevation`, `implicit_values`
  and `warp` run 2-5 times faster.
- Reading polyhedra from VTK, and any cells with
  `vtk_to_dataset(..., polyhedral=True)`, is about ten times faster:
  `PolyhedralTopology.from_cell_faces` matches the faces cells share in
  whole-array steps instead of a Python loop over them, and the reader takes a
  cell type's faces from its class once rather than calling `GetCell` per cell
  (512,000 polyhedra: 7.2 s to 0.7 s). The topology is the same, array for array.
- Triangle strips read about four times faster: their cell data is copied to
  the triangles by a NumPy gather per array, not a `vtkIdList` filled one id
  at a time (2M triangles: 319 ms to 74 ms).
- `generate_global_ids` over a `PartitionedDataSet` with a polyhedral piece:
  it read faces' `rows`, which a polyhedral topology's faces (of any size) do
  not have. It now takes their boundary points from the face lists.
- Reading hexahedra no longer sorts every hexahedron's corners to look for
  collapsed ones (Tecplot's bricks, degenerate Exodus hexahedra): only those
  with a collapsed edge are sorted, about 1.2 s and 2.6 GB less on 21M
  hexahedra.
- `L2.indices()` and `L2.blocks(values)` refuse cells holding different numbers
  of values (a mixed mesh, an order per cell), naming `space.offsets`. They cut
  the values into rows of the average size, so a tetrahedron and a hexahedron
  at order 1 read as two cells of 6 values each, raising only when the total did
  not divide evenly.
- Reading as polyhedra (`polyhedral=True`) is about four times faster again:
  triangles and quads are matched by their sorted points packed into two
  64-bit keys and Tack's parallel radix sort, where `np.lexsort` took 68 s on
  128M quads, and their face lists are built by columns, a few million cells at
  a time, with 32-bit point ids where they fit. 21M hexahedra, 67K wedges and
  508K polyhedra: 27 s and 19 GB, with `orient=True` for its wedges, which also
  no longer falls back to the cell-by-cell walk when the shaped cells were what
  disagreed.
- A `vtkCharArray` comes across as `u8` where the platform's `char` is unsigned (Linux on Arm) and as `i8` elsewhere; it was always read as `i8`, which turned values above 127 negative on Arm.
- `generate_global_ids` matches every point, not only boundary points, when
  pieces have ghost cells but no point marks.
- Lines cut from surfaces interpolate each crossing from its side's end first
  by position, not from its lower point id, so pieces sharing a side compute
  it to the same bits.
- Streamlines, advection and probes over pieces bounded each piece by its
  positions' box taken on the host on every call: a second of each call over
  the AhmedML mesh's 21M-cell piece, half the split reading's streamline time.
  The box is now the piece's locator's, found once when the locator is made;
  the split reading's 2,000 streamlines take 0.75 s, as the whole reading's
  do, where they took 1.71.
- Point gradients (`gradients(..., at="points")`) were NaN at a corner where a
  cell's Jacobian is singular, as where two of a hexahedron's faces meet flat
  (17 points of the AhmedML mesh). Such a cell has no gradient there -- it
  grows without bound toward the corner -- and is now left out of that point's
  average, the others' mean taken; VTK leaves its inverse Jacobian unset there,
  and Viskores counts the cell but adds nothing.
- A CPU device without f64 (the tests' stand-in for Metal) loaded kernels a
  CPU with f64 had stored in the disk cache, and so ran an f64 kernel Metal
  refuses; its cache target now says it has no f64. Point gradients at
  singular corners are also run on it in the tests, which is where #268's f64
  cast would have been caught.
- Filters over pieces whose points are marked copies stalled now and then: the
  check whether any mark has a bit (`ghosts._has`, on every `carry`) raised one
  atomic flag for every marked element, so the CPU kept it serial (0.14 s over
  the AhmedML mesh's 22.8M points) and its rechecks fanned it out at 1-1.2 s.
  It is now a flag per chunk of the marks, without atomics: a slice of the
  polyhedra piece takes 0.012 s every call, where it took 0.15 s with spikes
  of 1.2.
- `contour` of a polyhedral mesh raised where the iso-surface met an edge
  three or more times: a point within half an ulp of the level is a vertex
  on some of its edges and not others, so several faces trace one iso-edge.
  The AhmedML mesh's Q = 1, 2 and 5 contours (2.0M, 1.5M and 0.8M polygons)
  are made now; only their faces are refused.
- Advection stopped a particle at zero velocity wherever its speed was below
  3.5e-4 in the data's unit (a squared speed at the float's machine epsilon),
  so every streamline of the SciVis 2021 mantle (1e-9 m/s) ended at its seed.
  Zero velocity is now a squared speed at or below 1e-37; slower flows stop by
  `terminal_speed`, as vtkStreamTracer's do. The mantle's 500 streamlines are
  VTK's (300,500 points, lengths within 1.3e-5).
- A step leaving the mesh is bisected to the boundary, and the bisection's
  bracket was absolute (ten machine epsilons): on long steps in f32 (an ulp of
  50 is 3.8e-6) it never got that small, and the particle bisected forever.
  The bisection is now bounded at 64 halvings.
- AMR hierarchies of thousands of boxes took minutes or more: making an
  `AMRDataSet` compared every box with every other in Python, and
  `ghost_cells` and `average_down` built their sources box by box and filled
  through a plan with a pair for every two boxes that touch (AMR-Wind's 6,436
  boxes: 326,480 launches a fill). The marks are computed a level at a time,
  boxes that meet are found by a sort and sweep, the sources by kernels over a
  level's boxes at once, and a fill runs once per device's boxes. On AMR-Wind's
  levels 0 and 1 (6,436 boxes, CPU, VTK master), making the dataset from VTK's
  went from 129 s to 3.2, `ghost_cells` from more than 40 minutes to 7.6 s,
  and `average_down` from not finishing to 2.9 s, with every value the same to
  the bit; all five levels (15,598 boxes, 183M cells) take 8.2, 30 and 11 s.
- `vtk_to_dataset` of a `vtkOverlappingAMR` gives every box the hierarchy's
  origin, not one chosen from the box's own bounds, and places each box as it
  converts it.
- `average_down` of an image box weighs its cells by the product of its spacing,
  every cell alike, not by each cell's measure from its corners, which could
  differ in the last bits.
- `ghost_cells` of an `AMRDataSet` looks for sources only in each grown box's
  shell (its ghost cells, and its own points on its boundary), not in every
  cell and point it holds, places each box's values on the device, and looks
  each box's device up once: on AMR-Wind's levels 0 and 1 (6,436 boxes) it
  went from 7.6 s to 3.6, with every value the same to the bit, and on all
  five levels (15,598 boxes, 183M cells) it takes 20 s, `average_down` 9.
- A filter's output from pieces without ghost cells (`split` without ghost
  layers, the split reading of shaped cells and polyhedra) refused every
  reduction, though only the points it makes on shared edges and faces have no
  owner: each cell it made from one input cell keeps that cell's. The loss is now
  recorded per kind, so `integrate` (a contour's area) and statistics of cell
  fields work on such outputs, and point reductions still refuse.
  `extract_surface`'s cells, faces the pieces share, lose their marks too.
  Setting a dataset's `ghosts` to `None` clears the record, as other marks did.
- `integrate` and `cell_geometry` measure a non-planar polygon as its fan of
  triangles from its points' mean, Tack's model of a polygon and the area VTK
  reports, and a quad as its bilinear patch (by 4 x 4 Gauss points, within
  rounding unless it is twisted far out of plane), not either as its
  projection onto its Newell normal, which is smaller. Tack's contours of
  hexahedra are polygons, and a wrinkled one measured 1% small (the dam
  break's free surface; the Ford DrivAer's wake 7.5e-4). Both are signed by
  the normal, so a planar polygon whose mean lies outside it, or a planar quad
  that is not convex, still measures exactly. VTK measures a quad as its
  projection.
- A particle whose step leaves the mesh is pushed just outside it (an Euler
  step from the last point inside by the bisection's bracket, grown until no
  cell holds it), not by the distance to the boundary again, which Viskores'
  `SmallStep` does and Tack did, counting the distance once in the particle's
  time and path length. On one mesh it lengthened only a path's last segment;
  between pieces without ghost cells the path went on from there: on the Ford
  DrivAer's split pieces 59 of 200 streamlines ran past `max_length` (by up to
  0.31 m over 23 hand-offs) and one was pushed out of the mesh across a thin
  layer, 0.48 m into its 13.84.
- RK45's end at a wall, a push out that moves the particle less than
  `min_step`, measured the push after it was taken (always 0), so every RK45
  push out that stayed in the mesh ended the particle; it measures the move.
- Over pieces without ghost cells, a particle pushed out of one piece into a
  sliver no piece holds -- pieces whose shared faces are not one surface, a
  hexahedron's bilinear face against a polyhedron's fanned one in a
  `polyhedral="split"` reading -- goes on from where the push would have ended
  going the way to the boundary again, if a piece holds that, with the move in
  its time, length and stretch; it stopped there, outside.
- `dataset_to_vtk(..., copy=False)` of a dataset made by
  `vtk_to_dataset(..., copy=False)` from GPU arrays hands VTK back the arrays
  it shared, which DLPack cannot (they are read-only to Tack): a filter that
  keeps its input's points, as `Probe` does, failed.
- `vtk_to_dataset(..., copy=False)` makes a dataset from a host grid on the CPU
  whatever the default device: with a GPU the default (as
  `tack.interop.vtk.init_level_zero()` makes Level Zero), what it copied beside
  the shared arrays -- implicit cell types and offsets, converted values -- was
  made on the GPU, so filters gave empty outputs or refused arrays on two devices.
- A VTK algorithm whose filter raised kept the exception with its traceback,
  whose frames held the algorithm and VTK's pipeline information in a cycle;
  Python's collector crashed freeing it. The exception is kept without its
  traceback, printed as VTK printed it.
- A filter over pieces (`GenerateGlobalIds`, `GhostCells`, `MergePieces`) on a
  multiblock's `vtkMultiPieceDataSet` block ran on each piece alone, and gave
  empty pieces.
- `MergePieces` over MPI ranks gave the ranks other than the root nothing to
  convert, and failed there; they get an empty dataset of the output's type.
- Python crashed in its garbage collector when it freed, in a reference cycle (a
  traceback's frames, say), a VTK dataset one of Tack's VTK algorithms had read: the
  `DeleteEvent` observer that dropped the dataset's cache entry, referenced only by
  the dataset, was cleared as garbage before the deletion called it (a VTK bug, in
  9.7.1 and master). The cache entry is now an attribute of the dataset's Python
  wrapper, which VTK keeps while the dataset lives and frees with it: no observer.
  An entry is converted again when the dataset's modification time or time step
  changes; after changing a dataset in place, call `Modified()`, as VTK's own
  filters need.
- A `vtkOverlappingAMR` given to a VTK algorithm gave an empty hierarchy, with no
  error: VTK makes a hierarchy a partitioned collection, which they filtered as a
  tree into an output of the hierarchy's type that could not hold it.
- `vtk_to_dataset` of a `vtkOverlappingAMR` spread over MPI ranks marked each
  rank's boxes as if its own were the whole hierarchy; it now converts in VTK's
  world unless given one.
- `Advection` (and `Pathlines`, `Streaklines`) over pieces on MPI ranks waited
  forever where a rank had no particles, released particles at other steps, or
  saw its particles stop at another substep: the probes are collective, and a
  rank skipped them. Every rank now takes part in each.
- The VTK algorithms that add fields (`Elevation`, `Gradients`, `Project`, ... and
  `RunningStatistics`, `TimeDerivative`) on VTK's higher-order cells wrote Tack's
  values straight into VTK's point data: an order-2 field's in Tack's node order,
  not VTK's (`Elevation` came out scrambled, and its `Gradients` wrong), and values
  on Tack's points, the corners, for every VTK point, edge middles included (a
  1193-point grid got an array of 1193 tuples of 1227 components). Each VTK point
  now takes its node's value: the field put into the space the import made
  (`project`, exact for an `H1` field of that order or below and for values on the
  corners), its nodes matched to VTK's points as the import matched them. An
  integer field on the points is refused there, having no value at an edge's
  middle. Found running VTK's `quadraticQuadDeriv.py` on Tack.
- The VTK algorithms' outputs had no active attributes: Tack keeps no roles, so the
  arrays it carried came back plain, and mappers colored by nothing, glyphs and
  lighting found no vectors or normals (VTK's tests rendered white). An output now
  keeps its input's active scalars, vectors, normals, texture coordinates and
  tensors where it carries arrays of those names (`Probe` its source's), as VTK's
  filters pass them; `Elevation` and `ImplicitValues` make their array the active
  scalars, `SurfaceNormals`, `PointNormals` and `Tubes` theirs the active normals
  (a tube took the lines' carried `Normals` before, and was lit wrongly). Found
  running VTK's Python tests on Tack.
- `triangulate` cut a polygon into a fan from its first corner, which covers a
  concave polygon twice in part and leaves part uncovered (a U-shaped polygon of
  area 6 gave triangles of area 9). It now cuts ears in the polygon's plane, as
  VTK's `vtkTriangleFilter`: `n - 2` triangles that cover the polygon once, each
  wound as it is. Found running VTK's Python tests on Tack (the concave polygons of
  `vtkDiscreteFlyingEdges2D` and `vtkImageToPolyDataFilter`).
- The VTK algorithm `Transform` carried the input's vectors and normals as they
  were, so after a rotation they pointed the old way (and normals lit a moved
  surface wrongly). It now transforms the input's active vectors and normals, on
  points and cells, as VTK's `vtkTransformFilter` does, unless `vectors` or
  `normals` say otherwise. Found running VTK's Python tests on Tack.
- Slicing or contouring serendipity cells (quad8, hex20, wedge15) that carry a
  64-bit integer point field failed: the field's full values asked a device for
  what a backend has. Found running VTK's `cutter.py` on Tack (`cuttertest.vtu`'s
  node numbers).
- `extract_extent` of a flat curvilinear grid refused VTK's three-axis extent and
  rates (VTK always writes three; a grid's flat axis is the one whose pair is
  `(0, 0)`). Found running VTK's `NetCDFCF2DBounds.py` on Tack.
- A template method may pass one template it holds to a method of another
  (`self.values.scaled(self.mesh, i)`): the callee is resolved when that call is
  reached. It raised a `KeyError` for the callee's resolved name.
- `vtk_to_dataset` marks a cell `HIDDENCELL` where one of its points is
  `HIDDENPOINT`: VTK sees a cell only where it and all its points are visible (a
  structured grid blanked by points), and Tack's filters, which leave out hidden
  cells, contoured and cut through blanked points. The point marks are kept.
- The VTK algorithm `ConnectedComponents` refused its own output on `vtkPolyData`
  (it gave an unstructured grid where it promised its input's type): its arrays go
  onto a copy of the input, as VTK's `vtkConnectivityFilter` keeps it.
- `project` of a polygon (polyhedron, polyline) with no points gave `inf`, having
  read past the cell: it is `NaN`, the cell having no value. These three found
  running VTK's Python tests on Tack.
- `vtk_to_dataset` of a grid of VTK's higher-order cells keeps a point array and a
  cell array of one name as two fields, as on linear cells; the cell array
  replaced the point array (`CylinderQuadratic.vtk`'s `velocity` and `pressure`).
  Found by `test_vtk_higher_order.py` with `TACK_VTK_DATA`, which CI does not
  set; it failed since fields by name and space (0.7). That test also compared
  Tack's measure of a bent quad, a fan from its mean since 0.7, with VTK's two
  triangles; it now measures both surfaces with `vtkCellSizeFilter`.
- A VTK algorithm given a tree with a partitioned dataset of no partitions on
  any rank (the node and side sets an IOSS reader leaves unread) gives it an
  empty one without running the filter. `Streamlines` and `Advect` failed on
  them ("a rank with no pieces has no piece to start its seeds from").
- Box by box over an AMR hierarchy, the VTK algorithms that give `vtkPolyData`
  (`SlicePlane`, `Contour`, ...) give it for each box too, where they gave
  `vtkUnstructuredGrid`.
- `dataset_to_vtk` gives an image, rectilinear or structured grid that is part of
  a larger one (`StructuredTopology.offset`, as `split` makes it) its extent
  there, its points where they were. Every piece's extent started at 0, so
  pieces handed to VTK did not tile their whole, and `GhostCells` refused them.
- `MergePieces` of polygonal pieces (a contour's, a surface's) gives `vtkPolyData`;
  since #353 it declared a `vtkPolyData` output, made an unstructured grid and
  failed, which broke example 48. Examples 47 and 49 read and wrote the
  polydata outputs as unstructured grids, and failed too.
- Over MPI, `tack.communication.sort_by_key` and the matching behind
  `generate_global_ids`, `ghost_cells` and `merge_pieces` hung when one rank held
  every block and the others none: that rank took a shortcut without messages
  and the others waited in an exchange. Every rank now takes the same path.
- Over MPI, the algorithms over pieces take ranks that hold no pieces, as a reader
  that gives some ranks nothing makes them. Each rank must make the same calls
  that communicate, so a rank must not decide from its own pieces alone what the
  others decide from theirs:
  - `probe` of a `PartitionedDataSet` or an AMR hierarchy, `streamlines` and
    `advect` over pieces, `voxelize` of pieces or a hierarchy, and `Advection`
    (and the path filters) over pieces refused or failed on a rank without
    pieces ("needs pieces on this rank to start from", "a rank with no pieces has
    no piece to start its seeds from"), leaving the others waiting. Such a rank
    now takes part through a piece of nothing (`_stand_in`): its points and seeds
    go to the pieces that hold them, and the answers come back.
  - `update_ghosts`, `extract_surface` of pieces, AMR `ghost_cells` and
    `average_down`, `integrate` over pieces, and the VTK `Statistics`, `Entropy`
    and `Integrate` algorithms chose their fields (or a shortcut) from this rank's
    pieces; a rank without any now gets the others' choice.
  - `merge_pieces` onto a root rank without pieces refused; the merge is made on
    the first rank that has pieces and moved to the root.
  - `test_empty_ranks.py` runs every algorithm over pieces, AMR hierarchies,
    octrees, advection and the VTK algorithms over pieces on 3 ranks, with the
    pieces on one, two or none of them, against the same pieces in one process.
- Debug mode on HIP: the bounds check helper `__tack_in_bounds__` is
  `__noinline__`. hipRTC (ROCm 7.1.1) at `-O2` and above miscompiled it
  inlined, and five `test_high_order.py` cases reported in-bounds accesses out
  of bounds, with 0.5f's bits in an index's high half. CUDA's stays inline.
- Level Zero: every kernel requires sub-groups of 16
  (`intel_reqd_sub_group_size(16)`), so IGC never compiles one SIMD32. On a
  Max 1100 (IGC 2.7.11, compute runtime 25.05) IGC's SIMD32 code for five
  tack-vis kernels was wrong: crossings of high-order contours and slices
  (in f32, and with 64-bit ids), a simplex subdivision's lattice points, and
  AMR `ghost_cells`' marks. Their SIMD16 code, fully optimized, is right.
  Compute runtime 26.40 does not fix it.
- Examples 50-53 on Level Zero make their device in VTK's Level Zero context
  (`init_level_zero()`) where VTK has one, else Tack's own: under a VTK that
  holds Level Zero memory, their outputs went back to VTK from Tack's own
  context and were refused. The examples test skips 42 and 43, which need
  that context, under a VTK without one, as it skips a missing optional module.
- A VTK with DLPack but without the device's memory (VTK master's nightly
  wheels, whose Viskores runs on the CPU alone yet lists CUDA and ROCm among
  `dlpack_support.supported_device_types()`) no longer fails every VTK
  algorithm on a GPU as it hands its output back: the interop shares one value
  once per device (`_vtk_holds`, by memory domain, VTK's error for a refusal not
  shown), and copies to the host where VTK cannot hold the memory.

## 0.6.1 — 2026-10-28

A fix for Tack's CPU on many threads. 0.6.0 capped every fan-out's chunks
at 20 µs of estimated work, meant for a few costly elements such as
streamlines; on long ranges it cut the work into thousands of interleaved
chunks, and on Linux the threads then faulted the same huge pages of new
outputs at once. On 28 threads (an EPYC 7453) every CPU filter was 2-5x
slower than the performance page reported, behind VTK on six filters of
each mesh. 0.6.1 caps only short ranges, where the cap was meant to apply.
The performance page's Linux columns are measured again with the fix, and
its CUDA column against VTK is filled for every filter.

### Fixed

- A CPU fan-out caps its chunks at about 20 µs of estimated work only on a
  short range, where the pool's own chunks are at most 256 elements. On any
  range the cap cut long dispatches into thousands of interleaved chunks, and
  on 28 threads the workers faulted the same huge pages of new outputs at
  once: threshold of 20M hexahedra took 650 ms against 200, point to cell
  120 against 30, and the ONERA M6's cell locator build 5.7 s against 2.9
  (4.4 s against 2.8 on 8 threads; jura, EPYC 7453). Short ranges of costly
  elements, such as a few streamlines, still go one element a claim.

## 0.6.0 — 2026-10-28

The headline: the dataset layer takes data in the shapes simulations write
it. Meshes in pieces, over devices and MPI ranks, through
`tack.communication`, a subset of DIY with Tack arrays as messages: ghost
cells and global ids, merging, probes and particles that cross pieces,
statistics over ranks, GPU-aware MPI on CUDA and HIP. Point clouds, with a
point locator, SPH, deposits, friends-of-friends halos and periodic boxes.
Block-structured AMR and octrees. Physical units on fields and parameters,
and time on datasets, with filters that stream steps. The filters are
written with public names only, so users can write their own (the
Visualization Guide's "Write a Filter" chapter). Around them: polyhedra and
polygons through probe, gradients and advection; voxelization, cell quality
and feature edges; a two-level cell locator that took the streamlines of the
ONERA M6 wing (46.9M cells) from minutes to tens of milliseconds on every
GPU, 20-33x faster than Viskores' on CUDA, each building its locator; a
prototype NVIDIA Warp backend; and a performance page against VTK and
Viskores.

Validated on Linux (CPU; CUDA on an RTX 4060 Ti and an RTX 6000 Ada; HIP on
two MI210s; Level Zero on an Intel Max 1100), macOS (CPU and Metal on an M1
Max) and Windows Server 2022 (CPU, and CUDA on a T4 in TCC mode), with debug
mode, both id widths of the dataset tests and MPI over 2 and 3 ranks,
together with tack-examples' ports, among them four production datasets (up
to 46.9M cells and 512 pieces). The dataset filters agree with VTK's at 10k
to 1M cells. Against 0.5, two known costs remain on Windows' CPU: filters
over about 10k cells are 5-22% slower on two threads, where the first
fan-out costs more than it saves, and converting a mesh from VTK is 1.2-1.6x
slower.

### Added

- A performance page in the documentation: Tack's filters against VTK and
  Viskores at 20M hexahedra and tetrahedra on the CPU and the GPU (an EPYC
  7453 with an RTX 6000 Ada, an M1 Max with its GPU), streamlines on CUDA
  against Viskores' on the ONERA M6 wing and an aneurysm, and four
  production datasets against VTK. Every Tack output is checked against
  VTK's before its time counts.
- `CellLocator(data, levels=1)`: the cells binned on one uniform grid of
  about `density` bins a cell, every bin a leaf, as Viskores'
  `CellLocatorUniformBins` and VTK's `vtkStaticCellLocator` bin them; the
  default, `levels=2`, is the two-level grid (Changed). VTK and Viskores let
  the caller choose the locator too. On a small, even mesh one level searches
  fewer candidates (tack-examples' fire port: 6.6 a location, against 12 for
  two levels); `probe`, `advect` and `streamlines` take it as `locator=`. On
  a mesh graded toward a body in a far field (the ONERA M6) a few bins hold
  most of the cells, and two levels are the choice. Both find the same cells
  and coordinates, bit for bit.
- `td.merge_points(data, tolerance=0)`: coincident points made one, as VTK's
  `vtkStaticCleanUnstructuredGrid` and Viskores' `CleanGrid` merge them.
  Points at most `tolerance` apart join, and so do chains of them; each set
  becomes its smallest point, with that point's position, point fields,
  ghost marks and global id, and the cells stay the same cells on the
  merged points. It is for meshes written without merging their points. A
  simulation's 512 pieces store each point about 2.6 times, so matching
  points across pieces numbered 16.9M of them; merged piece by piece first,
  `generate_global_ids` numbers 6,310,285, the merged mesh's count. Point
  counts match VTK's static clean, exactly and with a tolerance.
- A "Write a Filter" chapter in the Visualization Guide: five built-in
  filters (`point_elevation`, `threshold`, `shrink`, `jump`, `triangulate`)
  and `entropy` over pieces rewritten with public names only, each section
  on what that kind of filter must keep (relative positions, the device,
  implicit arrays, quantities as parameters, units of results, ghost marks
  through `carry`, `keep_marks` or `drop_marks`, hidden cells, owned values
  over pieces). Its program, `docs/examples/data_filters.py`, holds every
  rewrite to the built-in in every part, and runs in `validate.py`.
- `tolerance=` on `generate_global_ids`, `ghost_cells` and `merge_pieces`
  (VTK's `vtkGenerateGlobalIds` has one): points of different pieces at most
  that far apart are copies, and so are chains of them, for pieces whose
  copies of a shared point differ in their last bits. Points are binned into
  cells a little over two tolerances wide, each sent to its own cell and the
  ones its box reaches, through the same sort over the pieces as exact
  matching; owners then agree along chains by exchanges between partners.
  The default, 0, matches exactly as before. On a simulation's 512 pieces in
  f64, merged first with `merge_points`, a tolerance of 1e-12 numbers
  6,310,285 points (the merged mesh's), where exact matching numbers
  6,311,919, and gives VTK's ghost cell count.
- What the built-in filters are written with is public, so a filter of
  your own is written the same way: `td.carry` and its maps (`td.Same`,
  `td.Take`, `td.Pieces`, `td.Interpolate`), `td.keep_marks` and
  `td.drop_marks`, `td.visible`, `td.owned_values` (which takes a field's
  name too), `td.Set.from_flags` (a selection a kernel flags, for
  `extract_cells`) and `td.units` (quantities as parameters; the units of
  differences, products and SI names). On the Data Model page. An audit
  that rewrote built-ins with public names only found three silent errors a
  user's filter made without them: a dataset built by hand dropped the
  record of lost ghost marks (reductions over pieces then counted shared
  points twice), a difference of degrees Celsius kept the unit, and a
  hidden cell was processed. `test_public_filters.py` keeps six such
  rewrites equal to the built-ins, every part of the result. `td.carry` is
  now the function; the module is still `tack.dataset.carry` for
  `from ... import`.
- `td.streamlines(..., step_unit="length")`: steps of a fixed arc length
  along the velocity's direction, as VTK's `vtkStreamTracer` steps by
  default, with `steps * step_size` the longest path and `time` still the
  time the particle takes (1 / |v| integrated beside the path by the same
  weights). The default, `"time"`, is unchanged. A time step sized for the
  fastest speed made tiny steps where the flow is slow: the carotid port's
  median step was 0.0005 long against VTK's 0.1, and GPU streamlines from
  a few dozen seeds were 4-100x VTK's time. With length steps the four
  streamline ports take VTK's steps, match its paths to 2e-7 (carotid
  2e-4), and run in 0.16-0.57x its time on 8 CPU threads.
- A Visualization Guide chapter, **Communicate Between Blocks**, on writing
  an algorithm over pieces with `tack.communication`: collectives,
  exchanges, a ghost `Plan` that makes a diffusion over blocks give the
  whole line's bits, `Routing` points to the block that holds them, and
  `sort_by_key`. Its examples run in `validate.py` and, unchanged but for
  the world, under `mpiexec`. The Runtime API contract gains a
  Communication section: what is collective, what does not depend on how
  blocks are assigned, and the errors.
- Two reference pages in the Visualization Guide, generated from the
  docstrings: **Algorithms**, every `tack.dataset` algorithm by what it does,
  each group with a table of what it gives, what it takes (pieces, AMR,
  octrees) and its VTK and Viskores counterparts; and **Data Model**, the
  classes, constructors and sources, and the VTK, MFEM and Warp interop.
  A test fails when an exported name has no entry. The site's API pages now
  see tack-vis (a Griffe extension, `docs/_ext/tack_namespace.py`): before,
  nothing under `tack.dataset` or `tack.interop` could be rendered.
- A prototype NVIDIA Warp backend (`tack.device("warp")`, the `warp`
  extra): kernels lower to generated `@wp.kernel` source from Tack's typed
  IR and run on Warp's CPU or CUDA device, on its current stream, with
  Tack's integer and floating-point semantics, `wrap_ptr`, DLPack and debug
  bounds checks; workgroups on CUDA only, no `tack.inspect`
  (`Backend.supports_inspect`). Its arrays are `wp.array`s. CI's `warp`
  job runs it and the interop on Warp's CPU device.
- `tack.interop.warp`: `from_warp`/`to_warp` share arrays both ways (Warp's
  vector and matrix types as Tack's), and `from_simdata`/`to_simdata`
  exchange datasets with warp-simdata. Point clouds, curves, surface meshes,
  OpenFOAM meshes and patches, CGNS sections (one shape, MIXED, NGON_n,
  NFACE_n), EnSight parts, corner-point grids and FLASH axisymmetric data
  (an `AMRDataSet` of revolved boxes, coordinates implicit) come across
  without copies, CGNS's 1-based and MIXED connectivities read lazily;
  other models through warp-simdata's VTK conversion.
- An unstructured topology of one shape whose offsets count in steps of the
  shape's points shares its connectivity as the shape's rows rather than
  gathering them, so an imported mesh stays one copy through the filters.
  Polyhedral, polygonal and polyline topologies keep implicit arrays as
  given.
- Implicit booleans: `td.Union(f, g, ...)`, `td.Intersection(...)` and
  `td.Difference(...)` combine implicit functions as VTK's
  `vtkImplicitBoolean` does (the smallest value, the largest, the first's or
  the rest's negated), on the device, wherever a function is taken (clip,
  slice, extract_geometry, extract_points, implicit_values); held to VTK in
  every implicit-function comparison.
- `td.surface_normals(data, consistent=True)`: a surface's polygons wound
  alike -- each connected region as its first cell, by union-find with
  parities over the edges on the device -- and its unit normals on cells and,
  summed and made unit, on points, as VTK's `vtkPolyDataNormals` without
  splitting (matched to it on spheres, scrambled windings, curved quads and
  polygons). A surface no winding makes consistent is refused.
- `td.extract_extent(data, extent, rate=1)`: a box of a structured grid's
  points (VTK's VOI, in the whole grid's indices), every `rate`-th, as a
  structured grid of the same kind -- an image stays uniform, a rectilinear
  grid's axes are cut, a curvilinear grid's points taken; a k-plane of a 3D
  grid is a 2D grid. Matched to `vtkExtractVOI`, `vtkExtractRectilinearGrid`
  and `vtkExtractGrid` point for point and cell for cell.
- `vtk_to_dataset` reads `vtkPolyData` -- what VTK's surface readers (STL,
  PLY, OBJ, `.vtp`) and its contours give -- as the unstructured grid of its
  cells in VTK's order: triangles and quads an `UnstructuredTopology`, larger
  polygons a `PolygonalTopology`, lines a `PolylineTopology`, triangle strips
  as their triangles (wound alike, each taking its strip's cell data). It
  was refused, and needed `vtkAppendFilter` first.
- `contour(..., dimension=)`: 3, 2 or 1, the cells to contour -- by default
  the mesh's highest -- each dimension's alone, never mixed as VTK mixes
  them: the boundary triangles beside a mesh's tetrahedra give lines with
  `dimension=2`. Contours of lines and polylines (`dimension=1`, or a 1D
  mesh) are vertex cells at the crossings, a crossing at a point whose value
  is the level that point once; they were refused. Matched to VTK per
  dimension on Viskores' `mixed-cell-shapes.vtk`.
- Voxelization: `td.voxelize(data, grid, fields=None, samples=4)`, each voxel
  of an image the fraction of its volume the data covers and each field's
  mean over it, by sampling through `probe`, so for datasets, pieces, AMR
  hierarchies and octrees alike.
- Cell quality: `td.cell_quality(data, measures)`, the scaled Jacobian,
  Jacobian ratio, condition number, edge ratio and shortest and longest edge
  of every cell, each 1 on its shape's ideal cell. Verdict's definitions
  (matched to VTK's `vtkMeshQuality` where it defines them), extended to every
  shape: order-2 cells along their curved edges, wedges and pyramids, and
  polyhedra and polygons by their fan tetrahedra and triangles.
- Feature edges: `td.feature_edges(data, angle=30.0, boundary=True,
  feature=True, non_manifold=True, manifold=False)`, a surface's edges by the
  cells meeting there and their angle, as VTK's `vtkFeatureEdges` (matched
  edge for edge), as line cells on the same points with an `EdgeKind` cell
  field.
- `clip` of polyhedral topologies: each cut cell becomes the polyhedron of its
  kept part, its faces clipped once (so the output is conforming) and its
  iso-polygons (López's, as `contour` traces them) as caps; volumes equal the
  shape-based clip's for planes.
- Time on datasets, for filters that stream steps (in situ): `DataSet(...,
  time=, time_unit=)`, kept by every filter as the origin is; pieces and AMR
  boxes at two times are refused; VTK's `DATA_TIME_STEP` both ways.
  `tack.dataset.temporal` holds what the temporal filters share: a step cache
  that copies what it keeps (`copy=False` keeps references) and the time
  between steps, converted between units.
- Running statistics and time derivatives over streamed steps:
  `td.RunningStatistics(fields)` (per point or cell mean, variance, minimum and
  maximum over the steps so far, by Welford's method on the device) and
  `td.TimeDerivative(fields)` (the backward difference since the previous
  step, in the field's unit over the time unit), each fed `update(data)` a
  step at a time, piece by piece for a `PartitionedDataSet`; one mesh over the
  steps (a new topology object over the same mesh is the same mesh).
- Advection through streamed steps: `td.Advection(field, seeds)` carries
  particles from step to step, the velocity linear in time between the kept
  previous step and the current one (each probed in its own mesh, so meshes
  may change; over ranks for pieces), with `inject(points, time=)` releasing
  particles at any time; `td.extend_paths` joins each step's particles into
  paths; `td.Pathlines` and `td.Streaklines` return the current lines every
  step.
- Physical units, a first pass: `Field(..., unit=)` holds the values' unit, a
  string in unyt's canonical spelling (`None` is unknown), and the spatial
  unit is the geometry's (`data.unit`; `uniform_grid`, `rectilinear_grid`,
  `point_cloud` and `DataSet` take `unit=`). Filters that keep a quantity keep
  its unit; the rest give `None`. Two known spatial units do not meet (probe,
  pieces, AMR boxes). `convert(data, unit, field=None)` converts the geometry,
  the origin in double precision, or a field, offsets included. The VTK
  interop maps `UNITS_LABEL`. Tack knows the US survey foot (`ftUS`).
  tack-vis now depends on unyt (pure Python), imported on first use.
- Derived units: gradients (per spatial unit), jumps, `dot` and `cross` (by SI
  name where one fits: N m is J), `composite`, cell and face measures, edge
  lengths, fluxes and their divergence, `perot`, statistics
  (`Statistics.unit`, `weight_unit` and `unit_of(name)`), deposits (counts,
  `per_volume`), SPH density and sums, halo masses and radii, outlier
  distances, normals, particle times. A difference of degrees Celsius is in
  kelvin; such values are not multiplied. `log_values` refuses a dimensioned
  field. `interpolate(..., unit=)` states a function's unit.
- Quantities as parameters: lengths (`voxel_downsample(cloud, "10 cm")`,
  radii, periodic boxes, smoothing lengths, softening, `slice_plane`'s
  origin), thresholds and isovalues (`threshold(data, "T", "20 degC", ...)`)
  and step sizes take a string, a `(value, unit)` pair or a unyt or pint
  quantity, converted on the host; a plain number is as before. A mass,
  density or velocity given as one quantity brings its unit to the result.
- Octrees, a first pass: `td.OctreeDataSet`, a grid of octrees (VTK's
  HyperTreeGrid with branching factor 2) stored as a linear octree for GPUs,
  each node its tree, level and Morton key, sorted in pre-order; leaves,
  centres and measures from the keys in kernels. Cell data on every node, as
  RAMSES and HyperTreeGrids store it; non-leaves `REFINEDCELL`, so statistics
  and entropy count each region once. `OctreeDataSet.from_cells` takes cells'
  levels and positions (RAMSES, yt); `vtkHyperTreeGrid` converts both ways
  through each tree's breadth-first descriptor.
- An octree's leaves as cells and its probe: `octree.leaves()` is an
  unstructured dataset of voxels (pixels in 2D) with the fields as cell data,
  coinciding corners one point (sorted on the finest level's lattice), hanging
  ones kept; `probe(octree, where)` takes each point's leaf by one binary search
  of the keys, as VTK's `vtkHyperTreeGridProbeFilter`, and over MPI ranks sends
  each point to the rank whose trees hold it. The boolean `octree.leaves()` of
  the first pass is now `octree.is_leaf`.
- Weighted statistics: `statistics(data, name, weights="volume" or a field)`
  and `entropy(..., weights=)`, for datasets, pieces, AMR hierarchies and
  octrees: a volume-weighted mean over cells of different sizes, a mass from
  a density. `Statistics.weight` is the weights' sum.
- Block-structured AMR, a first pass: `td.AMRDataSet(levels, ratios)`, levels of
  structured boxes placed by their topologies' `offset`, the hierarchy in index
  space as AMReX's, so boxes may be images, rectilinear or curvilinear grids
  (VTK's AMR holds the first two). Made with its marks: `REFINEDCELL` on cells
  a finer box covers (as VTK's `BlankCells`, which marks Tack's export alike),
  `REFINEDPOINT` (a new bit, 8; VTK has none) on points a finer level also
  holds, and siblings' shared points `DUPLICATEPOINT`, so each region and
  location counts once in `statistics` and `entropy`. Filters that make cells
  leave hidden and refined cells out (`Ghosts.MASKED_CELLS`). Overlapping
  boxes, improper nesting and images off their level's lattice are refused.
  `vtkOverlappingAMR` both ways, `amr_gaussian_pulse` (VTK's source), and the
  boxes' metadata gathered over MPI ranks.
- AMR ghost cells: `ghost_cells(amr, layers)` grows each box, a ghost from the
  sibling holding it, else from the coarser level (injection for cells, linear
  interpolation for points and for a grid's positions, so rectilinear and
  curvilinear boxes grow too), hidden where neither holds it;
  `update_ghosts(amr, ...)` refreshes them, and `average_down(amr, name)` puts
  fine data into the coarse cells and points it covers. Each is a kept
  `tack.communication.Plan` and a weighted sum, so they run over devices and
  MPI ranks.
- `probe(amr, where)`: each point in the finest level's cell holding it, over
  every rank's boxes; onto an image, the hierarchy resampled, equal to VTK's
  `vtkAMRResampleFilter` (which leaves points on the domain's upper faces
  unset).
- A dataset's origin: `DataSet.origin`, three doubles on the host, where its
  geometry's coordinates start. Kernels see only the geometry's values
  (`tack.f32`, or `tack.f64` where the device has it), so points far from
  zero, such as a LiDAR survey at UTM coordinates, keep millimetres in single
  precision. Filters keep it. Implicit functions, probe points, seeds,
  elevation points and functions of position take world coordinates,
  converted in double precision; `transform` moves the origin;
  `world_positions()` and `rebased(origin)`; `vtk_to_dataset(..., origin=)`,
  and `dataset_to_vtk` writes world positions. Pieces whose positions meet
  must share one origin.
- `td.PointLocator`: points in uniform bins sorted by bin, for radius
  neighbours (CSR) and the k nearest, deterministic (no atomics), and a device
  class for one's own kernels. On it, `voxel_downsample` (centroid, first or
  nearest point per voxel), `outliers` (radius or statistical, marked; VTK's
  `vtkRadiusOutlierRemoval` and `vtkStatisticalOutlierRemoval` keep the same
  points) and `point_normals` (PCA over the k nearest or a radius, signed
  toward a viewpoint or up; within 0.1 degrees of `vtkPCANormalEstimation`).
- On the point locator: `connected_components(data, radius=, min_size=)`
  (Euclidean clusters, friends-of-friends; VTK's
  `vtkEuclideanClusterExtraction` one to one), `deposit` (points onto an image
  by nearest grid point or cloud in cell: counts, sums, means, minima, maxima,
  per volume; LiDAR surface models), and `sph_interpolate`/`sph_density`
  (VTK's four SPH kernels in 1 to 3 dimensions, Shepard normalization; within
  float rounding of `vtkSPHInterpolator`). All deterministic.
- Periodic boxes in the point locator (`PointLocator(..., periodic=box)`):
  bins tile the box and wrap, distances to the nearest image; the point
  filters take `periodic=`.
- `halo_catalog`: cosmology halos from friends-of-friends groups, ported from
  Viskores' cosmotools and ParaView's LANL halo finder: count, mass (per
  particle), centre of mass, most bound particle and its potential (with
  softening), mean velocity and dispersion, radius, in periodic boxes. The most
  bound particle is exact, with Viskores' pruned search on every large halo,
  tightened to bound each particle against each far bin. Viskores' two test
  cases give the same halos and centres.
- Point clouds in pieces. `ghost_points(pds, radius= | k=)` gives each piece
  copies of the other pieces' points within a radius of its box, or exactly
  those among its points' k nearest (images across a periodic box's faces),
  with fields, sets and global ids; `update_ghosts` refreshes them. Normals,
  the radius outlier test and the SPH filters then run per piece with the
  whole's answers. `connected_components(pds, radius=, min_size=,
  periodic=)`, `outliers`, `voxel_downsample`, `deposit` and `halo_catalog`
  take a partitioned point cloud, in one process or across MPI ranks: pieces
  cut in index order give the whole's clusters, centroids and catalogues bit
  for bit. Regions over pieces are now joined from the regions that cross
  pieces alone and numbered by a sort over the pieces, so no rank holds every
  region; `min_size` works on meshes in pieces too, counting each point once.
- `tack.communication.Plan` lets a block send to itself, as a piece in a
  periodic box is its own neighbour.
- `td.point_cloud(positions, fields)`: a dataset of one implicit vertex cell
  per point (stored nowhere), float64 positions on an origin of their own,
  integer fields (a LAS classification, intensities) kept in their type. The
  first step of point clouds and SPH (the vault's proposal).
- `merge_pieces(pds)`: the pieces of a partitioned dataset as one dataset,
  each owned cell and each point once, as `vtkMergeBlocks` after
  `vtkRemoveGhosts`. Points matched and numbered as `generate_global_ids`
  does; hidden cells and global ids kept. Structured pieces that tile a box
  come back as that box's grid, `split`'s input exactly; others as an
  unstructured dataset. Across MPI ranks it is gathered on a root rank.
- `connected_components(data)`: cells connected through shared points, as
  VTK's `vtkConnectivityFilter` (all regions, numbered alike) and Viskores'
  `CellSetConnectivity`: `region` on the cells and `point_region` on the
  points, by union-find on the device. Of a `PartitionedDataSet` too: each
  piece labels its own, the copies of shared points join regions across
  pieces, and regions are numbered across every piece and MPI rank, as on
  the whole dataset where the pieces carry its global ids.
- `ghost_cells(pds, layers=n)` makes any number of layers, each grown from
  the points of the layer before (cells of any piece, so pieces thinner than
  `n` cells are fine), equal to `split`'s and `vtkGhostCellsGenerator`'s for
  two and three layers. Structured pieces stay structured: pieces tiling a
  box of cells become their blocks grown by the layers, with their offsets,
  values from their owners and grown coordinates (uniform, rectilinear or
  explicit), `split`'s ghost layers exactly; `update_ghosts` works on them.
- `probe` of a `PartitionedDataSet`: each point goes to the lowest piece one
  of whose own cells holds it (holding ahead of the locator's tolerance),
  is evaluated there, and its values come back, equal to probing the whole
  dataset, with or without ghost cells, on any devices or MPI ranks.
  `where` may be a `PartitionedDataSet`, giving one back.
- `tack.communication.Routing`: points to the block that holds them (by the
  blocks' boxes and the caller's ranked test), values sent with them
  (`send`) and answers returned in the points' order (`reply`). Particles
  across pieces now route through it.
- Particles across pieces: `advect` and `streamlines` of a
  `PartitionedDataSet` route each particle to the piece owning the cell
  that holds it, advance it there, and hand it on when a whole step ends in
  a ghost cell, in rounds until none is in flight, on any devices or MPI
  ranks (each rank its own seeds). With a ghost layer wider than a step the
  paths are the whole mesh's up to rounding (checked: positions within
  1.2e-6, the same steps and status); without ghost cells particles cross
  by the short step to the boundary. `streamlines` gives each piece the
  stretches it traced, with `seed` and `segment`.
- `advect` takes its own output as seeds and goes on from there: times,
  steps and status carried, stopped particles left as they are.
- `statistics` and `entropy` of a `PartitionedDataSet` work across MPI
  ranks: each rank passes its pieces (`PartitionedDataSet(..., world=)`),
  and the pieces' moments and histograms are gathered and combined in the
  pieces' global order, so the result has the same bits however the pieces
  are shared out, a rank with none included, and as in one process.
- Global ids: `data.global_ids` (`tack.dataset.GlobalIds`), an `i64` per
  point and per cell, the same on every piece holding it. VTK's `GlobalIds`
  attributes come in as them (no longer as fields) and go back out;
  filters carry them where an output entity is an input entity.
  `generate_global_ids(pds)` makes them as `vtkGenerateGlobalIds` does
  (equal ids and duplicate marks on pieces without ghost cells; on pieces
  with ghost cells it keeps the owners the marks say, where VTK reassigns
  them), in one process or across MPI ranks, and `ghost_cells` carries them
  into the ghost layer, matching points by them.
- `ghost_cells(pds)` gives each unstructured piece of a partitioned dataset
  a layer of its neighbours' cells, as VTK's `vtkGhostCellsGenerator` does:
  the same cells, points, marks (a shared point belongs to the lowest piece
  holding it) and values, checked against it and against `split`'s ghost
  layers. Pieces may be on any devices, or on MPI ranks (each rank its own
  pieces; `PartitionedDataSet(..., world=)`), the same code through
  `tack.communication`: boundary points matched by a distributed sort of
  their positions, cells sent to neighbours by `exchange`.
  `update_ghosts(pds, *fields)` refreshes ghost values from their owners
  through the two `Plan`s `ghost_cells` keeps, so a second filter that needs
  neighbours (point gradients of projected values) is right on every owned
  point.
- `tack.communication.sort_by_key(blocks, keys, values)`: every block's keys
  sorted together, the global order cut into one range per block, every
  copy of a key on one block (a sample sort, as DIY's), in one process and
  across MPI ranks.
- `tack.communication`: communication between blocks (units of work with
  global ids, each on a device, such as a partitioned dataset's pieces), the
  patterns that algorithms over pieces are written on. `Blocks.exchange`
  sends messages, each a dict of named Tack arrays, from any block to any
  block, arriving on the destination block's device; `all_reduce`,
  `all_gather` and `exclusive_scan` combine small host values over every
  block in global-id order, so a sum has the same bits however blocks are
  assigned; `link_by_bounds` finds neighbours by overlapping boxes; a `Plan`
  keeps an exchange (elements gathered on one block, scattered on another)
  for reuse, as ghost values are refreshed from their owners. DIY's model,
  with Tack arrays as the messages. `World()` is one process, blocks on any
  of its devices; `World(comm)` spans an mpi4py communicator's ranks (the
  new `mpi` extra), staging arrays through the host or, with
  `gpu_aware=True`, handing them to MPI through DLPack. The same cases run
  on both, and a CI job runs them under `mpiexec`.
- `contour` takes a sequence of values, and `slice` and `slice_plane` take
  one as `value` (parallel planes for `slice_plane`): every value's contour
  in one dataset, made in one pass over the cells, each output cell's value
  in a cell field `contour_value`. On every path: shapes, polyhedra, 2D
  lines and per-cell (`L2`) lines. 10 values on 1M hexahedra: 38 ms against
  80 for ten contours (CPU). As warp-simdata's `compute_many`, but one
  dataset, as VTK's and Viskores' contour filters give.
- `topology.validate()` checks the arrays a topology is built from, on their
  device, reading back a few integers: offsets that start at 0, never run
  backwards and end at their members' length, ids within range (points,
  faces, sides), and an unstructured topology's types and point counts. It
  raises `ValueError` saying what is wrong and how often. Debug mode
  (`tack.set_debug`, `TACK_DEBUG`) checks every topology's ids when it is
  made, before anything is derived from them: a bad id from a reader or a
  caller was read or written out of bounds, silently. As warp-simdata's
  `validate_datasets`.
- `for_each(kernel, data, space, ...)` runs a kernel over a space's nodes, each
  once (`DataSet.node_views`): views over points, edges, quad faces or cells
  whose entities have `node_count(i)` nodes with `node_dof(i, j)` and
  `node_position(i, j)`. `interpolate` is now one kernel over them, with no
  table of writers and no sort, and gives the same values; an `L2` geometry,
  which gives a shared node no position of its own, keeps the first cell's.
- A `@tack.device_class` object, or one it holds, can be passed to a
  `@tack.func` or to another template's method -- `helper(obj, i)`,
  `b.combine(a, i)`, `helper(self)`, `helper(self.base)` -- not only to a
  kernel: the callee is specialized for it, as a method is for `self`.
- A device function may `return f(x)` where `f` returns several values,
  vectors among them, and a name may hold such a result until it is
  unpacked (`hit = closest_hit(o, d)`, then `depth, normal, color = hit`).
- Partitioned datasets, the first phase of the partitioned-data proposal.
  `tack.dataset.PartitionedDataSet(partitions)` is a dataset in pieces, an
  ordered list of `DataSet`s as VTK's `vtkPartitionedDataSet`, each piece
  with its own topology, fields, sets and device. Work on one piece takes
  the piece: the caller iterates. `statistics` and `entropy` also take a
  partitioned dataset and a field's name, and combine the pieces (moments
  pairwise; histograms over the whole's range), each piece reduced on its
  own device. `split(data, pieces)` cuts a structured grid by index ranges
  into pieces that keep the geometry's kind and carry their block of the
  point, cell and `L2` fields and the sets. `vtk_to_dataset` and
  `dataset_to_vtk` convert `vtkPartitionedDataSet` and
  `vtkPartitionedDataSetCollection` (a `dict` of names to partitioned
  datasets).
- Filters carry ghost marks, the second phase of the partitioned-data
  proposal. They process a piece's ghost cells, as VTK's do, and what they
  make takes the owner of what it came from: an output cell its source
  cell's, a point on an input edge the edge's, a point inside a cell the
  cell's, a surface face its cell's, and a face two cells share or an edge
  the owner of its smallest point, by position. Over pieces with a ghost
  layer, the owned part of every filter's output is the whole's output, and
  `statistics` and `entropy` of the outputs are the whole's (checked for
  every filter that makes cells or points, chained, on polyhedra, in 2D,
  over `split`'s pieces and VTK's). `split(data, pieces, ghost_layers=n)`
  adds ghost layers; `remove_ghosts(data)` takes the duplicate and hidden
  cells out, keeping the points a piece owns, before rendering or writing
  one file. Filters never process hidden cells. Outputs of pieces without
  ghost cells, which cannot know what their neighbours make on the faces
  and edges they share, record that they lost the marks, so reductions
  refuse them, saying why.
- Ghost marks: every cell and point that several pieces hold has one owner,
  and the other copies are ghosts, marked in `data.ghosts` (a
  `tack.dataset.Ghosts` of `u8` marks per cell and per point, in
  VTK's `vtkGhostType` bits). `vtk_to_dataset` takes a `vtkGhostType` array
  as the marks rather than as a field, and `dataset_to_vtk` writes it back;
  `split` marks the points its pieces share as duplicates in all but one;
  `statistics` and `entropy` leave out duplicate and hidden cells and
  points, so over pieces with one owner per point they give the whole's
  answer. Filters that keep the cells and points keep the marks; the others
  do not carry them yet, and their outputs record that they lost them, so
  `statistics` and `entropy` refuse those outputs rather than count shared
  cells and points once per piece.
- Examples: `45_flow.py` (gradients, `flow_quantities`, `advect`,
  `streamlines` and `perot` on a swirl whose answers are known exactly) and
  `46_cut_and_sample.py` (`clip`, `extract_geometry`, `extract_points`,
  `slice` and `probe` with implicit functions and a reused `CellLocator`),
  each checking its results. The dataset tour now runs and checks the
  algorithms no example showed: `implicit_values`, `point_elevation`,
  `log_values`, `point_ids`, `cylindrical` and `spherical` (and back),
  `mask_points`, `extract_edges`, `tetrahedralize`, `as_polygons`, `perot`
  and `shrink`.

### Fixed

- HIP arrays take `hipMalloc` memory through a cache of dropped arrays'
  buffers, reused by size class in stream order, instead of the device's
  stream-ordered pool. On ROCm 7.1.1 the pool slowed with every allocation
  it had made (63 to 600 µs a `hipMallocAsync` over the 512 eta pieces'
  global ids, 2.7 ms when trimmed), and that port stalled for two hours;
  `hipMalloc` stays at 17 µs, and the cache spares the `hipFree` that waits
  for the device. The cache holds at most a quarter of the device, never
  memory handed outside Tack nor MPI arrays, and an allocation that runs out
  frees what it holds first (`Backend.release_memory`, which Metal's pool
  answers too). `TACK_HIP_ALLOCATOR=pool` still selects the pool.
- A dropped dataset frees its device memory at once. A topology interned
  its spaces in a dict and each space holds its topology, so every dataset
  with a field was a reference cycle, freed only when Python's cycle
  collector ran, which counts objects, not device memory: on a 16 GB GPU
  the 20M-cell checks ran out of memory with dropped datasets still on the
  card. The topology holds its spaces weakly. A topology's cached cell and
  point locators still hold a dataset of it, a cycle left for now.
- An allocation that runs out of device memory collects Python's garbage
  and tries once more. A dropped array held in a reference cycle keeps its
  memory until the cycle collector runs, which counts objects, not device
  memory: a topology's cached locator holds a dataset of that topology, so
  a mesh dropped after a probe or streamlines stayed on the GPU, and a
  16 GB card ran out with dropped meshes on it. Every backend raises
  `DeviceMemoryError` (a `MemoryError`) when an allocation runs out (CUDA,
  HIP, Level Zero, Metal; NumPy's `MemoryError` on the CPU), and
  `tack.empty` collects, waits for the device and retries once.
- The CPU fans out fewer elements than it has threads when each is dear
  enough to repay it. The threshold was floored at the thread count, so on
  jura's 28 threads the ONERA M6's 17 streamlines ran serially on every
  call (30 ms, against 7-12 on 8 threads), and a short range of dear
  elements never fanned out from 10 threads up. The floor is two elements,
  the fewest a fan-out can split; above it the costs decide.
- Advection on a GPU runs in launches of a bounded number of steps, each
  going on where the last paused, sized to about 0.2 s at the last launch's
  pace (the first takes 16 steps). One launch carried every particle's whole
  path, and a GPU driver resets a device whose launch runs past its limit:
  the ONERA M6's streamlines lost headroom's Level Zero device every time
  (7.5 s there; 2 s by default on Windows), and a lost device takes every
  array of the process with it. The office port's one launch took 4.9 s.
  A pause keeps each particle's position, time, steps and status, whether it
  stepped in this call (a hand-off between pieces needs it) and the cell it
  was last found in, so the paths are one launch's, bit for bit, on the CPU
  and on Metal. The CPU still runs one launch.
- A particle pushed across a mesh boundary that landed inside and went on
  took its next whole steps from a stale state: its RK4 stages placed at the
  bisection's last trial size rather than the step's, and, when a trial's
  stage had left the mesh, starting mid-step on the trial's stage
  velocities. Some such particles then stopped as if seeded outside
  (`SUCCESS` cleared). Particles that run along a curved wall, an
  aneurysm's, changed path: 82 of 20,000 seeds there.
- `carry`, `visible`, `owned_values` and `orient` run on their dataset's
  device. Made public for filters of one's own (#163), the first three had
  only been called from inside algorithms that had already made that device
  the default; called directly they allocated on the default device, and
  failed with "arrays on two devices" (six of `test_public_filters`'
  rewrites in a CUDA run, after an earlier test left CUDA the default).
  `orient` failed the same way on a topology on another device, and
  `without_hidden` and `refuse_hidden`, which the built-ins call, allocate
  on their dataset's device too. The "Write
  a Filter" chapter's `triangulate` makes its `ConstantArray` on the data's
  device.
- A compiled CPU kernel bound to another device's arrays raises a
  `TypeError` naming the kernel and the argument, where it wrote device
  pointers from the host and segfaulted (CUDA arrays, met through the Warp
  backend's run of `test_parallel_dims.py`). Arrays bind through
  `DeviceBuffer.host_address`, which only host memory has: the CPU's, and
  Warp's on its CPU device. A kernel call never got that far, since
  dispatch picks a device that can address every array.
- A repeated `extract_surface` of a mesh whose faces were never derived
  costs only the carry again. The fresh boundary (#157), which finds the
  surface without deriving every face, kept nothing, so every call paid
  for it: on jura, 20M hexahedra's warm surface went from 5.7 ms to 447 on
  the CPU (1.5 to 28 on CUDA). The topology now keeps the surface it found;
  5M hexahedra's repeated call on an M1 Max takes 0.3 ms, against 55-150.
- `np.from_dlpack` of a GPU array (CUDA, HIP, Level Zero) raised
  `SystemError` in place of NumPy's own error: the capsule's destructor, a
  ctypes callback, met and cleared the pending exception. C now gets native
  destructors.
- `point_normals` gave a line of points (no plane) an arbitrary normal, and
  on CUDA `[0, 0, 1]`: the least eigenvalue counts as repeated within the
  closed form's rounding, not only when exactly equal.
- A mesh with cells of more than one dimension (a surface's triangles among
  tetrahedra, as Gmsh and Exodus meshes carry): `cell_geometry` measures each
  cell in its own dimension, and `net_flux`, `perot` and `divergence` give its
  lower-dimensional cells zero; all three failed inside kernel resolution
  ("no @tack.func method 'side_position'"), and `perot` left those cells
  unwritten. Found with Viskores' test datasets, now in the tests
  (`_viskores_datasets.py`, measures held to Viskores' expected values).
- `as_polyhedra` refuses an inside-out cell beside an upright one, which no
  polyhedral face can record, saying so; `contour` and `clip` of the result
  failed later, with faces that "do not close" and a pointer to
  `check_winding`, which passes such cells.
- `tetrahedralize` and `triangulate` of a collapsed cell (repeated points, a
  pyramid written as a hexahedron) leave out its pieces with a repeated
  point, which measure nothing; VTK keeps them, but their repeated faces made
  every filter that derives faces refuse the output.
- `vtk_to_dataset` of a grid with a point array and a cell array of one name
  (VTK keeps them apart) kept only the cell array, so a contour of the name
  found nothing: the cell array is now the field `"<name> (cells)"`, with a
  warning. Viskores' `mixed-cell-shapes.vtk` has such a pair.
- `contour` and `clip` of polyhedra with a NaN in the field: the tracing took
  a NaN as neither above nor below the level, and its faces did not close
  ("check_winding"); a NaN is now below every level, as the shape path's
  cases have it, and an inside clip keeps it, so the clipped volumes are the
  shape path's on both sides.
- `contour` of a 2D mesh with a NaN in the field: the first crossing was
  found with `< iso` but crossings paired with `>= iso`, so fewer lines were
  written than counted and the rest held whatever `empty` gave them -- read
  as indices, a bus error now and then. A NaN is below the level there too.
- Point filters on GPUs (the validation round on mustafar-linux, headroom
  and illyad, 2026-10-25):
  - `PointLocator.neighbors` and `nearest` ran on the default device when no
    Tack array was among their arguments (the locator's own points, or host
    query points), and failed with arrays on two devices. The locator now
    has its dataset's device, as `CellLocator` does.
  - `voxel_downsample(keep="nearest")` broke a tie the other way on CUDA, HIP
    and Level Zero: a voxel of two points has them equidistant, and a GPU
    fused the squared distance into multiply-adds. Distances that decide a
    neighbour, the k nearest, a friends-of-friends link or a voxel's point
    are now named products (`squared_length`), rounded alike on every
    backend, as implicit functions' values already were.
  - Level Zero compiles with `-cl-fp32-correctly-rounded-divide-sqrt`: its
    f32 division (2.5 ulp) and square root (3 ulp) put centroids and means a
    bit off CUDA's, HIP's and the CPU's.
  - The docs said the point filters' sums had the same bits on every backend.
    They have them from run to run; counts, labels, neighbours and kept
    points are the same on every backend, and sums agree to rounding.
- GPU-aware MPI with HIP arrays: Tack's HIP arrays come from the
  stream-ordered pool (`hipMallocAsync`), which MVAPICH-Plus 4.1's UCX took
  for host memory (an assertion, or a segfault, at the first send; illyad).
  `World(comm, gpu_aware=True)` now hands such an MPI device copies in
  `hipMalloc` memory and receives into it (`Backend.mpi_direct`,
  `Backend.allocate_mpi_array`); the exchange runs at 36.8 GB/s a rank on
  the MI210s, against 44.6 for `hipMalloc` arrays sent as they are.
- GPU-aware MPI with CUDA arrays, the same way: Cray MPICH 9.1 could not
  export Tack's CUDA pool memory (`cuMemAllocAsync`) to another rank on the
  node ("Invalid count") and libfabric refused it across nodes (Perlmutter's
  A100s). Sends go as copies in `cuMemAlloc` memory and receives land in it:
  22.2 GB/s a rank over CUDA IPC on one node, 5.9 across two. `World`'s
  docstring says what such an MPI needs of the launch: each rank's device
  made before MPI starts, and every GPU of a node visible to its ranks.
- `slice` and `slice_plane` of a curved dataset (an order-2 geometry) raised
  `NotImplementedError` since #93 made `contour` refuse order-2 fields: the
  implicit function's values are in the geometry's space. They are now
  projected into order-1 H1 first, so the cells are cut linearized at their
  corners, as before #93. Example 43, which contoured an order-2 field, now
  contours its order-1 interpolant. Found on mustafar's CUDA round.
- Face derivation on AMD GPUs: `_sort_buckets` sorts a bucket of up to 16
  entries in a local array and writes it back once, instead of by insertion
  in global memory, where each shift is a store the next comparison reads.
  1M hexahedra on an MI210: the sort 2.48 to 0.76 ms, `faces()` 9.2 to 7.5,
  a fresh `extract_surface` 11.6 to 9.8; unchanged on an H100 and the CPU.
  Found on illyad's round.
- `tetrahedralize` over the pieces of a partitioned dataset splits each face
  pieces share along the same diagonal on both sides. A structured piece
  alternated its hexahedra by the parity of its own (i, j, k), so a piece
  starting at an odd index split the shared quads the other way; it now adds
  the topology's `offset`, the grid index of its first point, which `split`
  sets and `vtk_to_dataset` takes from an extent (`StructuredTopology(...,
  offset=)`). Other pieces split from each cell's corner of smallest local
  id, which differs between pieces; they now rank corners by global id where
  the dataset has global ids. A single dataset's tetrahedra are unchanged.
- The CPU threading policy kept fanning out kernels whose threads contend
  (many atomics on a few counters: a cell-type histogram over one type, a
  hot bucket), at 8-9x their serial time on 28 cores. A fan-out measured
  slower per element than serial was folded into the serial estimate, and
  the threshold still credited threads with a 10% gain. Now, in a kernel
  with atomics, such a fan-out is settled by one whole-range serial run: if serial wins, the kernel is
  contended and runs serially, with the back-off rechecks still trying
  threads now and then. jura: histogram 41.2 to 4.8 ms (one thread 4.6);
  yavin, one atomic counter over 1M elements: 39.2 to 2.3 ms. Found and
  fixed on jura.
- Implicit functions (`Plane`, `Sphere`, `Cylinder`, `Box`, `Planes`) give
  the same bits on every backend: each product in `value` is named, so the
  float contract rounds it on its own and no backend fuses it into a sum.
  A slice through grid points made 1015 triangles on Metal and 883 on the
  CPU (Metal's fused multiply-adds left noise about zero where the CPU had
  exact zeros, which `contour` merges); both now make 883.
- `as_polyhedra` and `extract_surface` scanned their faces' point counts (4
  a face) into the mesh's own id type, which a mesh is given for 12 per
  cell: a structured grid of 170M cells, or 120M hexahedra that share no
  points, still on 32-bit ids, wrapped the offsets. They now widen to 64 bits
  when the faces need it. Found by an audit of every scan in Tack; the other
  46 were covered by their bounds.
- A point just outside a pyramid, beside its apex, was located in no cell
  though within the locator's tolerance: Newton runs off there. Within 0.2%
  of the pyramid's size of the apex, it is now placed just past the apex,
  outside the hold margin, so it is near the pyramid (a cell holding it
  still ranks first) and takes the apex's value.
- A point at a pyramid's apex was located in no cell: every `(r, s)` at
  `t = 1` maps to the apex, so Newton's Jacobian is singular there and it
  did not converge. `Pyramid.world_to_parametric` now takes a point within
  `shapes.APEX_SNAP` (1e-6 of the pyramid's size) of the apex to be at
  `(0.5, 0.5, 1)`. In `vtkCellTypeSource`'s pyramid grid, all 18 apexes were
  found by no cell; both searches find them now. A point just outside a
  pyramid beside its apex, on the mesh's boundary, is still not found.
- `CellLocator.find`, `probe` and advection: which cell held a point on a
  face, edge or vertex that cells share was the sign of rounding noise, so
  the searches from the points and from the cells could put a mesh point in
  different cells, on the CPU (hexahedra, tetrahedra, wedges, pyramids) as on
  CUDA. A cell now holds a point within 1e-5 of its parametric domain
  (`shapes.HOLD_TOLERANCE`), so every cell that has a shared point holds it
  and the smallest id wins, in both searches and on every backend.
- CUDA: a product bound to a name, or returned from a device function, was
  fused into a later statement's sum (NVRTC's `--fmad=true` fuses across
  statements: all of 100k values on an RTX 4060 Ti), against the float
  contract's one rounding per named value. The CUDA generator now emits
  such products as `__fmul_rn`/`__dmul_rn`; a product inside a sum may
  still fuse, as the contract permits.
- DLPack: Tack set argument types on `ctypes.pythonapi`'s capsule
  functions, which are one object for the whole process, so its types and
  another library's replaced each other. Imported after Warp, Tack made
  Warp's capsule destructor raise `ArgumentError` at exit; imported before
  it, Warp's made Tack's destructor miss its capsule, and every array
  exported through DLPack and never consumed stayed pinned, its device
  memory never freed. Tack now binds the functions with prototypes of its
  own and touches nothing shared.
- `contour` read only an order-2 `H1` field's corner values, so a quadratic
  that rises inside a cell from zero at its corners gave no surface,
  silently. It now refuses an order-2 field (`NotImplementedError`) on
  every path, until cells are subdivided for contouring.
- CUDA: Tack works in the device's primary context (retained, and released
  when its last backend goes) instead of creating its own. PyTorch, CuPy and
  Warp use the primary context, so a launch of Tack's after one of them made
  it current no longer fails with `CUDA_ERROR_INVALID_HANDLE`. A context an
  embedding application made current is still adopted as before.
- `CellLocator.find`, and so `probe`, located a point in the lowest-numbered
  cell within the tolerance (0.1% of a cell's box diagonal), whether or not
  it held the point, and extrapolated there. On a Delaunay mesh of 16k
  tetrahedra, 2006 of 75k probes were wrong, by up to 1.7. Now a cell that
  holds the point ranks ahead of one it only comes near, and advection ranks
  its cells the same way. On GPUs without 64-bit atomics (Metal, Level Zero)
  the locator's limit halves, to 2^30 − 1 cells.
- `probe` grouped its points with two radix sorts whenever some lay outside
  the mesh (they share one bucket): 57 launches. `buckets.bucket_groups`
  counts, scans and scatters instead, leaving the unread bucket unfilled:
  19 launches, and on an RTX 4060 Ti 10.5 → 4.0 ms for 100k points.

### Changed

- **Breaking:** the VTK interop needs VTK 9.7.1 or later (the `interop`
  extra requires `vtk>=9.7.1`), and uses VTK's pythonic API throughout:
  keyword constructors, properties, dict-like point and cell data, implicit
  arrays. A one-component `vtkConstantArray` or `vtkAffineArray` in point or
  cell data becomes a `ConstantArray` or `CountingArray` field, and back.
- `key_order`, which merges contour and slice crossings, keeps the bucket
  sort on CUDA, HIP and Level Zero until buckets outnumber keys 64 to 1
  (8 to 1 on the CPU and Metal, as before). On CUDA the buckets were
  1.3-1.7x faster for a contour or slice of hexahedra at 13-20 buckets a
  key, where the radix sort had made them slower since it was introduced
  for memory (5M hexahedra's contour 5.4 to 7.8 ms); on the CPU radix is
  faster there (0.7-0.8x), and Metal goes either way.
- The Visualization Guide and `advect`'s docstring say when to advect on
  the CPU: each particle is one thread, so below a few thousand seeds a CPU
  copy of the mesh is usually faster than a GPU (the aneurysm's 25 seeds
  8.4 ms on 28 cores against 32 on an RTX 6000 Ada; 20,000 seeds 871
  against 331).
- `CellLocator` takes its cells' bounds on the device, a GPU's threads
  each over every k-th box and the CPU's over a block, and reads back only
  the partial results. It read every box back and reduced them with NumPy:
  on the ONERA M6 (46.9M cells) that was 1.0 s of the locator's 1.4 s
  build on an M1 Max's GPU, and over PCIe on a discrete GPU (jura's CUDA
  build took 4.9 s, against Viskores' 3.0). The M6's build is now 0.45 s on
  Metal and 1.7 s on the CPU (2.7 s); the bounds, so the grid and every
  result, are the same bit for bit.
- The cell locator bins cells on two levels, as Viskores'
  `CellLocatorTwoLevel` does: a coarse uniform grid of about 32 cells a bin,
  each bin a uniform grid of leaves sized by its count, about 8 cells a leaf.
  One level of about a cell a bin held up to 2.1M cells in a bin near the
  ONERA M6 wing (46.9M cells graded toward the wing in a far field a hundred
  chords wide), so a location there scanned 164k candidates on average, and
  a GPU's threads waited for their group's worst: Tack's streamlines on the
  M6 took 27.9 s for one seed on CUDA against Viskores' 3.5 s, and 230 s for
  17 on an M1 Max's GPU. Now the largest leaf holds 13,918 cells, and the 17
  streamlines take 0.36 s on Metal and about 4 ms warm on the CPU (1.5-2.3 s
  before). The build is faster (M6 5.6 s to 2.8 s on the CPU) with fewer
  entries (8.8 to 5.7 a cell); the aneurysm's 8.7M tetrahedra build in 0.68
  s (0.83), 12.9 entries a cell (7.8), and probe and streamlines there are
  as fast as before. Traces and probes are the same, bit for bit: the leaves
  only propose cells, and the search ranks them as before. Viskores sizes
  leaves at 2 cells; a bin's count is of the boxes overlapping it, a
  tetrahedron's box about six times its volume, and at 2 the aneurysm's
  tetrahedra were each entered 34 times (296M entries). A bin's leaves a
  side are rounded, where Viskores truncates: 32 cells a bin over 8 a leaf is
  1.59 a side, which truncated left every bin of a small, even mesh one leaf.
  tack-examples' fire port (10,462 cells, about 41 a bin) scanned 43
  candidates a location, against one level's 6.6, and its streamlines took
  twice as long on CUDA, HIP and Level Zero (jura's CUDA: 55 ms to 105).
  Rounded, 12 candidates (jura measured 83 ms with leaves about that size); the
  aneurysm 99 (110) at 15.6 entries a cell, the M6 unchanged, and the
  streamlines on the CPU as fast. `CellLocator(data, levels=1)` (Added) is
  the one level again, for meshes like the fire's.
- The CPU threading policy fans out a short range of costly elements on its
  first dispatch, and balances it. A range too short for a timing slice ran
  whole on one thread, and the first recheck ran it whole again; fanned out,
  17 elements went to 8 threads in chunks of 3. Now one element is timed
  first: a cheap one changes nothing, a costly one (100 µs or more) starts
  slices until one lasts 20 ms, and the rest fans out; a recheck believes a
  sample that long; and a fan-out's chunks hold at most about 20 µs of
  estimated work, so costly elements go one a claim. 17 streamlines on the
  ONERA M6 wing: the first two calls 8.8 and 8.5 s to 2.4 and 2.1, steady
  2.7 to 1.6 (M1 Max, 8 threads); an aneurysm's 25 steady 1.5 to 1.1-1.4 ms.
- Streamlines and `advect` try the cell a particle was last found in before
  searching the locator's bins. The cell is taken only where it holds the
  point well inside (by 1e-3 in its parametric coordinates, a hundred times
  the tolerance for holding a point), where no other cell of a mesh whose
  cells do not overlap can hold it, so the traces are the search's bit for
  bit; near a face, and in pyramids, surfaces and polyhedra, the bins are
  searched as before. RK4 streamlines on an aneurysm's 8.7M tetrahedra went
  from 7.4 ms to 1.7 (VTK on 8 threads: 3.6), and on the ONERA M6 wing's
  46.9M cells, with its boundary layer of wedges, from 4.9 s to 1.0 (VTK:
  3.9).
- Matching points across many pieces in one process no longer sends a
  message from every piece to every other. `tack.communication.sort_by_key`
  joins the blocks' sorted keys and cuts them where the splitters do when
  every block is on this rank and one device, and the matching behind
  `generate_global_ids`, `ghost_cells` and `merge_pieces` hands each piece
  its copies on the host there. The copies of a value are grouped from the
  sorted keys, sorting by value only where two values hash alike, instead
  of `np.unique` over rows. On a simulation's 512 pieces (5.5M voxels),
  global ids went from 83 s to 9.0 (VTK on 8 threads: 10.4) and a layer of
  ghost cells from 184 s to 28.7 (VTK: 27.3). Across MPI ranks the messages
  are as before.
- `merge_pieces` of unstructured pieces gathers cells' point ids in one go
  instead of slicing them a cell at a time in Python: 8.0 s to 0.9 s to merge
  an aneurysm's ten pieces (8.7M tetrahedra).
- A few crowded points no longer send a mesh's face derivation to the slow
  sorted path. Faces go into buckets by lowest point, and one thread sorts
  each bucket. Any bucket over 256 faces used to fall back to the sort, and
  an aneurysm's 8.7M real tetrahedra had one point of 266 among 1.5M. The
  test is now the sorting work: the buckets' squared sizes at most 512 per
  face, and no bucket over 8192. A fresh mesh's `extract_surface` there went
  from 930 ms to 350 (VTK on 8 threads: 532).
- Cold compiles cost less. Walking the IR uses an explicit stack, not
  nested generators, which handed each node up once per level of depth.
  Lowering copies syntax trees directly rather than through `copy.deepcopy`.
  Weak literal locals are found in one rewrite of the body, and copy
  propagation leaves alone the statements it has nothing to replace in.
  A cold probe of a mesh of five shapes went from 2.5 s to 2.1, with the
  generated code identical (LLVM IR hashed for 470 variants). Point links
  sort each bucket by entry alone, without reading keys every entry of a
  bucket shares.
- Filters hold less memory at scale, measured beside VTK's by the new
  `benchmarks/vtk_compare/memory.py`, which samples the allocator's bytes in
  use through one call in a process of its own. At 20M hexahedra on the CPU:
  - `contour`'s peak is 183 MB, down from 671 (VTK 74); `slice_plane`'s is
    294, down from 725 (VTK 319). Merging a few million crossings no longer
    allocates bucket counts and offsets over every point: `key_order` radix
    sorts keys that are far fewer than their buckets. The per-cell counts are
    scanned in place.
  - `extract_surface` of a fresh mesh peaks at 2.1 GB, down from 4.0 (VTK
    1.5); `faces()` holds less while it works too. Past 20M hexahedra the
    fresh boundary takes the points in ranges, holding at most 2 GiB of face
    records at once, for the same surface. At 100M hexahedra it takes 1.46 s
    and peaks at 4.7 GB, against VTK's 1.47 s and 7.6 GB; in one pass it
    took 3.9 s. A face record no longer
    stores its lowest point, which its bucket already gives. Each bucket
    counts its faces and numbers them in place of per-record flags, run ids
    and slots, and the surface's rows come from each boundary face's cell
    rather than a slot for every (cell, face).
  - Cell to point of a cell field on cells of one shape reads the cells'
    values through the point links, rather than a contribution for every
    corner (640 MB at 20M). At 20M hexahedra its peak went from 1.7 GB to
    1.07 (VTK 0.99), and its warm time from 25 ms to 12 (VTK 96); at 100M
    tetrahedra it peaks at 1.9 GB against VTK's 3.4.

  Every filter runs at 50M and 100M hexahedra and tetrahedra.
- A fresh mesh's first filters cost less at scale. Faces are derived by a
  counting sort of the (cell, face) records into buckets of their lowest
  point, each record placed once and each bucket sorted in place, with
  rows and sides from the cells in their order (the same faces, id for id);
  `extract_surface` of a mesh whose faces are not derived keeps the faces
  of one cell from those records without deriving the whole incidence; and
  point links of a one-shape mesh bucket its connectivity as it stands. At
  20M hexahedra on 8 threads, a fresh `extract_surface` went from about
  430 ms to 193 (VTK 249), at 20M tetrahedra from about 300 to 166 (VTK
  216).
- Metal reuses the buffers of arrays that have died (1 MiB or more, by
  capacity): a new buffer's pages are mapped and cleared on first touch at
  about 13 GB/s, so a filter's large temporaries cost more than its kernels
  (640 MB took 50 ms to first fill). At 20M hexahedra on an M1 Max,
  threshold went from 67.7 to 25.4 ms, contour 41.6 to 28.7, slice 50.1 to
  32.3, cell centers 14.7 to 6.8 (VTK on 8 threads: 152, 73, 59, 301). A
  reused buffer is not cleared, so `tack.zeros` fills it; a new array's
  contents were always unspecified.
- Small meshes pay less around their kernels: filters keep their constant
  tables (clip's cases, corner orders) on each device instead of uploading
  them per call (`_common._table`); `exclusive_scan(..., close=True)` writes
  the total at `output[n]` in the scan's own launch, so CSR offsets need no
  launch of their own; and clip scans its five kinds of counts in one
  launch below 2048 cells. A clip of 360 triangles went from 1.0 ms to 0.49
  (VTK's `vtkClipPolyData` 0.14), from 28 launches to 17.
- `extract_surface` of a 2D mesh, by default, is the mesh itself -- its cells
  passed through, hidden ones left out -- as VTK's geometry filter passes
  polygons; it was the boundary edges, which `extract_surface(data,
  boundary_faces(data))` still gives.
- `gradients` of 2D cells -- triangles, quads, pixels and polygons, in a 2D
  mesh or among 3D cells -- lie in the cell's plane, as VTK's and Viskores'
  do (matched to `vtkGradientFilter` and `vtkCellDerivatives` on a curved
  surface, and to Viskores' `UnitTestGradientExplicit`); they were zero. A
  polygon's is Green-Gauss over its edges, exact for a linear field. Lines
  and vertices still give zero.
- `cell_geometry` and `cell_quality` of shape cells no longer derive the
  mesh's faces: a shape's faces walk from its tables (`_FaceWalk` is in every
  shape cell's domain view), so on a fresh mesh of 1M tetrahedra volumes take
  6.5 ms instead of 24 (VTK's `vtkCellSizeFilter` 39 ms); polyhedra and
  polygons still derive theirs. Deriving faces computes each side's
  orientation in the pass that matches the sides (1.5 ms less at 1M
  tetrahedra).
- `uniform_grid` takes an origin and spacing of one value per size (a 2D
  image no longer needs a third spacing) and a 2 x 2 direction for a 2D
  image. `DataSet.rebased` keeps an image an image, its corner moved in double
  precision, and a rectilinear grid its axes: a raster at survey coordinates
  rebased near them holds its points exactly, and stays one to deposit onto.
- The documentation's datasets and visualization chapters are now the
  **Visualization Guide**, a tab of its own (`docs/vis-guide/`) with
  `tack.dataset`'s reference and design records; the User's Guide is about
  kernels. Pages moved without redirects. New chapters: Point Clouds and
  Particles (LiDAR cleaning and gridding, SPH resampling, halos in a periodic
  box, each a checked example) and Algorithms Across Pieces (split from Work
  with Data in Pieces, with running over MPI ranks and point clouds in
  pieces).
- **Breaking:** `Planes(origins, normals)` follows the data's precision, as
  the other implicit functions do, and takes no `dtype`: filters evaluate a
  copy of its planes in the geometry's floating type on its device, made
  once. A plane at x = 0.1 was off by about 1e-9 on f64 data unless
  `dtype=tack.f64` was asked for.
- `tetrahedralize` makes tetrahedra that meet face to face, as VTK's
  `vtkDataSetTriangleFilter` does. Viskores' fixed tables split each
  hexahedron the same way, so neighbours cut their shared face along
  different diagonals: on a 2 x 2 x 2 cube of hexahedra, `extract_surface`
  of the tetrahedra returned every inner face too (96 triangles for 48).
  Now a structured grid's hexahedra take 5 tetrahedra, the two mirror splits
  alternating with the parity of (i, j, k), and any other cell is the cone
  from its corner of smallest point id over the faces without it, each
  quadrilateral face split from its own smallest corner, whatever order the
  cells and points come in: hexahedra and voxels give 6 tetrahedra (were 5),
  wedges 3, pyramids 2, as VTK's counts.
- `probe`, `advect` and `streamlines` given no locator reuse the one the
  mesh's topology keeps (`cached_locator`), instead of building one per
  call: probing 100k points in 1M hexahedra again takes 13 ms instead of
  48 on an M1 Max's CPU. A locator records its geometry's state, and is
  rebuilt (the kept one) or refused (one passed in) once Tack has written
  the positions, through a write count on every array's storage
  (`DeviceBuffer.version`, which kernel launches and host writes keep, at
  no measurable cost per launch). A locator passed in now serves any
  dataset of the same topology and geometry, not only the dataset it was
  made from. Writes through memory shared with another library are not
  seen.
- `contour` and `slice` where the isovalue equals a point's value: the
  crossings of every edge at that point are now that point, once, and a
  triangle, polygon or line left with two corners at one point is dropped,
  as warp-simdata does. Before, each edge's crossing there was its own
  point, and the cells between them had no area or length: 702 of 45k
  triangles on a 1M-hexahedron grid whose plane of points lay on the level,
  23% of the triangles on 10k tetrahedra. The area is unchanged; the cell
  and point counts now differ from VTK's and Viskores', which keep them.
  On every path: shapes, polyhedra, 2D lines and per-cell (`L2`) lines.
- A topology made from a stored cell-types array groups its cells 16× faster
  on the CPU at 1M cells (36 → 2.2 ms), 7× at 100k: the types are counted by
  chunks into private tables, with no atomics. With one counter per type, a
  mesh of one type, as most readers give, put every worker on one atomic
  counter. A fresh unstructured mesh paid this before its first filter
  (found comparing with warp-simdata on the CPU).
- `gradients` and `flow_quantities` work on polyhedra, which have no
  reference element: each cell's gradient is Green–Gauss over its faces,
  each fanned into triangles from the mean of its points, which is exact for
  a linear field and, on a parallelepiped, the trilinear gradient at its
  center. At the points it is the average of the cells' gradients, as on
  shapes. Polygons and polylines get zero, as lower-dimensional cells do.
- `cell_centers` and `values_at_centers` work on polyhedra, polygons and
  polylines: a cell's center is its cell point, the mean of its distinct
  points, and point data takes the mean of their values there (the value
  of an interpolant linear on tetrahedra fanned from that point, so exact
  for a linear field); cell data is its own value.
- `CellLocator` and `probe` work on polyhedra, locating points in
  sub-tetrahedra: each face fanned from its points' mean, the fans from
  the mean of the cell's points. Point data is linear on each (exact for
  a linear field, continuous across faces). `find` returns `(cells, tets,
  pcs)` there; field views evaluate with `tet_value(c, t, pc)`, and the
  cell view gives `locate`, `cell_point`, `face_point` and `tet_jacobian`.
  A cell not star-shaped from its cell point is refused.
- `advect` and `streamlines` run through polyhedra, located in the same
  sub-tetrahedra (the kernel's `_PolyhedralMesh` reuses the views'
  location code), with point velocities linear on each and cell
  velocities constant: RK4 through a linear field is exact, and leaving
  the mesh bisects and pushes out as on shapes.
- Implicit topology arrays. An `UnstructuredTopology`'s `types`, `offsets`
  and `connectivity` may be implicit arrays, kept as they are rather than
  stored: a `ConstantArray` of types for cells all of one shape, a
  `CountingArray` of offsets for cells all of one size, and a
  `CountingArray` connectivity for cells that each have their own points
  (`tack.dataset.topology.uniform_cells` makes the first two). Filters
  whose cells are all one shape now give them: `contour` and `slice`,
  `tetrahedralize`, `triangulate`, `extract_edges`, the point extractions
  and `probe` (vertex cells: all three implicit), and `threshold`,
  `extract_cells`, `extract_geometry` and `mask` of a topology of one shape.
  A million triangles store their 12 MB of connectivity and nothing else,
  against 5 MB more before. `dataset_to_vtk` and `vtk_to_dataset` map them
  to and from VTK's `vtkConstantArray` and `vtkAffineArray`, through VTK's
  pythonic API (`ug.cells = (type, vtkCellArray(offsets=...,
  connectivity=...))`, `cells.offsets_array`, `grid.cell_types`). Code that
  reads a topology's arrays as `tack.Array`s should use `numpy()`,
  `np.asarray` or `tack.arrays.materialize`, which accept both.
- **Breaking:** `CellLocator.find` returns `Locations(cells, pcs, tets)`, a
  named tuple of three on every dataset, where it returned `(cells, pcs)`,
  or `(cells, tets, pcs)` on polyhedra. `tets` is `None` except on
  polyhedra. `cells, pcs = locator.find(p)` becomes
  `cells, pcs, _ = locator.find(p)`, or reads `.cells` and `.pcs`.
- `CellLocator.find` and `probe` search from the points where cells can be read
  by id (`tack.dataset.by_id`: an order-1 geometry over the linear shapes,
  structured grids, polyhedra): one launch over the points instead of two
  passes over the cells and two groupings, and `probe` reads point and cell
  data by cell id, one launch per field. Advection shares the same search
  and evaluation. Probing 100k points warm: 13.5 → 5.8 ms (10k cells) and
  30 → 16 ms (1M) on an M1 Max's CPU, 15.3 → 8.3 ms and 23.5 → 6–8 ms on its
  GPU. The cell-side search remains for what `by_id` cannot serve, and for a
  fresh locator's first query of few points in many cells, which it answers
  without building the cell bins; both give the same answer.
- Structured grids derive their faces and edges by arithmetic, in one pass
  over the cells with no sort: ids, sides, orientations and signs identical to
  the general derivation's. On a grid of 1M hexahedra, faces 24 → 7.9 ms and
  edges 22 → 9.5 ms on an M1 Max's CPU, 32 → 13.6 ms and 35 → 7.8 ms on its
  GPU; up to 7× on small grids.
- `CellLocator`, `probe` and advection run on polygons, and `CellLocator` and
  `probe` on polylines. A polygon is fanned from the mean of its points into
  triangles, and a point within the tolerance of one is located there (held
  when its projection falls in it), point data linear on each (exact for a
  linear field); a polyline locates on its segments, point data linear along
  them. Polygons not star-shaped from their mean point are refused, as such
  polyhedra are.

## 0.5.0 — 2026-10-22

The headline: the API freeze. Arrays are NumPy-style (`tack.Array` is
storage; "field" means a dataset's field only), devices are values
(`tack.device("cuda:1")`), kernels use NumPy's and GLSL's names in place
of Taichi's, and the dataset package is `tack.dataset`, with one name per
operation: see the **Breaking** entries under *Changed*. Around it:
several devices of one type in one process, datasets that compute on their
own device, launches queued on every GPU backend with `asyncio` waits,
compiled kernels kept on disk, a debug mode that checks every array index,
a CPU thread pool compiled from LLVM IR, indexable device classes with
`tack.arrays`' implicit arrays in the core, and structure-of-arrays
storage. Windows works for the first time, on the CPU and on CUDA.

Validated on Linux (CPU; CUDA on an RTX 4060 Ti, an RTX 6000 Ada, an H100
and four A100s in one process; HIP on two MI210s; Level Zero on an Intel
Max 1100), macOS (CPU and Metal on an M1 Max) and Windows Server 2022 (CPU,
and CUDA on a T4), with debug mode and both id widths of the dataset tests,
together with tack-examples' 57 examples. The dataset filters agree with
VTK's at 10k to 1M cells and with Viskores' on CUDA, and run faster than
0.4.0's on the CPU and on CUDA, the backends compared. One thing is untested: memory export
(`Array.export_memory`) on Windows. A GPU in TCC mode, as Data Center GPUs
run on Windows Server, cannot export at all.

### Added

- Windows. The whole suite passes on Windows Server 2022 on the CPU and
  with CUDA (a T4), after the fixes below: the disk cache stopped writing
  when Windows refused to replace an open file, and CUDA failed on a GPU
  without memory pools. The CPU's fan-out takes the `ThreadPoolExecutor` there, as
  before: the JIT-compiled pool is POSIX only, so a launch needs more work
  before it is worth threading (a fan-out costs 70–130 µs on Windows
  against a few on Linux).
- Indexable objects. A `@tack.device_class` that defines `__getitem__`,
  `__setitem__` and `__len__` as `@tack.func` methods is indexed in kernels
  as an array is: `obj[i]`, `obj[i] = v`, `obj[i] += v` (the index evaluated
  once) and `len(obj)`, which may bound the parallel loop. Several indices,
  `obj[i, j]`, arrive as one vector. One kernel then takes a stored array or
  any indexable object, compiled once for each. An instance tuple of
  numbers, `self.shape = (nx, ny)`, is one runtime scalar per entry, read
  as `obj.shape[k]` with a literal `k`; a new shape does not recompile. A
  class tuple, `shape = (4, 4)`, is read the same way and compiled in. A
  store without `__setitem__`, and an atomic on an indexable object, are
  refused when the kernel compiles.
- `tack.arrays`: the implicit arrays (`CountingArray`, `ConstantArray`,
  `CartesianProduct`, `UniformCoordinates`, `Permutation`, `Concatenate`,
  `View`, `Strided`, `ExtractComponent`, `Components`, `Function`,
  `Transform`, `Cast`, `RandomUniform`, `RandomNormal`) move from
  `tack.dataset.arrays` into tack-core and are indexable objects: any
  kernel takes them as it takes a stored array, `a[k]` and `len(a)`, not
  only through a dataset field. **Breaking:** import them from
  `tack.arrays`; `tack.dataset` no longer re-exports them.
- `tack.algorithms` (scans, sorts, `gather`, `unique`, `reduce_by_key`, the
  statistics, `copy`) take implicit arrays as inputs, read in their kernels
  without storing them, and implicit arrays have `sum`, `min`, `max` and
  `mean` with a stored array's semantics (computed into a temporary, then
  reduced). An implicit array is on the device of the arrays it holds
  (`a.device`), so these run and allocate there.
- Writable implicit arrays: a kernel's store into a `View`, `Strided` or
  `Permutation` lands in the array it wraps (a permutation's is a scatter),
  `Components(x, y, z)` writes each component to its own array, and
  `Concatenate` and `ExtractComponent` write through too. An array is
  writable when what it writes through is; a store into a computed array
  is refused when the kernel compiles, and a read-only stored array stays
  read-only behind a wrapper.
- Structure-of-arrays storage. `tack.soa(x, y, z)` makes a vector array
  (a matrix array with `dtype=tack.mat2()`...) whose components are
  existing arrays of scalars, without copying, so data that arrives one
  array per component (another library's tensors through `from_dlpack`,
  a device pointer through `array_from_ptr`, on the CPU or a GPU) is a
  `tack.Array` as it is. `layout="soa"` on `empty`, `zeros`, `ones`,
  `full`, `array`, `asarray` and the `*_like` constructors (which keep a
  prototype's layout) makes new ones; `a.layout` is `"aos"` or `"soa"`
  and `a.components` gives the component arrays. Kernels index an SOA
  array exactly as an interleaved one, loads, stores, components,
  augmented stores and vector atomics, each component its own kernel
  parameter; one kernel takes either layout, compiled once for each. On
  the host it reads and writes as any vector array, and `copyto` and
  `tack.array(a, layout=...)` convert between layouts. Not available for
  an SOA array: a scalar atomic at a flat component index, and export as
  one buffer (DLPack, `export_memory`, VTK), which its components do.
- `tack.__version__`, the installed tack-core's version, and a docstring
  for `tack` (`help(tack)` said nothing: the docstring followed code).
- NumPy-style arrays. `tack.Array` is backend storage; `tack.dataset.Field`
  is a function space plus coefficients, and "field" means only that now.
  Constructors take the shape first, as NumPy's do: `tack.empty(shape,
  dtype=f32)`, `zeros`, `ones`, `full(shape, fill_value, dtype=None)`,
  `arange([start,] stop[, step], dtype=None)`, `empty_like`, `zeros_like`,
  `ones_like`, `full_like`, `array(obj, dtype=None)`, `asarray`, `concat`
  and `array_from_ptr(ptr, shape, dtype)`. Any NumPy dtype is accepted
  where a dtype is (`np.float32`, `"float32"`). Python floats default to
  f32 and ints to i32 (i64 if a value needs it), since f64 is not on every
  backend; a NumPy array keeps its dtype. Vectors and matrices are element
  types: `tack.empty((n,), tack.vec3())`, `vec(n, dtype)`, `mat2`...`mat4`,
  `mat(rows, columns, dtype)`; `shape` counts elements and `a.numpy()`
  gives `(*shape, 3)`. Data moves with `a.numpy()` or `np.asarray(a)`,
  `await a.numpy_async()`, `tack.copyto(a, host)` (strict, as
  `np.copyto`: the host values have `a.numpy()`'s shape or broadcast to
  it) and `a.fill(value)`. Arrays also have `ndim`, `nbytes` and
  `reshape(*shape)` with one `-1`.
- `asyncio` waits: `await tack.sync_async()` and `await
  a.numpy_async()` wait for the device in a worker thread, so an
  application's event loop (trame, Jupyter, a server) keeps running
  while kernels finish; kernel calls themselves are unchanged. On CUDA
  the wait is an event on the context's stream, on Metal the last
  committed command buffer, on Level Zero a host-visible event signalled
  by a barrier on the backend's command list, on HIP an event on the
  device's stream; on the CPU, whose launches complete before returning,
  the wait is inline. During a 490 ms CUDA kernel another task ran 436
  times while one awaited `sync_async`, and not once under `tack.sync()`;
  during a 685 ms Level Zero kernel (Intel Max 1100), 311 to 384 times;
  during a 1.45 s HIP kernel (MI210), 1353 times.
  Level Zero's `to_numpy`, `read_range` and reductions now wait on such
  an event outside the backend's lock, so another thread may launch
  while one waits. Every HIP wait (`tack.sync()`, reads, reductions) is
  on an event too, through ctypes: hip-python's wrappers hold the GIL
  while the runtime waits, which stopped every other Python thread for
  the length of the kernel, and ROCm's `hipStreamSynchronize` holds the
  stream, so a launch from another thread took the whole wait (350 ms)
  and now takes 60 µs. CUDA reads wait for the stream before they copy,
  for the same reason: a copy to pageable host memory waited with the
  context held, and a launch from another thread took the whole wait
  (910 ms on an H100), now 43 µs.
- Compiled kernels are kept on disk, so a later process starts warm: the
  first contour of a 21³ grid in a new process went from 445 ms (CPU) and
  1.1 s (Metal) to 31 ms on an M1 Max. The lowered kernel and each
  compiled variant are stored, keyed by content; an edited kernel, device
  function, constant or template class, another Tack or another machine
  compiles again. `TACK_CACHE_DIR` chooses the directory (by default the
  platform's per-user cache directory) and `TACK_CACHE=off` turns it off;
  `tack.cache.configure()` does either from Python. For MPI jobs on a
  shared file system, `python -m tack.cache pack` gathers a warm-up run's
  entries into one bundle, which each rank reads once (or rank 0 reads and
  broadcasts, `tack.cache.load_bundle(comm=...)`); under MPI nothing is
  looked up or written per kernel. The entries take at most
  `TACK_CACHE_SIZE` (1 GiB by default); the least recently used go first.
  Every backend stores compiled code: CUDA its PTX, HIP its code objects,
  Level Zero its native binary, with the SPIR-V to fall back on. Debug
  mode's kernels are stored apart from the others.
  See *The Kernel Cache* in the User's Guide.
- Debug mode: `tack.set_debug(True)` or `TACK_DEBUG=1` checks every
  array index in every kernel, one dimension at a time. An
  out-of-bounds access is skipped (a load gives 0) and the call raises
  `IndexError` naming the access, its file and line, the index and the
  extent, and how many more were skipped; an access inside a device
  function names it. It runs on every backend, compiles separately from
  normal kernels, and does not compile per array length. Without it,
  generated code is unchanged.
- `pytest -m quick`, a module per area of the code, 80 s on four
  workers, and `pytest -n auto` for the whole suite across processes
  (pytest-xdist is in the `dev` extra): 4:30 on eight workers where the
  serial run took 28 minutes, with the same counts. See CLAUDE.md,
  "Tests, in three tiers".
- An `interop` extra for VTK and MFEM (PyMFEM), the references the dataset
  tests compare against: `uv sync --extra interop` installs them from the
  lock, where a later `uv sync` used to remove a hand-installed copy. MFEM
  is left out where PyMFEM has no wheels (anywhere but macOS on Apple
  silicon and Linux x86_64).

- Datasets compute on their own device. A topology, field, set or dataset
  is on the device its arrays are on (`.device`), and every algorithm,
  derived entity, `for_each`, `Set` and `CellLocator` runs and allocates
  there, whatever the default device; before, any dataset off the default
  device failed at its first launch ("arrays on two devices"). Parts on two
  devices are refused when a dataset is made. `DataSet.to_device(device)`
  copies one; `uniform_grid`, `rectilinear_grid`, `wavelet` and `tangle`
  take `device=`. Implicit arrays are on the device of the arrays they
  hold: `ConstantArray`, `UniformCoordinates` and `CartesianProduct` take
  `device=`, `to_device` moves one, and one made from arrays on two devices
  is refused.
- Edge sets and `extract_edges`. `Set(data, "edges", ids)` selects edges by
  their id in `topology.edges()`; an edge set is an iteration domain for
  `for_each`, and `as_polyhedra` carries it. `extract_edges(data,
  edges=None)` gives the edges (all of them, or an edge set, its name or
  ids) as line cells on the same points, each from its lower point id to its
  higher, as VTK's `vtkExtractEdges` finds them: values on edges become
  cell values and edge sets cell sets; cell fields are left behind.
- `interpolate(data, function, space)`: a field whose values are a
  `@tack.func` of position at the space's nodes (`H1` of either order, `L2`,
  `Constant`, values on points or cells), evaluated through the geometry
  (curved included) without storing node positions. A node shared by
  several cells is written once, by the first cell that has it; before, a
  quadratic field took projecting the geometry into its space, a stored
  array of node positions, and a kernel over them.
- Order-2 `H1` on triangles, quads and pixels: the six-node triangle and the
  nine-node biquadratic quad, the faces of the tetrahedron's, wedge's and
  hexahedron's order-2 elements. A mesh without quad faces of solids
  (tetrahedra, triangles, quads) no longer derives faces for its order-2
  layout.

### Changed

- The documentation site keeps every release's docs: `/latest/` (the newest
  release, where the site's root goes), `/0.4/` and on, and `/dev/` for
  `main`, with a version selector and a banner on all but the latest.
- `advect` and `streamlines` keep their particles on the dataset's device:
  seeds given as an array of 3-vectors are copied there, the particles'
  positions are the output's geometry as they are, and `streamlines`
  chooses and numbers its paths with scans. Only counts come back to the
  host; the particles and seeds used to go there and back.
- `extract_surface` keeps a solid's order-2 `H1` fields and geometry at
  order 2, the trace of the solid's: a face's corners, edge middles and (a
  quad's) center are the surface cell's nodes, so their values are gathered
  by entity id, not evaluated. It kept only the points' values before; a
  2D mesh's boundary lines still do.
- Offsets and ids that are a linear ramp are computed where they are read
  (`CountingArray`) rather than stored: an order-0 `L2` space's offsets, a
  structured grid's `L2` offsets, and the edge groups' ids, which were also
  rebuilt on every `edges().groups()` call. Order-1 `L2` on an unstructured
  mesh already shared the connectivity's offsets.
- **Breaking: tack.dataset names, one per operation.** `external_faces(data)`
  is `extract_surface(data)`, whose `faces` now default to the boundary.
  `to_points`, `to_cells`, `discontinuous` and the public `traces` are gone:
  `project(data, field, H1(data) | Constant(data) | L2(data) |
  SideTraces(data))`. `point_cloud(data)` is `mask_points(data, 1)`. The old
  `divergence`, an outward sum with no volume, is `net_flux`, unchanged;
  `divergence` is now the finite-volume divergence, `net_flux` over each
  cell's measure. `statistics` returns a `Statistics` named tuple
  (`s.mean`, not `s["mean"]`), and `check_winding` a cell `Set`.
  `clip(..., invert=True)` is `clip(..., inside=True)`, and `contour`'s
  `isovalue` is `value`. The reference states the rules: what each kind of
  algorithm returns, and that the point filters are the cell filters'
  point twins (point fields carried, cell fields left behind).
- **Breaking:** CPU threading policy v1 is removed; the policy that has
  been the default since 2026-08-10 is the only one. `TACK_CPU_POLICY=v2`
  (or unset) is still accepted, and any other value, `v1` included, raises
  `ValueError`. `CPUBackend.policy` and `CPUBackend.margin` are gone (the
  margin in use is `_margin()`).
- The install instructions say how to install from source: Tack is not on
  PyPI yet, and `pip install tack-core` found nothing. The README, the
  guide, the backends reference and the errors for a missing backend now
  give `uv sync --extra ...` or `pip install -e 'packages/tack-core[...]'`
  in a clone.

- **Breaking:** `tack.deprecated` is gone: the first vis worklets
  (`flying_edges`, `compute_normals`, `cell_to_point`, `amr_blanking`) and
  the finite-element helpers in `tack.fe`, with the examples written on
  them (contour, marching cubes, flying edges, point-to-cell, `mini_vtk`,
  the `tack.fe` isolines and isosurfaces). `tack.dataset` supersedes them:
  `contour` for `flying_edges`, `to_points` for `cell_to_point`,
  `gradients` and `project` for the finite-element helpers.
- **Breaking:** `tack-rendering` (path tracer, rasterizer, volume renderer)
  is archived in `archive/tack-rendering` and no longer part of the
  workspace, the tests or the documentation, until the dataset API and its
  implementation settle. Its user-guide chapter, the isosurface tutorial,
  the two cross-package examples and the rendering benchmarks moved with it.
- **Breaking:** the dataset package is now `tack.dataset`, replacing
  `tack.data`. Use `import tack.dataset as td`; submodules move with it
  (for example, `tack.dataset.algorithms`; the implicit arrays have since
  moved on to `tack.arrays`). The dataset model and algorithm signatures
  are unchanged.

- **Breaking:** devices are values, spelled as the array API spells them,
  and `tack.init` is gone with the `tack.cpu`/`metal`/`cuda`/`hip`/
  `level_zero` constants. `tack.device("cuda")` (or `"cuda:0"`, `"cpu"`,
  `"metal"`, `"hip"`, `"level_zero"`) is a `tack.Device`, the same one
  every time; options (`num_threads`, `external_context`) make a new one.
  `tack.set_default_device(...)` chooses the default, `get_default_device()`
  reads it, and `with tack.device(...):` sets it in a block, per thread.
  `tack.set_debug()` replaces `init(debug=...)`; `TACK_DEBUG` is read at
  import. Arrays know their device (`a.device`, `a.to_device(...)`, and
  `device=` on every constructor), and a kernel runs where its arrays are,
  so changing the default no longer strands arrays made before it; arrays
  on two devices are refused, by `tack.algorithms`, `copyto` and `concat`
  under their own names. `tack.algorithms` run on their inputs' device,
  and `tack.sync()` and `tack.sync_async()` wait for every device.
  `"cuda:N"`, `"hip:N"` and `"level_zero:N"` reach every GPU, and
  `tack.device_count(type)` counts them; a bare `"cuda"` or `"hip"` is the
  device the thread already uses (a framework's current CUDA context, HIP's
  current device), else 0. With a second device in use, every CUDA or HIP
  operation makes its own context or device current and restores the
  thread's. DLPack carries the device number both ways.
  `tack.from_dlpack(x)` keeps memory on its own device when the default
  cannot wrap it, as the array API's does: CUDA memory with the CPU as
  default is a CUDA array, host memory with Metal as default a CPU array;
  `device=` still refuses what it cannot wrap. `Backend.init_options`
  is `Backend.device_options`. A warm dispatch costs about 0.4 µs more
  (6.7 to 7.1 µs on an M1 Max) for reading its arrays' devices.
- **Breaking:** kernel values and loops are spelled as NumPy and GLSL
  spell them, and Taichi's names are gone. Vectors are `tack.vec2`,
  `vec3`, `vec4` (N scalars, one scalar for every component, or one
  N-vector); a list of scalars is still a vector. Matrices are `tack.mat2`
  to `mat4` (rows, or N*N row-major scalars), `tack.mat(row, ...)` for any
  shape up to 4x4, and `tack.identity(n)`; on the host the same names are
  element dtypes. A zero-dimensional array's element is `a[()]`, in kernels
  and on the host; `a[None]` is refused. `tack.ndrange` is `tack.ndindex`.
  Removed: `tack.Vector`, `tack.Matrix`, `tack.template()` (an annotation
  nothing read) and `data_oriented`.
- **Breaking** for code that reaches inside Tack: storage is "array"
  throughout the core too. The module is `tack.lang.array`; the backend
  contract's hooks are `allocate_array`, `fill_array` and `reduce_array`;
  the IR's nodes are `IRArrayLoad`, `IRArrayStore` and `IRArrayExtent`;
  messages, docstrings, the language contract and the design pages say
  "array" where they said "field". The User's Guide chapter is
  `02-arrays-and-types.md`.
- **Breaking:** the old storage API is gone, without deprecation:
  `tack.field` (use `tack.empty(shape, dtype)`), `field_from_numpy` and
  `field_like` (`tack.array`), `field_from_ptr` (`array_from_ptr(ptr, shape,
  dtype)`), `Vector.field` and `Matrix.field` (`tack.empty(shape,
  tack.vec(n, dtype))`, `tack.mat(...)`), `a.to_numpy()` (`a.numpy()`),
  `a.from_numpy(x)` (`tack.copyto(a, x)`) and `to_numpy_async`
  (`numpy_async`). `zeros`, `ones`, `full` and `arange` take NumPy's
  argument order. The flat host layout of `Vector.field` and `Matrix.field`
  went with them: a vector array's `shape` counts vectors and its values
  download and upload as rows, and `copyto` refuses a flat array.
  `tack.interop.vtk`'s `vtk_to_field` and `field_to_vtk` are
  `vtk_to_array` and `array_to_vtk(array, ...)`.
  `local_array_like` and `shared_like` remain allocation-only helpers.
- A scan's small levels are one launch: a level of at most 4096 elements
  on the CPU, or one tile on a GPU backend, is scanned in order by one
  thread, and the total is read as one element instead of copied by a
  kernel. A million elements take three launches on the CPU instead of
  eight; a contour at 1M cells makes 26 launches instead of 40 and a
  threshold 18 instead of 33. Floating sums now match `np.cumsum` bit
  for bit through 4096 elements on the CPU and 2048 on GPUs (512 and 8
  before).
- A warm dispatch costs less host time: the fast path keeps each
  array's signature and each buffer's address and byte range once
  computed, holds the variants it resolved instead of looking them up,
  learns each template class's attribute layout once, checks array
  overlap in one walk of the ranges, packs the pool's arguments with one
  `struct.pack_into`, and skips the template scope and the texture
  walks when a call has none. A serial dispatch of a kernel with eight
  array arguments went from 66 to 34 µs on a 2 GHz Xeon, and inside a
  dataset filter the host time per launch from 149 to 130 µs.
- A CPU launch wakes only the pool workers it uses: each worker waits on
  its own mailbox, a launch of `k` threads posts to `k - 1` of them, and
  the rest sleep on, so the `k` of every launch are the same `k`.
  A hot fan-out of two threads on a pool of sixteen cost 11 µs waking
  every worker and costs 4.4.
- `tack.empty` returns uninitialized storage on every backend; `tack.zeros`
  is the call that pays for zeros. The CPU backend used to fill every new
  array on the calling thread, which at 5M cells on eight threads of a
  2 GHz Xeon was half of threshold's and cell-to-point's time and grew
  with the thread count. An array is now first touched by the kernel that
  writes it, and a CPU fill of 256 KiB or more (so `zeros`, `ones`,
  `full`) runs as a kernel through the dispatch, so the threads share it
  and the pages land where they will be used; the executor's threads are
  made under the NUMA interleave policy as the pool's are. The dataset
  filters' fully written flag and offset arrays take plain arrays.
  `Backend.fill_array` is the hook. On Linux, `TACK_CPU_MALLOC=1` also
  tells glibc to keep freed pages (`mallopt`: the mmap threshold at its
  32 MiB cap, no trim under 4 GiB of free heap, the heap grown 256 MiB
  at a time), since a filter reallocates the same temporaries on every
  call and takes their page faults every time: threshold at 5M on 8
  threads of one NUMA node 126 → 104 ms, slice 76 → 65. Off by default:
  it changes `malloc` for the whole process, and unbound on a 28-core
  EPYC it did not pay.
- The CPU backend fans a range out through a thread pool JIT-compiled from
  LLVM IR, with one ctypes call per launch and no Python on the workers'
  path, instead of one `ThreadPoolExecutor` task per worker. A fan-out that
  cost 60 µs per worker on a 2 GHz Xeon (1.2 ms for eight after an idle
  gap) costs a few microseconds, so short launches thread too: the dataset
  filters at a million cells went from 1.05× to 1.9× on eight threads
  there. Workers claim chunks from a counter rather than taking one equal
  share each, so one slowed by a sibling hyperthread or an uneven range
  takes fewer. Dispatches from several
  Python threads take turns at the pool, and a pool whose workers cannot
  all be started is dismantled before the executor takes over.
  `TACK_CPU_POOL=python` keeps the executor, which remains the fallback
  on Windows.
- A dispatch whose argument signature was seen before resolves its
  variant from that signature instead of rebuilding the key, and the
  CPU backend passes array addresses rather than typed pointer objects:
  a warm dispatch costs 40 µs instead of 70 on a 2 GHz Xeon, on every
  backend, which for a filter of forty launches is a few milliseconds.
- NUMA interleave no longer replaces a bound thread's memory policy: a
  process started under `numactl --membind` keeps its binding, and the
  probe at the CPU backend's start puts back the policy it found instead of the
  default.
- CUDA and HIP launches are queued on a stream and return at once; the
  host waits where it reads an array (`numpy()`, indexing, reductions,
  DLPack export) or calls `tack.sync()`. Fills and host writes queue in order
  with the launches, and an array dropped while a queued launch still uses
  it is freed after that launch (stream-ordered memory). A contour of a
  million cells on an RTX 4060 Ti went from 9.6 to 7.1 ms, now the cost
  of its 31 launches' host side alone; on an MI210, from 6.3 to 4.8 ms.
  Errors of a launch surface at the next wait rather than at its call, as
  on Metal.
- Level Zero launches are queued the same way, on one asynchronous
  immediate command list with a barrier after each command; fills and
  host writes queue in order with them (from a private copy of the host
  array, so it may change at once), and a dropped array is freed after the
  work queued before it (the driver's deferred free). On an Intel Max
  1100, 100 small launches went from 8.7 to 2.6 ms, and a contour of a
  million hexahedra from about 24 to 18 ms. Ordering by barriers rather
  than an in-order list: Intel's in-order lists cost about 100 µs of device
  time per kernel on driver 25.05. Copies to the device up to 16 MiB and
  from it up to 64 KiB go through pooled USM host memory: from pageable
  memory the driver does not return from appending such a copy until the
  queue has drained, so a launch whose scalars were uploaded behind a
  680 ms kernel waited 682 ms, now under 1 ms, and 100 small launches with
  a changing scalar take 5.4 ms rather than 8.7.

- A CPU array of 1 MiB or more takes the NUMA interleave policy for its
  range when it is allocated (`mbind`), so its pages spread over the
  memory nodes whichever thread touches them first, a serial launch,
  `from_numpy` or a NumPy temporary included, and keep spreading when
  glibc hands the block out again. Pages already placed on a reused block
  are moved once (`move_pages`, in 2 MiB steps, so no huge page is split).
  A thread bound with `numactl` keeps its binding. On a two-node EPYC at
  28 threads, tet contour's and slice's reused blocks had sat 77–89% on
  one node, and those filters ran 10–20% slower.
- Allocating a vector or matrix array costs a third of what it did:
  shape products use `math.prod`, where `np.prod` made a NumPy array of
  each small shape tuple, six times per vector allocation (10 µs each on
  a 2 GHz Xeon). The small point filters on CUDA (cylindrical and
  spherical coordinates, warp, point transform, point elevation) lost
  80 µs a call at 10k–100k points against 0.4.0, and are now faster than
  it.

### Fixed

- CUDA on a GPU without memory pools, such as a Data Center GPU in TCC mode
  on Windows: every allocation failed, since the backend always used
  `cuMemAllocAsync`. Such a device allocates with `cuMemAlloc`, and a
  dropped array is freed once the stream is idle, at the next allocation
  or sync, because `cuMemFree` waits for the whole device (1.26 s behind
  an unrelated kernel on a T4). Exporting memory from such a device, which
  has no virtual memory management either, now says so.
- The disk cache stopped storing for the rest of the process on Windows
  when another process had an entry open: Windows refuses to replace an
  open file, and the cache took that for an unwritable directory. It now
  keeps the entry already there, since entries are named by their content.
- On Intel GPUs with IGC 2.7.11 (compute runtime 25.05), a vector or matrix
  component selected at a runtime index defined by a conditional
  expression in a sequential loop could read the wrong component: IGC
  miscompiles a chain of 64-bit compares. The selector is compared as an
  i32 (recorded in the vendor workarounds).

- The nearest point of a cell, outside a quad, hexahedron, wedge or pyramid,
  was its position at clamped parametric coordinates, exact only for an
  axis-aligned box or a rectangle (TA-39). It is now the nearest point of a
  solid's faces, or of the quad itself, a quad taken as two triangles (exact
  when planar), as VTK finds it for a hexahedron since 2026-07. `CellLocator`,
  `probe` and advection accept a point within a tolerance of it, so a point
  just outside a skewed cell is found where it can be.
- A parallel loop bounded by a local, `n = x.shape[0]; for i in range(n)`,
  failed at dispatch ("Cannot resolve loop range expression"), and so did
  `len(obj)` for an indexable object whose `__len__` returns an array's
  `shape[0]`. Such a bound is now read from the arguments, and a new
  length reuses the compiled kernel.
- A device-class instance attribute that shadows a class constant with
  something other than a number (`obj.SCALE = [1, 2]`) made every call
  fail with "unhashable type: 'list'". Such an attribute now drops the
  constant, as a subclass attribute does; a method that reads it gets the
  error that names the attribute.
- HIP works with hip-python 10, the newest on PyPI and what
  `tack-core[hip]` installs. Tack imported `hip.LONG_VERSION`, which 10 does
  not have, so HIP read as unavailable and its tests skipped; it also lacks
  `hip.chip`, and returns the output structs of `hipPointerGetAttributes`
  and `hipGetDeviceProperties` instead of filling them, and gives the device
  name as `str`. 7.x keeps working.
- Grids with a size of 1 before another axis -- `uniform_grid((1, 3))`, a
  VTK image of dimensions `(1, m, n)` -- were refused; so were grids of one
  point or none (`(1, 1, 1)`, `(0,)`, `wavelet(extent=0)`). A size of 1 now
  adds no cells in any position, and a grid with no axis of two points has
  none.
- Cells of no volume and faces of no area (coincident points, a grid of
  spacing 0) have finite geometry: measure 0, the centroid at their points'
  mean and zero normals, so the flux through them is 0. Their gradients,
  flow quantities and Perot reconstruction are NaN, as before, and now
  documented.
- A contour or clip crossing beside an infinite or NaN value lies at the
  finite end, where the crossing and the fields carried to it were NaN.
- `algorithms.histogram` and `entropy` refuse a range that is not finite
  (as `numpy.histogram` does), a reversed range and fewer than one bin;
  `entropy` of no values says so, and of one value is `0.0`, not `-0.0`.
- A mesh of line cells (a 1D grid, VTK lines) derived no faces, so its
  `external_faces` were empty and `face_geometry`, `upwind_flux`,
  `divergence` and `perot` failed with an internal error. Faces of lines
  are not derived yet, and asking now raises `NotImplementedError`.
- `extract_cells` takes a cell `Set`, or its name in `data.sets`.
- `dataset_to_vtk` writes a grid of one point or none as a grid.
- On HIP, a variable could hold two values: hipRTC fused a product into one
  read of it while another read it rounded, so `x - x` was the product's
  rounding error rather than zero. A contour of a computed field (such as
  `Function(ramp, n)`) then put some crossings at an edge's end. HIP now
  fuses multiply and add only inside one expression, as the language
  contract says.
- `extract_surface` checked neither face ids nor a face `Set`'s topology:
  an id past the face count, or negative, was read out of bounds and gave
  wrong cells silently, and another topology's set was used as this one's.
  Both are refused now, as `extract_cells` refuses them; out-of-range ids
  raise `IndexError`, counted on the ids' device.
- `tack.algorithms.norm`, `var`, `std`, `absmax`, `count_nonzero`, `dot`
  and `histogram` read a given `n` as scalar components of a vector array
  (`norm(v, n=2)` of 3-vectors took two components); `n` now counts
  vectors.
- Vector implicit arrays (`UniformCoordinates`, `CartesianProduct`,
  `Components`, a vector `ConstantArray`) failed to compile in `gather`,
  `unique`, `reduce_by_key` and the statistics; they give what the same
  vectors stored give. `unique` takes vector keys, stored or implicit.
- A kernel with a `return` shared one result array across calls, so two
  threads calling it at once could read each other's result. It is kept
  per thread.
- `vtk_to_dataset` refused a grid of `VTK_POLYGON` cells, which is how
  `dataset_to_vtk` writes a polygonal topology (a polyhedral contour or
  slice, `as_polygons`); it reads it back as a `PolygonalTopology`. A
  warped or transformed structured grid, which keeps its topology over
  explicit coordinates, could not be exported; it is a `vtkStructuredGrid`,
  and `vtk_to_dataset` reads one.
- 3D `contour`, `gradients` and `values_at_centers` refused a field of
  values on points (such as `point_ids` or a gradient at the points), which
  `project`, `threshold` and 2D contour read as order-1 `H1`; all of them
  do now.
- `Planes` stored its planes in f32 whatever the data; `Planes(origins,
  normals, dtype=tack.f64)` keeps f64 data at its precision.
- `tack.arrays.Strided` given NumPy data kept a NumPy dtype where a Tack
  type belongs; it stores the data as the other implicit arrays do.
  `Cast` converts to all ten scalar types.

## 0.4.0 — 2026-10-13

The headline: `tack.data`, a dataset API in `tack-vis`. Meshes, the
fields on them and over fifty visualization algorithms run on every
backend, and their outputs agree with VTK's on hexahedral and tetrahedral
meshes from 10k to 1M cells, and with Viskores' on CUDA. The core gained
what that work needed: queued Metal launches, workgroup scans and sorts on
GPUs, fills that stay on the device, 2D and 3D grids for `ndrange`, and
device classes that hold and iterate over one another. Some kernels now
compute different results: f64 kernels using float-literal locals, the
statistics in `tack.algorithms`, floating-point scans on GPUs (in the last
bits), and i8/i16 negation on CUDA drivers before 13.3; see *Kernels that
now compute different results*.

Validated on Linux (CPU; CUDA on an RTX 4060 Ti and an RTX 6000 Ada; HIP
on an MI210; Level Zero on an Intel Max 1100) and macOS (CPU and Metal on
an M1 Max), both id widths of the dataset tests, together with the 57
example checks in tack-examples.

### Added

#### The dataset API

- **`tack.data`** (in `tack-vis`): meshes, the fields on them, and the
  algorithms that work on both. It is described in the User's Guide
  chapter *Datasets and visualization* and the *Dataset API* reference.
  - **Topologies:** structured grids; unstructured meshes of the linear
    cells, vertex to pyramid; and polyhedra, polygons and polylines.
  - **Fields:** a field is a space (`H1` of order 1 or 2, `L2`, `Constant`,
    `Values`) and its values. The values may be an implicit array that
    stores nothing, such as a grid's coordinates.
  - **Sets:** named sets of points, cells or faces, carried through
    filters.
  - **64-bit ids** wherever a mesh needs them.
  - **Algorithms:** over fifty, among them contour (and López's on
    polyhedra), slice, clip, threshold, external faces, gradients,
    `project` between spaces, the cell locator and probe, advection and
    streamlines. All run on every backend.
  - **Your own kernels** see a mesh as the built-in ones do, through
    `for_each` and per-shape views.
  - **Interop:** `tack.interop.vtk` converts whole datasets with VTK, and
    `tack.interop.mfem` reads MFEM meshes and grid functions.
- **What the algorithms promise.** They check field ownership before
  dispatch, including empty inputs, and reject locators built for another
  dataset. `dot` and `cross` require matching dtypes, as `composite` does;
  use `Cast` for conversion. Fractional computations convert integer
  storage before arithmetic, so clipping differences do not overflow;
  integer magnitude and logarithms return floating point fields.
  Computation uses f64 for f64 input, otherwise f32, with the
  corresponding precision limits. `statistics` sums integer values exactly
  in f64 (to 2^53) instead of wrapping; its moments use the floating
  precision. Face positions refresh after editing L2 geometry. Shrink
  accepts quadratic geometry through its corner values, and probe samples
  a quadratic target's points while keeping its geometry. Advection and
  host-point probing preserve the source's coordinate precision.

#### Device classes

- A `@tack.device_class` object may hold others as attributes, to any
  depth: `self.base.get(k)` in a method, `obj.base.step` in a kernel. The
  tree is one template argument. Composed objects inline to the code one
  would write by hand, and a class holding itself is refused.
- A kernel can loop over a `@tack.device_class` object: `for c in cells:`.
  The class declares the attributes the loop runs through as
  `__tack_iterate__`, one to three names, fastest first. One makes `c` an
  integer over `range`; two or three make `c` the vector of indices over
  an `ndrange`, which as the parallel loop launches in its own shape. The
  object's methods take `c` as it comes, so one kernel runs over objects
  that iterate differently. See *Iterating an Object* in the User's Guide.

#### Runtime and algorithms

- `tack.sync()` waits for every launch queued on the current backend. It
  is a no-op where launches complete before returning.
- `exclusive_scan` and `inclusive_scan` take `total=False` to skip the
  total: nothing is read back to the host, so on Metal the scan stays
  queued with whatever follows. The radix sort's passes and the dataset
  algorithms that only need offsets use it; a 1M-key sort went from 10.7
  to 9.4 ms on CPU and from 7.7 to 6.3 ms on Metal.
- `argsort` and `sort_by_key` take more than 2^31 - 1 keys, with i64
  indices beyond that count; `argsort(..., index_dtype=tack.i64)` asks
  for them at any size.

### Changed

- `@tack.data_oriented` is now `@tack.device_class`, a name for what it
  does: its objects can be kernel arguments, and kernels specialize on
  their class. `@tack.data_oriented` remains as an alias, so existing
  code (and code written for Taichi) runs unchanged.
- The first visualization worklets moved to `tack.deprecated`, superseded
  by `tack.data` (see *Added*):
  - `flying_edges`, `compute_normals`, `cell_to_point` and `amr_blanking`
    are now under `tack.deprecated.algorithms`;
  - the finite-element helpers in `tack.fe` are now `tack.deprecated.fe`.

  They still import from their new places. `tack.algorithms` no longer
  re-exports them and now holds the compute primitives alone (scans,
  sorts, copies). Change `from tack.algorithms.flying_edges import ...` to
  `from tack.deprecated.algorithms.flying_edges import ...`, or move to
  `tack.data.contour`.
- The examples that used the deprecated worklets now use `tack.data`:
  - the isosurface tutorial, `examples/32_pathtrace.py` and
    `examples/37_insitu_pipeline.py` pass the contour's own device fields
    to the renderer;
  - `tack-vis/31_multiblock_flying_edges.py` is now
    `31_multiblock_contour.py`, one contour per block;
  - the `tack.fe` examples moved to `tack-vis/examples/deprecated/` and
    still run in the examples sweep.

  Examples that save an image write it under `results/`, as the others
  already did, instead of into the working directory.
- Metal launches are queued instead of waited for one by one. The host
  waits when it touches field memory, reduces a field, or calls the new
  `tack.sync()`; an error in a launch surfaces at that wait rather than
  at the call. Results are unchanged. To time a Metal kernel, call
  `tack.sync()` before reading the clock.
- The CUDA backend warns, at its first compile, when NVRTC is older than
  13. NVRTC 12.0 miscompiled one of the ported examples (lattice
  Boltzmann on an H100: density off by up to 1.6e-2, against 2.3e-6 with
  NVRTC 13.2). The warning names the remedy: the `nvidia-cuda-nvrtc`
  wheel for a CUDA version no newer than the driver's, which cuda-python
  finds before a system toolkit.

### Kernels that now compute different results

- `var`, `std`, `norm`, `absmax`, `count_nonzero` and `dot` in
  `tack.algorithms` no longer combine into a one-element f32 (or i32)
  atomic accumulator. Each forms its terms in the inputs' precision --
  f64 if any input is f64, otherwise f32, an integer input converted
  first -- and reduces them as `Field.sum()` and `Field.max()` do, so the
  field-reduction contract applies. f64 data is now summed in f64:
  `var` of 1e8 + 0..999 is exact where f32 gave nonsense. `count_nonzero`
  counts in i64 and no longer wraps past 2^31. Integer products no longer
  wrap before converting, and neither does the mean `var` takes of
  integer data (four equal i64 values of 2^62 had a variance of 2e37).
  `var` returns a Python float, not NumPy's float32, and `absmax`
  propagates a NaN.

- On GPUs, a floating-point `exclusive_scan` or `inclusive_scan` combines
  lanes in tree order, so it can differ from `np.cumsum` in the last bits
  from the ninth element on. CPU results are unchanged.

- A local assigned once, to float literals, takes the precision of what
  it meets where it is read, as the literal written there would: with an
  f64 `x`, `a = 0.1; y = x * a` uses the f64 nearest 0.1. It was an f32
  local, so f64 kernels computed with the f32 rounding of the literal.
  This includes literals a device function returns, such as a cell's
  parametric center returned as a vector of literals, which made shape
  weights f32 on f64 data. A local assigned only literals more than once,
  such as the result of a device function returning literals from
  several branches, takes the kernel's float precision, as a float scalar
  argument does: f64 when a field is f64. f32 kernels are unchanged.

- On CUDA, negating an i8 or i16 value (`-x`, `abs(x)`) and then widening
  it keeps the wrapped result: `-(-32768)` in i16 widens to −32768. On
  drivers before CUDA 13.3 it widened to 32768. NVIDIA's PTX assembler,
  which those drivers run on Tack's PTX, drops the sign extension after a
  16-bit `neg` or `abs`, on every architecture; ptxas 13.3 fixed it. The
  CUDA generator now negates i8 and i16 values with a 32-bit inline-PTX
  subtraction, so no 16-bit negation reaches the driver.

### Performance changes users may notice

- A kernel taking a `@tack.device_class` object no longer compiles anew
  for every length of a field the object holds. The object's fields were
  keyed by their full shape; they are now keyed by rank, and the
  dimensions the code actually bakes in (a row stride, a length used in
  arithmetic) still specialize, as they do for plain field arguments. A
  contour of a 64^3 grid on Metal at an output size not seen before went
  from 36 to 9 ms. The bucket sort's key template recompiled for every
  size of contour, slice and clip; the dataset views likewise hold
  fields of every mesh's size.

- Metal no longer waits about 0.3 ms after every launch: a contour of 100k
  cells, some 76 launches, spent 26 ms waiting. It also lets Metal clear
  new buffers instead of clearing them on the CPU, which was 252 of
  399 ms when deriving the faces of 5M hexahedra.
- `tack.zeros` on Metal takes the new buffer as Metal cleared it. It used
  to fill it on the CPU, and to wait for every queued launch first, a
  round trip of about 0.26 ms, though no launch can use a buffer that
  new. Contour and threshold of hexahedra on an M1 Max: 10k cells
  4.5 -> 4.1 ms and 3.6 -> 3.0 ms, 1M cells 7.7 -> 6.1 ms and 6.9 -> 5.4 ms.
- Scans reduce, then scan, by chunks instead of by Blelloch's tree: a few
  launches per factor of 256 rather than two per factor of two. On GPUs a
  chunk is a 2048-element tile scanned by a workgroup. Exclusive scan of
  1M i32: CPU on 8 threads 3.95 -> 0.86 ms, Metal 12.5 -> 1.1 ms.
- The radix sort sorts 4096-key tiles in shared memory on GPUs, and finds
  its key range without contended atomics. 10M u32 keys on Metal:
  153 -> 39 ms. 6M u64 keys on 8 CPU threads: 838 -> 89 ms; eight
  threads had been four times slower than one.
- `Field.fill`, and so `tack.zeros`, `ones` and `full`, stays on the
  device on CUDA, HIP and Level Zero instead of uploading a host array.
  In a 5M-cell contour on CUDA, filling had taken 24 of 40 ms.
- A `@tack.device_class` argument's class constants are found once per
  class instead of on every launch.

- A kernel's parallel loop over a two- or three-dimensional
  `tack.ndrange` no longer divides to recover its indices. GPUs launch a
  2D or 3D grid of the loop's shape, and CPU workers walk their chunk row
  by row. On an M1 Max a 200³ stencil went from 4.7 to 0.9 ms on Metal and
  from 6.5 to 1.7 ms on CPU, and a 4000×2000 elementwise kernel from 3.3
  to 0.4 ms on Metal. Results are unchanged. Kernels using workgroup
  primitives, and `ndrange` over four or more dimensions, keep the old
  lowering; so does a GPU launch with an extent past the device's grid
  limits, decided per dispatch with no recompile.

### Fixes with no source change needed

- Metal buffers, textures, libraries and pipelines are freed.
  pyobjc-framework-Metal 12.1 describes Metal's `new...` methods without
  their ownership, so nothing Tack created was ever released.
  Twenty dropped 64 MB fields had left 1.25 GB resident; they now leave
  32 MB.
- A zero-element field on HIP can be passed to a kernel. hip-python gave
  it no device pointer, and the launch failed.

- A vector or matrix `tack.constant` with a type, or of integers, read at
  a runtime index (`FACES[4 * f + k]`) is one load from a constant array
  instead of a chain of conditional expressions, a comparison per entry
  at every lookup. On Metal a loop that stores to a field and does many
  such lookups crashed the compiler service at pipeline creation (M1
  Max, macOS 26): "Compilation failed due to an interrupted connection".
  The result is unchanged, including an index outside the table giving
  the last value. Untyped float constants keep the chain, since their
  components take the precision of what they meet.
- An `ndrange` whose sizes are both negative (`ndrange(-3, -4)`) runs no
  iterations. Their product is positive, and the loop ran 12 iterations
  with negative indices.
- On CUDA with NVRTC 12.0, compiling a kernel no longer changes the
  calling thread's locale. NVRTC 12.0 left the thread in the C locale, so
  from the first compile on Python's default text encoding was ASCII, and
  `open`, `read_text` or `write_text` without an encoding failed on any
  non-ASCII character. The compile now restores the thread's locale.

### Known limitations

- **Dataset algorithms scale weakly with CPU threads on x86 Linux.** One
  thread is faster than VTK on one, but the multi-pass algorithms
  (contour, slice, threshold), the point and cell data conversions and
  external faces gain little from more threads. From 1 to 8 threads at 1M
  cells the median speedup was 1.05× on a Xeon E5-2650 and 1.5× on an
  EPYC 7453, where VTK got 2.4× and 3.3×; on the EPYC, 28 threads were
  slower than 8. On an M1 Max contour, slice and threshold scale 1.6–1.8×.
  On CUDA they were faster than VTK on all 28 of the EPYC's cores in every
  case at 1M cells.
- **ROCm 7.0 and 7.1 miscompile some integer code on HIP.** Their device
  compiler (AMD clang 20, inside hipRTC) can evaluate a 64-bit signed
  comparison wrongly when it sits inside a long integer expression; the
  kernel returns a wrong value with no error. Tack's generated source is
  correct, and right on CUDA, as host C++ and under ROCm's clang 23. Use
  a newer ROCm where possible; on 7.0 and 7.1, check integer-heavy kernels
  against the CPU backend.
- **CUDA with NVRTC older than 13 can compute wrong results** (see
  *Changed*); Tack warns at the first compile.
- **HIP on gfx90a (MI200) samples 3D textures in software.** The device
  reports image support but refuses linear 3D filtering, which is all
  `Texture3D` offers; Tack detects this at init and falls back. MI300
  parts have no texture hardware and always sample in software.
- **Metal compiles a kernel whose sequential loop stores to a field in a
  function of its own**, to avoid an Apple GPU compiler fault (0.3.0's
  *Fixes*). The fault's mechanism is not known, so the rule is wider than
  the trigger; it costs under 3% on the kernels that take it.
- **Not in this release:** dataset fields of order above 2, struct types
  (use one field per member), a 3×3 SVD (2×2 only, in the examples),
  swizzles such as `v.xy` or `v.yzw` (write `tack.Vector([v.y, v.z,
  v.w])`), and automatic differentiation.
- **Windows has not been revalidated.**
- **HIP leaks one hipRTC program object per compiled variant.**
  `hiprtcDestroyProgram` segfaults in hip-python (still in 7.2.2), so Tack
  does not call it; the variant cache keeps compilations rare.
- **Level Zero relies on two workarounds for Intel driver defects:** f64
  `floor`/`ceil` restore the sign of zero with `copysign`, and narrow signed
  negation and `abs` are emitted as non-inlined helpers.
- **Raster depth bias:** in mixed scenes, path-traced surfaces are pushed
  back by a fixed 0.5% relative depth bias so a wireframe lying on its own
  surface is drawn; it can drop wire pixels at grazing angles.

## 0.3.0 — 2026-10-07

The headline: the kernel language grew the vocabulary that simulation and
graphics code reaches for, found by porting 50 Taichi, Warp and
awesome-taichi programs and checking each against a NumPy reference on
every backend. Vectors work wherever scalars do, small matrices and
masks are values, kernels can be methods and return a value, atomics
return the old value, and constants, `tack.math` and `tack.random` cover
what the ports carried themselves. The same work found and fixed wrong
results that 0.2.0 shipped with, most importantly a Metal compiler fault
that dropped field updates inside loops, and a vector assignment that
read its own partly overwritten target on every backend; see *Kernels
that now compute different results* and *Fixes*.

Validated on Linux (CPU, CUDA on an RTX 4060 Ti, HIP on an MI210 and an
MI300X, Level Zero on an Intel Max 1100) and macOS (CPU and Metal on an
M1 Max), together with the 57 example checks in tack-examples.

### Added

#### Vectors and matrices

- Vectors in the places a scalar works. The math builtins (`min`, `max`,
  `abs`, `floor`, `sqrt`, `pow`, ...) and the casts (`int`, `float`,
  `tack.f32`, ...) apply to each component, with a scalar argument
  repeated: `min(max(v, -1.0), 1.0)`, `int(floor(p))`. A conditional
  expression selects whole vectors on one condition. `x`, `y`, `z` and
  `w` name the first four components, to read (`v.x`, `vf[i].y`) and to
  assign (`v.x = a`, `v.y += a`). A component of a vector field element
  can be stored directly: `vf[i][c] = a`, `vf[i][c] += a`, `vf[i].y = a`.
  `vf[i] += vec` and `vf[i] *= scalar` update an element in place. A
  vector indexes a field one dimension per component: `grid[cell]` is
  `grid[cell[0], cell[1]]`. All of these used to fail IR verification
  with "expected expr node", or with an `AttributeError` for the
  augmented store. Vectors of different widths in one operation, a
  component name past the width, and a vector combined into a scalar
  target are rejected at lowering with the source position.
- `tack.Matrix`: small fixed-size matrices, up to 4×4, scalarized like
  vectors. `tack.Matrix.field(n, m, dtype, shape)` allocates a field of
  them. In kernels, `tack.Matrix([[a, b], [c, d]])`, rows given as
  vectors, and `tack.Matrix.identity(n)` build one; `@` multiplies
  matrices and vectors, with `+`, `-`, `*`, `/` entry by entry;
  `transpose()`, `trace()`, and for 2×2 and 3×3 `determinant()` and
  `inverse()`; `A[i, j]` reads and writes entries; `u.outer_product(v)`
  makes one from two vectors. Matrices pass through device functions,
  tuple assignment, conditional expressions, field stores and atomics as
  vectors do. See *Matrices* in the User's Guide.
- A list of scalars in a kernel or device function is a vector:
  `pos[i] = [x, y]` is `pos[i] = tack.Vector([x, y])`, wherever a vector
  is accepted. It was rejected as "unsupported List". An empty list, a
  list of lists (write `tack.Matrix`), a list as an assignment target and
  a loop over a list are still rejected.
- Vector components by runtime index: `vec[k]` reads and `vec[k] = x` /
  `vec[k] += x` write the component a loop variable or field value
  selects, through a chain of selects; past the last component a read
  gives the last component and a write does nothing. `vf[i][c]` and
  `f(x)[c]` index a vector-valued expression without naming it first.
  These used to fail IR verification with "expected expr node". A literal
  index out of range, and a subscript of a scalar value, are rejected at
  lowering with the source position.
- Elementwise comparison: `v > 0.0`, `lo <= v <= hi`, `a == b` on vectors
  and matrices give a mask, a vector of `0`/`1` per component; `and`,
  `or` and `not` act per component on masks; `any(mask)` and `all(mask)`
  reduce one; `tack.select(mask, a, b)` picks per component, with scalar
  masks and scalar arms broadcast. A mask as an `if` or `while` condition
  is rejected with the hint to reduce it.
- `min` and `max` take two or more values, as in Python
  (`min(a, b, c)`), and vectors have the reductions `v.sum()`, `v.min()`
  and `v.max()` over their components. `v.normalized(eps)` divides by
  `norm() + eps`.
- `v.norm(eps)` is `sqrt(v.norm_sqr() + eps)`, for a length that must not
  be zero. It was "norm() takes no arguments".
- A device function can return vectors among several values:
  `return distance, normal, color`, unpacked with
  `d, n, c = closest_hit(...)`. Only scalars could be returned together;
  a vector slot was never bound and lowering failed on its name.
- Tuple assignment to field elements and vector components:
  `x[i], v[i] = p, q` (whole vectors included), a swap such as
  `a[i], b[i] = b[i], a[i]`, and `lo[i], hi[i] = f(...)` for a device
  function that returns two values. The whole right side is evaluated
  first and the targets are then assigned from left to right, so
  `a[i], i = x, j` stores at the old `i`. Only plain names were accepted
  as targets.
- Vector fields exchange data with NumPy one row per vector:
  `from_numpy` accepts `(*shape, n)` as well as the flat storage shape,
  and `to_numpy(vectors=True)` returns `(*shape, n)`. Plain `to_numpy()`
  still returns the flat array.
- `fill` takes one element's value for a vector or matrix field:
  `colors.fill([1.0, 1.0, 1.0])`. A scalar still sets every component.

#### Constants, math and random numbers

- `tack.constant(value, dtype=None)`: a named constant that kernels and
  device functions may read from the scope that defines them.
  `DT = tack.constant(0.01)` at module level lets a kernel write `DT`; it
  reads as the literal, and stays an ordinary Python number for host
  code. Kernels still capture nothing else: a plain module-level value
  raises `NameError` as before, and the message now names
  `tack.constant`. With a dtype the constant is typed, so
  `tack.constant(747796405, tack.u32)` multiplies in wrapping u32
  arithmetic (a plain literal there promotes to i64) and
  `tack.constant(0.1, tack.f64)` is the exact double. `math.pi`, `math.e`
  and `math.tau` can be written in kernels.
- Vector and matrix constants: `tack.constant((0.5, 0.5, 0.0))` is the
  vector with those components in a kernel, `tack.constant(((1, 0), (0, 1)))`
  the matrix with those rows, typed per component as scalar constants
  are. On the host they index, iterate and convert to NumPy arrays.
- `sinh`, `cosh` and `tanh` as math builtins, on every backend.
- `tack.math`: `fract`, `mix`, `clamp`, `saturate`, `smoothstep`, `step`,
  `sign`, `length`, `distance` and `normalize` as device functions with
  GLSL's definitions, on scalars and vectors alike.
- `tack.random`: random numbers in kernels from a counter-based generator
  with explicit state. `seed(index, stream)`, then `u, state =
  uniform(state)`, `normal`, `direction2`, `direction3`; NumPy mirrors
  (`np_uniform`, ...) draw exactly the same values, and `uniform` and the
  states are bit-identical on every backend. The example ports' `rng`
  module, adopted.

#### Kernels, loops and atomics

- A kernel may end in `return expr`, with a return annotation naming the
  type (`-> tack.f32`, `-> float`, `-> int`); the call returns that value
  as a Python number. It is computed after the launch by a one-thread
  epilogue kernel that stores it into a hidden field, so it may read
  fields, arguments, constants and locals bound outside the loop, not a
  local the loop assigns. A return anywhere else is still rejected.
- An atomic is an expression as well as a statement: `slot =
  tack.atomic_add(counter, 0, 1)` is the element's value just before the
  update, on every backend and for every supported type, which is how
  threads claim unique slots and append to a shared list. On a vector
  field with a vector value it is the vector of the components' old
  values. Barriers remain statement-only.
- Atomics take an index per dimension: `tack.atomic_add(grid, (i, j), v)`,
  where a vector supplies one index per component
  (`tack.atomic_add(grid, cell, v)`). On a vector field a vector value
  updates every component of the element. Multi-dimensional and vector
  targets previously needed hand-linearized indices, and a tuple index
  failed IR verification.
- `tack.ndrange` takes `(start, end)` pairs as well as sizes:
  `for i, j in tack.ndrange((1, n - 1), (1, m - 1))` visits the interior
  of a grid. An empty or reversed pair runs no iterations.
- One element of a field can be read from host code: `f[i]`, `f[i, j]`,
  `f[None]`, giving a Python number or, for a vector or matrix field, a
  NumPy array. Writing an element, slicing and iterating are refused with
  a message pointing at `from_numpy`, `fill` and `to_numpy`. On CPU and
  Metal the read is a slice of unified memory; on CUDA, HIP and Level
  Zero it is a copy of just the element.

#### Data-oriented classes

- A kernel can be a method of a `@tack.data_oriented` class:
  `@tack.kernel def step(self, dt)` is called as `grid.step(0.1)`, and the
  object is a template like any other argument. It failed with "expects 1
  arguments, got 0". `tack.inspect` takes the bound form too.
- A `@tack.func` under `@staticmethod` in a data-oriented class is called
  through `self` like the other methods. It was not found.
- A `@tack.data_oriented` object can hold a `@tack.func` as an instance
  attribute and call it from its methods or from a kernel
  (`self.smoothing = cubic`, then `self.smoothing(r, h)`). Which function
  the attribute holds is part of the kernel's specialization.

#### Algorithms

- `tack.algorithms.argsort`, `sort_by_key`, `gather`, `unique` and
  `reduce_by_key`: a stable radix sort for `i32`/`u32`/`i64`/`u64` keys
  and segmented reductions over runs of equal keys, built from ordinary
  kernels and the scan so they run on every backend. See *Sorting and
  segmented reductions* in the User's Guide.
- The `tack.algorithms` statistics (`var`, `std`, `norm`, `absmax`,
  `count_nonzero`, `dot`, `histogram`) accept vector fields and reduce
  over all components in storage order; they used to fail IR
  verification.

### Kernels that now compute different results

- A vector assignment whose right side reads its own target now reads
  the old components. Vectors are assigned one component at a time, and
  the right side was evaluated in between, so later components saw
  earlier ones already overwritten: `v = v.normalized()` (the result
  was not a unit vector), `v = v.cross(w)`,
  `v = tack.Vector([v[1], v[2], v[0]])`, `v += v.cross(w)` and
  `a[i] = a[i].cross(b[i])` all computed wrong vectors, on every backend
  and without an error. Components that could observe an earlier
  component's assignment are now evaluated first, as Python does. For a
  store to a field element that means any component that loads from a
  field, since fields may share storage. Assignments whose components
  only read their own component (`v = v * 2.0`, `a[i] = a[i] + b[i]`)
  computed the right result before and still do.

- A scalar stored to a vector field element sets every component:
  `vf[i] = 0.0` names element `i`, as a load of `vf[i]` does, and as
  `vf[i] *= 2.0` does. It wrote the single component at flat index `i`.
- A vector field's element index is computed in 64 bits. It was narrowed
  to i32 before being scaled by the vector width, so a field of more than
  2^31 components was indexed wrongly.

### Code that is now rejected

- A multi-dimensional field indexed with the wrong number of indices.
  `grid[i, j]` on a three-dimensional field linearized with the sizes it
  had and silently addressed another element; so did a vector index of
  the wrong width and an atomic's index. The dispatch that binds such a
  field now raises `TypeError` with the source position. A single index
  is still a flat, row-major index.
- A vector or tuple anywhere one value is required, such as a comparison
  (`vf[i] < 1.0`), a condition, a loop bound or `print`. Operations that
  do not map over a vector's components used to fail IR verification
  with "expected expr node" and a path into the lowered tree; every such
  statement now raises `UnsupportedSyntaxError` naming the kernel or
  device function and the source position.
- A vector stored where it does not fit: `s[i] = vec` into a field of
  scalars or a local array, and `vf[i] = vec` into a field of vectors of
  another width. Both wrote components at offsets computed from the
  value's width, over whatever was there.
- Reading a `for` loop's variable after its loop, before the name is
  assigned again, raises `NameError` at lowering with the read's position,
  on every backend. CPU raised a `NameError` from codegen and the GPU
  backends failed to compile when the name had no other binding; when it
  did (`d = 100` before `for d in range(4)`), every backend silently read
  the outer value where Python gives the last value iterated. Copy the
  value to another name inside the loop. Sibling loops may still share a
  variable, and a loop variable may still be assigned as an ordinary local
  afterwards.

### Fixes with no source change needed

- The size of a local or shared array may be arithmetic on constants and
  dimensions: `tack.local_array(tack.f32, MAX_HITS * 4)` is folded to one
  constant when dimensions are resolved. Only a literal was, and Metal
  refused the kernel with "array size is not a constant expression".
- Reading a zero-dimensional field to the host (`f.to_numpy()` for
  `shape=()`) crashed the process on CUDA: cuda-python does not take a
  zero-dimensional NumPy array as a host buffer. The copy now goes through
  a flat view.
- A parallel loop may start at a negated argument: `range(-n, n + 1)` and
  `tack.ndrange((-n, n + 1), ...)` failed with "Cannot resolve loop range
  expression: IRUnaryOp", because the host evaluates the launch size and
  had no case for a unary minus.
- Unary plus on an integer (`+n`) compiles on Metal, CUDA, HIP and Level
  Zero. The generators emitted a one-argument call to the two-argument
  add helper, which the device compilers rejected.
- **Metal computed wrong results, silently, for a field element updated
  in a loop.** `for k in range(n): total[0] += x[k]; counter[0] += 1` left
  `total` at `start + x[n - 1]`: Apple's compiler read `total[0]` once,
  before the loop. It happened when the element's address did not depend
  on the thread, the loop also stored to another field of the same type,
  and the trip count was a runtime value; `while` loops and f32 fields
  were affected alike. Present since field pointers moved into one
  argument buffer, so in 0.2.0. A loop that stores to a field is now
  compiled in a function of its own, which costs under 3% for the kernels
  that have one and nothing for the others.
- A zero-dimensional scalar field, `tack.field(dtype, shape=())`, can be
  allocated on Metal. Its buffer zeroed the new storage with `[:]`, which
  a zero-dimensional array rejects with "too many indices". CPU was not
  affected, nor zero-dimensional vector fields.
- A subclass of a `@tack.data_oriented` class inherits its bases'
  `@tack.func` methods and class constants and may override them. Only
  the class's own were found, so a method calling an inherited one failed
  with "self.weight ... is neither a class constant ... nor a @tack.func
  method". A subclass without the decorator is handled the same way.
- CPU threading on two-thread machines: one worker the scheduler took
  off its core could make a cheap kernel look expensive. The fan-out's
  per-worker median picked the slower of two workers, so a single stalled
  worker set a floor under the serial cost estimate and raised the
  kernel's threading threshold with it, until a later recheck corrected
  it. The median over an even number of workers is now the lower middle
  value. Machines with three or more threads were not affected.
- A field of zero elements can be allocated on Metal and CUDA, as it
  already could on CPU and Level Zero. A filter that selects nothing now
  returns an empty field instead of failing to allocate it.
- A kernel that used a name as a sequential loop's variable and later
  assigned it as a plain local (`for d in range(4): ...` then `d = ...`)
  failed to compile on Metal, CUDA, HIP and Level Zero with an undeclared
  identifier; CPU accepted it. The C-family generators now treat the loop
  header's declaration as ending with its block, so the later assignment
  declares a new local, as on CPU.
- A field can be passed down through any number of nested device
  functions. Three or more levels failed to compile on every backend
  ("Cannot coerce float* to i32" on CPU, a pointer-to-integer error from
  the GPU compilers): copy propagation resolved one link of the chain of
  parameter copies per pass and left a local holding the field.
- A `@tack.data_oriented` template's vector field attributes are now
  detected as vector fields: `self.vel[i, j]` in a template method, or
  `obj.vel[i, j]` in the kernel, lowered as a scalar field access and
  failed IR verification.
- A `@tack.func` given a vector field now loads whole vectors from it:
  `return vf[i]` inside the function lowered to a single scalar load, so
  a kernel doing `out[i] = f(vf, i)` wrote one component and left the
  rest. The inliner propagated vector-variable and texture metadata to a
  function's parameters but not vector-field metadata.
- A device function's locals bound by tuple unpacking (`t, u = f()`, or
  `for i, j in ...`) are now renamed when the function is inlined. They
  escaped renaming before, so the inlined body overwrote the caller's
  variables of the same names.

### Tooling

- `test_a_stale_parallel_rate_gets_re_measured` failed about one run in
  thirteen on two-thread machines and once on a CI runner: it asserted
  that two threads beat one on a bandwidth-bound kernel, which they need
  not. The scenario now uses a compute-bound kernel, so the assertion
  tests the recovery path it was written for.
- The CPU threading tests no longer fail on two-thread machines: the
  per-worker median above was the cause, and a test that compared against
  the estimate before its own fan-out now compares against the one after.

### Known limitations

- **ROCm 7.0 and 7.1 miscompile some integer code on HIP.** Their device
  compiler (AMD clang 20, inside hipRTC) can evaluate a 64-bit signed
  comparison wrongly when it sits inside a long integer expression; the
  kernel returns a wrong value with no error. Tack's generated source is
  correct, and right on CUDA, as host C++ and under ROCm's clang 23. Use
  a newer ROCm where possible; on 7.0 and 7.1, check integer-heavy kernels
  against the CPU backend.
- **HIP on gfx90a (MI200) samples 3D textures in software.** The device
  reports image support but refuses linear 3D filtering, which is all
  `Texture3D` offers; Tack detects this at init and falls back. MI300
  parts have no texture hardware and always sample in software.
- **Metal compiles a kernel whose sequential loop stores to a field in a
  function of its own**, to avoid an Apple GPU compiler fault (see
  *Fixes*). The fault's mechanism is not known, so the rule is wider than
  the trigger; it costs under 3% on the kernels that take it.
- **Not in this release:** struct types (use one field per member), a
  3×3 SVD (2×2 only, in the examples), swizzles such as `v.xy` or `v.yzw`
  (write `tack.Vector([v.y, v.z, v.w])`), and automatic differentiation.
- **Windows has not been revalidated.**
- **HIP leaks one hipRTC program object per compiled variant.**
  `hiprtcDestroyProgram` segfaults in hip-python, so Tack does not call it;
  the variant cache keeps compilations rare.
- **Level Zero relies on two workarounds for Intel driver defects:** f64
  `floor`/`ceil` restore the sign of zero with `copysign`, and narrow signed
  negation and `abs` are emitted as non-inlined helpers.
- **Raster depth bias:** in mixed scenes, path-traced surfaces are pushed
  back by a fixed 0.5% relative depth bias so a wireframe lying on its own
  surface is drawn; it can drop wire pixels at grazing angles.

## 0.2.0 — 2026-10-05

The headline: Tack now has a written language contract, and the compiler
enforces it on every backend. Most of the changes below are cases where a
kernel used to compute a **different answer on different backends**, or
silently did something other than what its source said. Code that relied
on one backend's old behavior can change results or start raising.

### Kernels that now compute different results

Each was a correctness fix: the old result disagreed with Python, with the
contract, or with another backend.

| Change | Before | Now | Contract section |
|---|---|---|---|
| Integer `/` | Truncated on CUDA/HIP/Metal/OpenCL; on CPU narrowed back to the integer type | True division, f32 result: `7 / 2 == 3.5`. Use `//` for floor division, or `int(a / b)` within range for truncation | *Required: true division* |
| Integer `//` and `%` with negative operands | Truncated toward zero on GPUs | Python floor semantics: `-7 // 3 == -3`, `-7 % 3 == 2`, `7 % -3 == -2` | *Required: integer floor division and remainder* |
| Integer `**` and `pow` | Went through floating `pow`; large results lost precision; `pow` and `**` could disagree | Exact power modulo 2^N at the **base's** type: i8 `3 ** 5 == -13`. Cast the base to float for floating power | *Required: integer power* |
| Floating `//` and `%` | CPU `%` took the dividend's sign; GPU `//` narrowed through an integer and overflowed | Python semantics in the promoted float type, correct near rounded boundaries; `//` returns an integer-valued float | *Required: floating floor division and remainder* |
| Fixed-width integer overflow | Varied by backend; GPU code relied on C promotion and signed-overflow undefined behavior | Every `+`, `-`, `*`, unary `-`, `~`, `<<`, `&`, `\|` and `^` wraps at its annotated width before the next operation uses it: i8 `127 + 1 == -128` | *Required: fixed-width arithmetic and integer conversion* |
| Mixed-signedness promotion | Inconsistent; unsigned high bits could compare as negative | Narrowest type holding both ranges: i8 with u16 → i32, i32 with u32 → i64 | same |
| Integer casts and stores | Widening followed the destination's signedness on CPU | Reduce mod 2^N, widen by the **source** signedness: i8 `-1` → u64 is 2^64 − 1 | same |
| Right shift of unsigned values, unsigned comparisons, `min`/`max` | Signed on CPU in places; integer `min`/`max` could go through floating point | Logical right shift for unsigned; exact integer comparison | same |
| CUDA and Metal floating math | `--use_fast_math` on CUDA and fast math on Metal for every kernel | Safe math everywhere: NaN, infinity and signed zero preserved; `(a+b)+c` keeps its grouping; fused multiply-add still allowed. CUDA/Metal images move closer to the CPU reference | *Floating-point execution policy* |
| `sqrt`, trig, `exp`, `log` on integer or mixed arguments | Argument precision varied; CPU libm calls could fail to compile | Arguments convert to the annotated result precision first: f32 unless an argument is f64 | same |
| Comparisons and `and`/`or` | CPU could yield −1 for true | Normalized i32 `0`/`1`; `and`/`or` return Booleans, not an operand | *Types and numerical behavior* |
| Field `min()`/`max()` | Device paths mishandled values beyond ±1e38, same-sign infinities and NaNs | Full normal range, infinities, NaN propagation, order-independent zero ties; empty `min`/`max` raise `ValueError`, empty `sum` is +0, empty `mean` is NaN | *Field and parallel reductions* |
| Top-level `range(start, end)` | Ignored `start` on some paths; empty ranges could run | Runs exactly `[start, end)`; nothing when empty or reversed | *Execution and ordering* |
| Evaluation order | Not defined | Left to right; `and`/`or` short-circuit; conditional expressions evaluate one arm; augmented assignment evaluates its index once | same |
| Overlapping field arguments | Could read stale values on Metal and through `restrict` on other backends | Program order preserved when fields alias, including the same field twice and overlapping views | *Memory and aliasing* |
| Raster points and wireframes | Coincident primitives could produce mixed colour channels | Depth winner and its colour selected together, deterministically: lowest primitive index at equal depth | Rendering |
| Mixed solid and raster scenes | Wireframe and point actors were path traced as surfaces; volume plus wireframe dropped the volume | Raster actors composite over the path-traced or ray-cast image with depth | Rendering |
| Float literals in f64 code | Always f32: `x_f64 * 0.1` multiplied by 0.10000000149…, and `tack.f64(0.1)` widened that rounded value | A literal takes the precision of the operand it meets (NumPy NEP 50 "weak" scalars), so f64 expressions use the exact double and `tack.f64(0.1)` is exact. Comparisons with a literal change for values between its f32 and f64 roundings. f32 kernels are unchanged. Edge cases: a local assigned only literals (`a = 0.1`) is still f32, so write `tack.f64(0.1)`; `i * 0.1` is f32; write `1.0 / 3.0`, not `1 / 3` | *Required: float literals are weakly typed* |
| Textures after their field changes | CPU sampled the field live; GPU hardware paths kept the data from first use (and could serve a stale texture at a reused address) | Every backend samples a copy taken by `tack.texture3d()`; call `tex.update()` after changing the field. `render_volume()` refreshes its volume each call | *Memory and aliasing* → Textures |
| Locals set before the parallel loop and reassigned inside it | On CPU the value carried from one iteration to the next within a worker's chunk; GPU threads each started fresh | Every iteration starts from the value set before the loop, on every backend | *Execution and ordering* |
| CUDA/HIP/Level Zero launches of 2^32 or more iterations | The thread index wrapped silently (on Level Zero because Intel's `get_global_id` wraps at 2^32); reductions over 2^32 elements on Level Zero summed wrongly | Every iteration runs; launches beyond a backend's grid limit raise `ValueError` | *Execution and ordering* |
| Level Zero: wrapped negation and `abs` of the signed minimum, widened | Intel's IGC 2.7.11 produced `-(-32768)` as 32768 in an i32 result, and `abs(INT_MIN)` as 2^31 in i64 | The contract's wrapped value, through a workaround in the generated code | *Required: fixed-width arithmetic and integer conversion* |

### Code that is now rejected

These raise a diagnostic naming the kernel or device function and its
source line, instead of being silently dropped or miscompiled.

- `assert`, `try`, `with`, generators, imports, nested definitions,
  comprehensions, annotated assignments, keyword or starred arguments.
- `break` out of the top-level parallel loop, and kernel `return`.
- Reading a name the kernel never binds (`NameError`); capturing a
  numeric value from the enclosing Python scope.
- Binding a second name to a local or shared array (`view = tmp`).
- A negative integer literal exponent; a literal zero or negative `range`
  step.
- Mixing any signed integer with `u64` without an explicit cast,
  including in comparisons, conditional arms and joined assignments.
- Calling an ordinary Python function, or a runtime function value, from a
  kernel; recursive device-function calls. Device calls now resolve through
  the defining module, so importing another module with a same-named
  function can no longer change a kernel.
- **On CPU:** kernels using `shared`, `shared_like`, `barrier`,
  `thread_id`, `block_sum`, `block_min` or `block_max`. CPU does not
  emulate workgroups; use `local_array` / `local_array_like` for private
  scratch. `examples/09_shared_memory.py` now requires a GPU `--arch`.
- **On GPUs:** collectives in a launch whose iteration count is not a
  multiple of 256, and barriers or block reductions inside branches the
  compiler cannot prove uniform.
- Atomics outside the declared domain (for example 64-bit atomics on Metal
  or Level Zero, 8/16-bit atomics on GPUs), atomics on private or shared
  arrays, and unaligned imported atomic targets.
- Block reductions on non-f32 arguments without an explicit `tack.f32(...)`.
- Field stores, atomics, barriers, block reductions and `print` outside
  the parallel loop, before or after it, including through inlined device
  functions. They used to run a backend-dependent number of times (once
  per CPU chunk or probe, once per GPU thread). Plain local assignments
  stay allowed. A kernel with no parallel loop, two of them, or one inside
  a branch now gets a source diagnostic too.
- `tack.texture3d()` on an f64 field, with `interp` other than `'linear'`
  (`'nearest'` was accepted but silently linear), or with a shape whose
  W·H·D differs from the field's size. A `Volume` built from an f64 field
  now fails at construction.
- Kernels that may store to a read-only field: `field_from_ptr` without
  `writable=True`, or a DLPack import flagged read-only.
- DLPack import of a Metal (`kDLMetal`) tensor, which never worked, now
  fails up front with a clear message.
- `tack.inspect(..., mode="optimized")` on a non-CPU backend, which used to
  return the source text. Inspection also applies dispatch's dtype check.
- An unrecognized `TACK_CPU_POLICY` value (it silently selected v1).

### Fixes with no source change needed

- CUDA: a shared context can outlive the backend that created it, and a
  field whose context is gone reports it instead of faulting.
- CPU: floating `atomic_min`/`atomic_max` are now real atomics (they were a
  load-compare-store race).
- CPU: field loads and stores no longer claim four-byte alignment, which
  was wrong for byte fields and unaligned imported buffers.
- Level Zero: integer `min`/`max`, float ternaries and nested local arrays
  compile on the real device compiler; signed zero survives f64
  `floor`/`ceil` despite an Intel driver defect.
- Python-legal names that are keywords in CUDA, OpenCL or MSL (`default`,
  `half`, `kernel`, …) now work as parameter and local names.
- NumPy 1.x works again with the CPU backend.
- `tack.algorithms` statistics and scans: `var` and `std` with an explicit
  `n` smaller than the field used the whole field's mean (`var([1, -2, 3,
  -4], n=2)` was 2.5, not 2.25), and `histogram` without a range used the
  whole field's range; both now cover the first `n` elements. Every
  statistics and scan function now raises `ValueError` for an `n` outside
  a field instead of reading, or for the scans writing, past it. A scan of
  zero elements returns 0 instead of reading index −1, and `var`/`std` of
  zero elements return NaN.
- `exclusive_scan` scanned in `i32` whatever the field dtypes, truncating
  floats and wrapping 64-bit integers, and both scans returned their total
  through `i32`. Both now scan in the output field's dtype and return the
  total in it, as a Python `int` or `float`.
- A process that exits while another library (VTK's DLPack support, for
  example) still holds an unconsumed Tack DLPack capsule no longer crashes
  at interpreter shutdown.
- Compiled-variant caching: vector width and alias relationships are part
  of the key; template classes release their cached code when collected.
- GPU backends: dispatching one kernel from several threads at once is
  serialized per compiled variant (per backend on Level Zero). It used to
  share the scalar argument buffer, and on CUDA 603 of 1200 concurrent
  dispatches got another thread's scalars. CUDA dispatch from a thread
  other than the one that called `tack.init` still needs Tack's context
  made current.
- `field.sum()`/`min()`/`max()` over 2^32 or more f32 elements work on
  CUDA, HIP and Level Zero, and use NumPy on Metal, instead of failing with
  `struct.error`.
- `from_numpy` on a reshaped view works on CPU and Metal; `field_from_ptr`
  checks the memory space of NumPy-integer and `CUdeviceptr` pointers, and
  on HIP a field wrapped from a NumPy-integer address now works (its copies
  failed with `hipErrorInvalidValue`).
- `tack.init(arch=tack.cpu, num_threads=N)` is accepted; `TACK_NO_REINIT=0`
  (and `false`, `no`, `off`) now means off; kernel errors name the kernel
  once; unreadable device-function source raises `RuntimeError` like a
  kernel's; textures no longer leak a GPU texture object per compiled
  kernel.

### Performance changes users may notice

Measured on the project's test machines.

| Change | Effect | Why |
|---|---|---|
| CPU threading decisions | Large CPU dispatches no longer stall on whole-frame serial rechecks; the CPU path tracer and cheap-kernel dispatches improved on the measured hosts | Threading-policy repairs |
| CPU disjoint-field specialization | Up to −70% on store-accumulator and −22% on stencil kernels on an older Xeon; ~3.5 µs extra per CPU dispatch | Proven non-overlapping fields compile with `noalias` |
| Cold compile | First call of a large kernel is slower (CUDA path tracer roughly +50% on the oldest test host) | IR verification at every pass boundary; partly offset by faster IR cloning |
| CUDA path tracer, warm | ~6–7% slower | Safe floating-point math |
| CPU path tracer, warm | ~4% slower on the oldest test host | Wrapped i32 index arithmetic; a renderer-side fix is planned |
| Textures | One extra full copy of the data per texture, on CPU as well | Textures copy their field |
| `render_volume()` | One device copy of the volume per call | Keeps the ray caster and the path tracer showing the same data |

### Tooling

- `docs/reference/language-contract.md`: the language contract.
- `validate_all.py --arch <backend>` checks every call and fails when the
  requested backend is unavailable.
- Host-side compiler checks honor `TACK_CLANG` and fail under
  `TACK_REQUIRE_CLANG=1` instead of skipping.
- `TACK_EXAMPLES_ARCH` selects the backend for the example sweep.
- New diagnostic probes in `benchmarks/`: threading decisions, LLVM
  path-tracer comparison, volume cold start, IR cloning, raster differential.

### Known limitations

- **ROCm 7.0.2 miscompiles some integer code on HIP.** Its device compiler
  (AMD clang 20, inside hipRTC) can evaluate a 64-bit signed comparison
  wrongly when it sits inside a long integer expression. The kernel then
  returns a wrong value with no error. Tack's generated source is correct;
  the same kernel is right on CUDA, as host C++ and under ROCm's clang 23.
  It depends on the surrounding expression, so Tack can't avoid it. Use a
  ROCm release newer than 7.0 where possible; on 7.0.2, check
  integer-heavy kernels against the CPU backend.
- **Windows has not been revalidated** for this release; the supported
  backends were validated on Linux (CPU, CUDA, HIP, Level Zero) and macOS
  (CPU, Metal on an M1 Max and an M3).
- **HIP leaks one hipRTC program object per compiled variant.**
  `hiprtcDestroyProgram` segfaults in hip-python, so Tack does not call it;
  the variant cache keeps compilations rare.
- **Level Zero relies on two workarounds for Intel driver defects:** f64
  `floor`/`ceil` restore the sign of zero with `copysign`, and narrow signed
  negation and `abs` are emitted as non-inlined helpers. Results follow the
  contract; both defects will be reported upstream.
- **Raster depth bias:** in mixed scenes, path-traced surfaces are pushed
  back by a fixed 0.5% relative depth bias (not slope-scaled) so a
  wireframe lying on its own surface is drawn. It can drop wire pixels at
  grazing angles or let very close geometry show through.

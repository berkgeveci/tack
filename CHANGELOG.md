# Changelog

All notable changes to Tack are recorded here. Rules cited by name live in
[`docs/reference/language-contract.md`](docs/reference/language-contract.md).

## Unreleased

### Added

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

### Changed

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

### Fixed

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

### Added

- Examples 50-53 in `packages/tack-vis/examples/`: VTK pipelines on real files
  from VTK's test data, each from a reader to a rendered picture (`--png`) and
  checked against VTK's filters. 50 is an Exodus file's collection
  (`disk_out_ref.ex2`, IOSS): an isosurface, streamlines as tubes, the surface.
  51 is an OpenFOAM case's multiblock (`squareBend`): gradients, vorticity, a
  slice, integrals over the inlet and outlet. 52 is an OpenFOAM time series (the
  cavity) through `Pathlines`, driving and in situ. 53 is an Enzo AMR file:
  statistics and the mass over the hierarchy, a slice of each level. The files
  are fetched once by their SHA-512 (`_vtk_data.py`, `vtk_data.sha512`) or read
  from `TACK_VTK_DATA`; each takes `--arch` for the device.
- The VTK guide lists every algorithm with VTK's filter that does the same work,
  and has a section each on pieces and trees, MPI ranks, GPUs, summaries, AMR
  hierarchies and time.

### Changed

- The temporal VTK algorithms (`RunningStatistics`, `TimeDerivative`,
  `Advection`, `Pathlines`, `Streaklines`) take a `vtkMultiBlockDataSet` or a
  `vtkPartitionedDataSetCollection`, which they refused, in both modes: each
  partitioned dataset and each block that is one dataset has a Tack filter of its
  own, and the output is a tree of the same shape, at the step's time (the
  tree's, or its datasets'). The OpenFOAM, IOSS and CGNS readers give trees.

### Fixed

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

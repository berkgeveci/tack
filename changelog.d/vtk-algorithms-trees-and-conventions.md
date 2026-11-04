### Added

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

### Fixed

- A VTK algorithm whose filter raised kept the exception with its traceback,
  whose frames held the algorithm and VTK's pipeline information in a cycle;
  Python's collector crashed freeing it. The exception is kept without its
  traceback, printed as VTK printed it.
- A filter over pieces (`GenerateGlobalIds`, `GhostCells`, `MergePieces`) on a
  multiblock's `vtkMultiPieceDataSet` block ran on each piece alone, and gave
  empty pieces.

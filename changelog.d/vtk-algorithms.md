### Added

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

### Changed

- `tack.interop.vtk` is a package (its names unchanged), with `algorithms`
  beside it.

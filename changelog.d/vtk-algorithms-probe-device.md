### Added

- `tack.interop.vtk.algorithms.Probe`, `td.probe` as a VTK algorithm in the
  manner of `vtkProbeFilter`: the source's fields (input port 1,
  `SetSourceConnection`/`SetSourceData`) at the input's points, on the input's
  structure, with `valid`.
- A `device` parameter on every Tack VTK algorithm: the filter runs there, its
  input moved there once (and kept, so the filters after it find it there).
  By default a filter runs where its input's memory is, which for a reader's
  output is the host.

### Fixed

- `dataset_to_vtk(..., copy=False)` of a dataset made by
  `vtk_to_dataset(..., copy=False)` from GPU arrays hands VTK back the arrays
  it shared, which DLPack cannot (they are read-only to Tack): a filter that
  keeps its input's points, as `Probe` does, failed.

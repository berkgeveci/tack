### Added

- `dataset_to_vtk(copy=False)` and `vtk_to_dataset(copy=False)` share Metal arrays
  with VTK, with any VTK: on Apple silicon a Metal buffer is host memory, so VTK
  holds views of the buffers themselves (after one wait), and with Metal the
  default device they come back as the same buffers. VTK's own memory is wrapped
  in place where it lies on whole pages and copied to Metal otherwise, so the
  dataset is on Metal throughout. Metal's `wrap_ptr` (and
  `tack.array_from_ptr(..., device="metal")`) takes a NumPy array of host memory
  for this.

### Fixed

- `vtk_to_dataset(..., copy=False)` makes a dataset from a host grid on the CPU
  whatever the default device: with a GPU the default (as
  `tack.interop.vtk.init_level_zero()` makes Level Zero), what it copied beside
  the shared arrays -- implicit cell types and offsets, converted values -- was
  made on the GPU, so filters gave empty outputs or refused arrays on two devices.

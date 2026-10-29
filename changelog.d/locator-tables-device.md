### Fixed

- `CellLocator.cell_bins()`, `leaves()` and `cell_neighbors()` build their
  tables on the dataset's device. Called directly on a dataset on a GPU (not
  through `find`, `probe` or advection, which set the device), they allocated
  on the default device and their first kernel refused arrays on two devices.

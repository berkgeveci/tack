### Fixed

- `vtk_to_dataset` reads a `vtkBitArray` as one u8 value (0 or 1) per point or
  cell; it came as its packed bytes, eight values to one, and conversion
  failed (`kitchen.vtk`'s `hpor`).

### Changed

- **Breaking:** an array is made only of a type its device supports:
  `tack.empty` and every constructor, copy and `to_device` on it,
  `array_from_ptr` and DLPack imports raise `TypeError` for an `f64` array on
  Metal (or on a Level Zero device without `f64`), where before the array was
  made and the first kernel given it raised.
- `vtk_to_dataset(dtype=None)` with a default device without `f64` (Metal)
  reads floating-point arrays as `f32`, which such a device's arrays take,
  where it kept VTK's doubles there before.

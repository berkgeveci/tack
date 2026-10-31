### Added

- `tack.set_default_float(tack.f64)` and `tack.get_default_float()`: the
  float type of arrays nothing else types (constructors without a `dtype`,
  Python floats, `vec`/`mat` types without a component type, implicit
  arrays) and of `tack.dataset`'s constructors, sources and readers. f32
  unless set; kernels' precision still follows their arrays.

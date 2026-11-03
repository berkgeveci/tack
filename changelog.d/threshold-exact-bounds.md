### Changed

- `threshold` and `threshold_points` keep a value exactly when it lies in
  `[lower, upper]` as given: on an f32 field the bounds are rounded inward to
  f32, since the nearest f32 of a bound may lie just outside it (f32(-0.1) is
  -0.10000000149), which kept values VTK, comparing in double, leaves out (one
  cell of the AhmedML mesh's 18,027,253). f64 and integer fields are unchanged.

### Changed

- A device without f64 (Metal, a Level Zero device without fp64) refuses a
  kernel that computes anything in f64 -- a `tack.f64` cast, an f64 constant,
  local or parameter, from the kernel or a function it inlines -- with a
  `TypeError` naming the cause, once its types are known and before code is
  generated (`refuse_f64`). Metal's code generator refused only f64 casts,
  with a `NotImplementedError`. A CPU posing as such a device, as the dataset
  device tests make one, refuses them too, so those tests find on the CPU what
  only Metal's compiler found before (subdivision's `tack.f64` casts, on the
  higher-order branch).

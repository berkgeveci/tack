### Fixed

- `probe` rounds an integer field evaluated in a cell to the nearest integer,
  halves away from zero, as VTK's probe does; it truncated (a `u8` field of 0
  and 1 probed three quarters of the way came out 0).

### Fixed

- Gradients in f32 on thin cells. A cell's Jacobian summed its corners'
  absolute positions, and its field's parametric derivative its values: on a
  cell 250 times longer than thick, near coordinates of 1 and a pressure of
  1e5, those sums lost the short edge and the small differences to rounding.
  The ONERA M6's boundary-layer wedges' f32 pressure gradients were off by
  more than 0.1% in 6,569,170 cells (14%), by more than 100% in 380, the worst
  93 times. Both are now formed from differences from the first corner and
  value (the shapes' gradients sum to zero, so the result is the same),
  exact in f32: 31 cells off by more than 0.1%, the worst 0.32%, as in f64
  (1 cell, 0.33%), at the same cost; on Metal, which has no f64, 26 cells,
  0.37%. Cell quality's Jacobian measures and order-2 geometry follow the
  same rule; locating points already did.

### Added

- `Field(space, values, unit, gradient=)`: a field's world gradient where
  something other than its basis gives it, such as a solver's own
  reconstruction, as a field on the same topology (3 values each for a scalar,
  3k row by row for a vector of k). `world_gradient`, `world_jet` and
  `gradients` read it, so `flow_quantities` and every kernel built on them use
  the solver's derivatives; the Hessian it does not determine is NaN.

### Changed

- `world_gradient` and `world_jet` ask the field's view, whose default
  (on every field view whose basis gives the derivatives) differentiates its basis as
  before. A space that computes its values can give its own.

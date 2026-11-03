### Added

- `register_field_resolver(resolve)`: every algorithm's field argument may be text
  that names no field, which `resolve(data, text)` turns into a `Field`. Field
  expressions register one, so `contour(data, "mag(U)", 1.0)` contours an
  expression.
- A space evaluated where it is read may give `default_subdivision`, the lattice
  steps contour, clip and threshold cut its cells into by default.

### Changed

- Filters carry a field's given gradient with it, and make a field evaluated where
  it is read again on their output (its space's `rebind`) instead of carrying
  its placeholder values.
- `dataset_to_vtk` writes such a field's values (its space's `stored`).

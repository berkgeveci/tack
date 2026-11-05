### Added

- Fields evaluated where they are read: a space may declare itself `evaluated`
  (its view computes the field by `value` and `corner_value`; it stores no values),
  `continuous` (its crossings are shared between cells, as `H1`'s are), its
  `default_subdivision` and its `integration_order`, and give `rebind` and
  `stored`. `contour`, `subdivide`, `clip` and `threshold` read such a field
  through its view (`threshold` by its values at the lattice points, not as cell
  data), filters make it again on their output, `integrate` integrates
  it by quadrature to its order, and `statistics`, `entropy`, `owned_values` and
  `dataset_to_vtk` read its stored values. `Space` declares each, with its
  default.

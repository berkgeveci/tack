### Changed

- `statistics`, `entropy` and `integrate` take any field argument: a name, a
  `Field`, or text a registered resolver reads. A field evaluated where it is read
  is reduced by its stored values (`owned_values` stores it) and integrated
  through its view by quadrature. `integrate` takes names, text, or a mapping of
  names to any of these.

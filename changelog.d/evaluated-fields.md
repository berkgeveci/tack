### Added

- Fields evaluated where they are read: a space whose `evaluated` attribute is
  true is read through its view's `value` and `corner_value` alone, and
  `contour`'s subdivision computes it at each lattice point; one whose
  `continuous` attribute is true shares its crossings between cells, as `H1`
  does, so its surface is watertight. An expression of continuous fields is
  such a field.

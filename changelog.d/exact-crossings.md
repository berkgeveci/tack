### Added

- Exact crossings: `contour`, `slice`, `slice_plane` and `clip` of a field above
  order 1 or a curved geometry put each crossing of a sub-cell's edge where the
  field itself takes the value along it (a fixed number of Illinois steps from
  the lattice values), and carry the geometry and every field there, so the
  vertices lie on the true surface whatever the subdivision. A crossing on an
  edge or face comes from that entity's own values in a frame its corners fix,
  so neighbouring cells and pieces put it at the same bits. `exact=False` keeps
  the crossing between the lattice values.

### Changed

- **Breaking:** the cuts above (with `exact=True`, the default) move their
  vertices onto the field's roots: positions and carried values differ from the
  lattice values' interpolation (by about the square of the steps); the cells are
  the same.

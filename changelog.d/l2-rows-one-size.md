### Fixed

- `L2.indices()` and `L2.blocks(values)` refuse cells holding different numbers
  of values (a mixed mesh, an order per cell), naming `space.offsets`. They cut
  the values into rows of the average size, so a tetrahedron and a hexahedron
  at order 1 read as two cells of 6 values each, raising only when the total did
  not divide evenly.

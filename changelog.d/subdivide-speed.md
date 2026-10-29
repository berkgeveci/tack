### Changed

- Subdividing a scalar field in a tensor basis (contour and slice above order
  2, `subdivide`) reads each cell's values once, not once a lattice point: an
  order-6 `H1` field's `dof_index` for every node at every point was most of
  the time. 32^3 hexahedra, an order-6 contour: Metal 457 to 122 ms, the CPU
  278 to 166 ms; order 3, Metal 31 to 27 ms. The values are the same, to the
  bit.

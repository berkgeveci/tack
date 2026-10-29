### Changed

- Subdividing a scalar field in a tensor basis (contour and slice above order
  2, `subdivide`) reads each cell's values once, not once a lattice point: an
  order-6 `H1` field's `dof_index` for every node at every point was most of
  the time. 32^3 hexahedra, an order-6 contour: Metal 457 to 122 ms, the CPU
  278 to 166 ms; order 3, Metal 31 to 27 ms. The values are the same, to the
  bit.
- Locating a point in a curved cell (the cell locator, `probe`, advection on a
  curved geometry) reads the cell's nodes once, not twice at each Newton
  step through `dof_index`: an order-3 curved probe of 100k points on 16^3
  hexahedra, Metal 177 to 73 ms, the CPU 52 to 28.5 ms. The same cells and
  coordinates, to the bit.

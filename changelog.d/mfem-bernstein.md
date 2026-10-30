### Changed

- The MFEM interop copies a positive basis' coefficients (`H1Pos`, and `L2`
  with `BasisType.Positive`) into Tack's Bernstein space of that order, on
  quads, hexahedra, triangles, tetrahedra and wedges, permuted as nodal values
  are; they were evaluated at Gauss-Lobatto (or equispaced) nodes.
  `mfem_grid_function` gives a Bernstein field back in MFEM's positive basis.

### Added

- `mfem_to_dataset(..., dofs="table")` and `mfem_field(..., dofs="table")` keep
  a curved geometry and `H1` fields of order 2 and up numbered as MFEM numbers
  them: the values as they are, in a space numbered by MFEM's element-to-DOF
  table (`H1(..., dofs=table)`), on every basis whose values are copied.

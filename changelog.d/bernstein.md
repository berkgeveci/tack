### Added

- Bernstein spaces: `H1(data, order=p, basis="bernstein")` and `L2(...,
  basis="bernstein")` on segments, quads, pixels, hexahedra, voxels, triangles,
  tetrahedra and wedges, and a curved geometry in Bernstein form. The values are
  the Bernstein polynomials' coefficients (control points, for a geometry),
  which bound the cell's field or geometry. `interpolate` and `project` fill
  them from values at the equispaced nodes. Each edge's and face's coefficients
  are formed in a frame its corners' positions fix, so pieces sharing it agree
  to the bit. Every filter sees the same polynomials as the equispaced Lagrange
  space of the order.
- VTK's Bezier cells (curves, quadrilaterals, hexahedra, triangles, tetrahedra,
  wedges), both ways, as Bernstein spaces: `vtk_to_dataset` reads them,
  refusing rational weights other than 1, and `dataset_to_vtk(...,
  higher_order="bezier")` writes them. `higher_order=True` chooses Bezier cells
  for a dataset with Bernstein spaces.
- `H1` of order 2 on segments.

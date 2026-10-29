### Added

- `H1` and `L2` of any order on triangles and tetrahedra, with equispaced nodes
  (`nodes="equispaced"`, VTK's and gmsh's Lagrange simplices): values shared by
  entity as on the tensor shapes (`p - 1` per edge, `(p - 1)(p - 2) / 2` per
  triangle face, `(p - 1)(p - 2)(p - 3) / 6` per tetrahedron), a triangle
  face's placed through its orientation in each cell. `interpolate`, `probe`,
  `gradients` and `carry` take them; polynomials of the order are reproduced
  exactly.
- `contour`, `slice` and `subdivide` of fields above order 1 on triangles and
  tetrahedra, on Freudenthal's sub-simplices of the cells' lattice: a lattice
  point shared by cells is computed once, and in every piece holding it to the
  same bits, so surfaces close and pieces agree.
- Curved triangles and tetrahedra of any order: a geometry in `H1` of order 3
  and up (equispaced) on simplices, located as it is (its Bernstein control
  points bound it, Newton inverts its map) by `probe`, the cell locator and
  advection, and subdivided by `subdivide`, `contour` and `slice` (order 2's
  curved simplices too), each shared lattice point from its edge's or face's
  own nodes.
- VTK's Lagrange triangles and tetrahedra, both ways
  (`dataset_to_vtk(..., higher_order=True)`, `vtk_to_dataset`), and MFEM's
  triangles and tetrahedra of any order: closed-uniform values copied, MFEM's
  default bases evaluated at Tack's equispaced nodes, exactly; curved MFEM
  simplex meshes come across as curved geometry.
- Wedges of any order (equispaced): `H1` and `L2`, curved wedge geometry,
  subdivision for `contour`, `slice` and `subdivide` (Freudenthal's triangles
  in layers), VTK's Lagrange wedges both ways, and MFEM's wedges.

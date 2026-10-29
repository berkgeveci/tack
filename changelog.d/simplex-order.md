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

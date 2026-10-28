### Added

- Bases of any order on segments, quads, pixels, hexahedra and voxels
  (`tack.dataset.bases`): `H1(data, order=p, nodes=...)` from order 3 and
  `L2(data, order=p, nodes=...)` from order 2 (or on Gauss-Legendre nodes from
  order 1), on Gauss-Lobatto nodes (the default, MFEM's for `H1`),
  equispaced ones (VTK's Lagrange cells) or, for `L2`, Gauss-Legendre ones
  (MFEM's default for `L2`). `H1` holds `p - 1` values per edge, `(p - 1)^2`
  per quad face and `(p - 1)^d` per cell after the points, each cell reading
  a shared edge in order from its lower point id and a face in the frame of
  its own row, so cells turned every way share them; `L2` keeps each cell's
  `(p + 1)^d` values. `interpolate`, `project`, `probe`, `gradients` and
  curved geometry of any order work on them; polynomials of degree `p` come
  back exactly.
- `subdivide(data, level)`: curved or high-order cells as linear sub-cells on
  each cell's lattice of `level` steps, fields at the lattice points (VTK's
  `vtkCellGridToUnstructuredGrid`). A lattice point shared by cells is one
  point, evaluated once, by the first cell holding it.
- The MFEM interop takes `H1` from order 3 and `L2` from order 2 on quads and
  hexahedra, and curved meshes of any order on them: on Gauss-Lobatto,
  Gauss-Legendre and closed uniform nodes the values are copied in Tack's
  numbering, on any other basis (Bernstein) evaluated at Tack's nodes,
  exactly either way. `mfem_grid_function(data, field, mesh)` gives a field
  back as an MFEM `GridFunction`, the same DOFs in MFEM's numbering.
- `contour`, `slice` and `slice_plane` take `subdivision=`: on the tensor
  shapes, a field above order 1 or a curved geometry is cut on linear
  sub-cells, by default as many lattice steps as the largest order of the
  field and geometry, and only the cells whose field's Bernstein
  coefficients can hold the value are refined. The surface has no cracks;
  its error falls as the square of the step. `subdivision=1` cuts the cells
  linearized at their corners.

### Changed

- Contours and slices of order-2 fields on hexahedra and quads, which were
  refused, are cut on sub-cells; of a curved geometry on them, which were
  linearized at the corners, likewise (`subdivision=1` for the corners).

### Fixed

- Points in curved cells are located in the cells as they are. The cell
  locator bounded and inverted every cell by its corners, whatever the
  geometry's order, so a point where a curved cell bulges past its corners
  (at order 2 as well) was missed or found in the wrong cell. A cell is now
  bounded by the box of its Bernstein control points, which hold it, and
  inverted by Newton's method on its own map.

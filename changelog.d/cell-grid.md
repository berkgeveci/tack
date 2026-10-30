### Added

- `vtk_to_dataset` reads a `vtkCellGrid` (VTK's discontinuous-Galerkin cells, as
  `vtkCellGridReader` reads `.dg` files): its cells on the shape attribute's
  corners, `C1` attributes copied (shared as `H1`, per cell as `L2`), `constant`
  ones as `Constant`, and every other `HGRAD` or Bezier basis resampled by VTK's
  evaluator onto Tack's equispaced space of its order, exact where the spaces
  agree. `HCURL`, `HDIV` and pyramids' rational bases are left out with a
  warning, curved pyramids refused. It needs a VTK newer than 9.7.1, with
  `vtkCellAttribute`'s cell-type accessors and the evaluator's interpolate-only
  mode. `test_vtk_cell_grid.py` holds every `.dg` file of VTK's test data to
  VTK's evaluator, cell centres, probe, ranges, linear cells and transform (it
  skips without them).
- `dataset_to_vtk(data, cell_grid=True)` writes a `vtkCellGrid`: each shape's
  cells a DG cell type (`vtkDGCell.SetCellSpec`), the geometry its shape
  attribute, order 1 in VTK's complete basis `C` (shared for `H1`, per cell for
  `L2`), higher orders in its arbitrary-order basis `A` (VTK's `A` nodes are
  Tack's equispaced nodes in Tack's order, which the tests show on hexahedra,
  quads, tetrahedra and wedges), `Constant` fields as `constant`. Reading takes
  each type's corners from its cell spec and orders types by the specs'
  offsets, so a grid round-trips.
- `vtk_to_dataset` reads every VTK higher-order cell Tack has a shape for, of any
  shapes, dimensions and orders in one grid (Lagrange, Bezier, quadratic,
  biquadratic, triquadratic, quadratic-linear, the 7-node triangle and cubic
  lines, with linear cells beside them): each type's nodes matched to Tack's by
  VTK's parametric coordinates and copied, and where Tack's cell has nodes a
  VTK cell lacks, VTK's own interpolation functions give them, exactly.
  `test_vtk_higher_order.py` holds VTK's higher-order test files and one cell of
  each type to VTK's interpolation, derivatives, contours, slices, clips,
  boundaries and edges (it skips without VTK's test data).

### Fixed

- `vtk_to_dataset` of a grid of VTK cell types Tack has no shape for (the
  biquadratic quad, among others) built a topology it could not read, with no
  word; they are read (above) or refused, naming the types. Pyramids above
  order 1 and quadratic polygons are refused.
- A serendipity field of integers truncated its face centres (computed in its
  own type); they are computed in floating point, as its basis is evaluated.

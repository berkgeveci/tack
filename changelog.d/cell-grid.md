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

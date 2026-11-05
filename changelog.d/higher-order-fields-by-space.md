### Fixed

- `vtk_to_dataset` of a grid of VTK's higher-order cells keeps a point array and a
  cell array of one name as two fields, as on linear cells; the cell array
  replaced the point array (`CylinderQuadratic.vtk`'s `velocity` and `pressure`).
  Found by `test_vtk_higher_order.py` with `TACK_VTK_DATA`, which CI does not
  set; it failed since fields by name and space (0.7). That test also compared
  Tack's measure of a bent quad, a fan from its mean since 0.7, with VTK's two
  triangles; it now measures both surfaces with `vtkCellSizeFilter`.

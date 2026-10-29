### Fixed

- `vtk_to_dataset` refuses VTK's quadratic cells (`vtkQuadraticTetra` and the
  rest), naming them, as its docstring said it did; they came across as
  cells no algorithm took, every one failing with "are not linear shapes".

### Added

- The VTK comparison harness (`benchmarks/vtk_compare`) times point gradients
  against `vtkGradientFilter`, and the performance page says what its dashes
  are: comparisons not made, not failures (VTK's Viskores filters have no cell
  gradient, volume or centers; the Linux run timed VTK against Tack's CPU
  alone).

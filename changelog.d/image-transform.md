### Changed

- `transform` keeps an image an image: the affine map is folded into its origin
  and direction (`UniformCoordinates`, any invertible direction, a shear too),
  in double precision, and no position is stored; VTK receives a `vtkImageData`
  with that direction matrix. A singular map still gives stored positions.

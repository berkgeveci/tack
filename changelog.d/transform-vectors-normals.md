### Added

- `transform(..., vectors=, normals=)`: fields that turn with the map, its linear
  part applied to a vector (a velocity, a displacement) and its inverse transpose
  to a normal, made unit length again so it stays normal to the moved surface.

### Fixed

- The VTK algorithm `Transform` carried the input's vectors and normals as they
  were, so after a rotation they pointed the old way (and normals lit a moved
  surface wrongly). It now transforms the input's active vectors and normals, on
  points and cells, as VTK's `vtkTransformFilter` does, unless `vectors` or
  `normals` say otherwise. Found running VTK's Python tests on Tack.

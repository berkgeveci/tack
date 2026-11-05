### Fixed

- The VTK algorithms' outputs had no active attributes: Tack keeps no roles, so the
  arrays it carried came back plain, and mappers colored by nothing, glyphs and
  lighting found no vectors or normals (VTK's tests rendered white). An output now
  keeps its input's active scalars, vectors, normals, texture coordinates and
  tensors where it carries arrays of those names (`Probe` its source's), as VTK's
  filters pass them; `Elevation` and `ImplicitValues` make their array the active
  scalars, `SurfaceNormals`, `PointNormals` and `Tubes` theirs the active normals
  (a tube took the lines' carried `Normals` before, and was lit wrongly). Found
  running VTK's Python tests on Tack.

### Fixed

- `dataset_to_vtk` makes normals VTK's active normals, which its renderers and
  filters read: a 3-vector point field named `Normals` or `normals`, and a
  3-vector cell field named `Normals`, `Normals (cells)` or `cell_normals`. A
  surface's normals carried through a Tack filter (a clip of a VTK surface
  with normals) came back as a plain array, and VTK drew the surface flat.

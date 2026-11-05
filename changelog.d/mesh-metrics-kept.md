### Changed

- `cell_geometry`, `face_geometry` and `surface_normals` keep what they compute
  with the mesh: computed once a mesh, shared by every dataset on the same
  geometry, and made again when the geometry changes (another geometry, or
  positions Tack has written). `extract_surface` of the same mesh gives the same
  surface geometry, so the surface's normals are computed once over many steps
  that change only fields.

### Fixed

- `face_geometry`, `upwind_flux`, `net_flux`, `divergence`, `perot` and
  `surface_normals` take a surface with lines among its polygons (EnSight's
  and VTK's surfaces carry them), the lines having no edges of the
  surface's: zero flux and a zero normal. `net_flux`, `divergence` and
  `perot` of a mesh of vertices are zero. Both failed inside the kernels
  (`no @tack.func method 'side_position'`, `'num_faces'`).

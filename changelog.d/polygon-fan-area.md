### Fixed

- `integrate` and `cell_geometry` measure a non-planar polygon (and a non-planar
  quad of a surface mesh) as its fan of triangles from its points' mean, Tack's
  model of a polygon and the area VTK reports, not as its projection onto its
  Newell normal, which is smaller. Tack's contours of hexahedra are polygons,
  and a wrinkled one measured 1% small (the dam break's free surface; the Ford
  DrivAer's wake 7.5e-4). Each triangle's area is signed by the normal, so a
  planar polygon whose mean lies outside it still measures exactly.

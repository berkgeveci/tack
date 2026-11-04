### Fixed

- `integrate` and `cell_geometry` measure a non-planar polygon as its fan of
  triangles from its points' mean, Tack's model of a polygon and the area VTK
  reports, and a quad as its bilinear patch (by 4 x 4 Gauss points, within
  rounding unless it is twisted far out of plane), not either as its
  projection onto its Newell normal, which is smaller. Tack's contours of
  hexahedra are polygons, and a wrinkled one measured 1% small (the dam
  break's free surface; the Ford DrivAer's wake 7.5e-4). Both are signed by
  the normal, so a planar polygon whose mean lies outside it, or a planar quad
  that is not convex, still measures exactly. VTK measures a quad as its
  projection.

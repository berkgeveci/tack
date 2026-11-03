### Changed

- **Breaking:** `extract_surface` puts a surface on the points its faces use,
  renumbered in order, as VTK's geometry filter does, not on all the input's
  points. A surface's point statistics are now the surface's, and the surface
  of pieces matches 1.24M points on the AhmedML mesh, not 45.6M. A 2D mesh
  passed through keeps its points if its cells use them all. The fresh
  boundary's renumbering is kept with it, so a repeated call still costs only
  the carry.

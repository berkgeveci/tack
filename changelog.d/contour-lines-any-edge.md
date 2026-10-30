### Fixed

- `contour` and `slice` of a 2D mesh match its cells' sides by their points
  instead of through the topology's faces, so a surface with an edge of three
  or more cells (PyVista's action figure, room and sextant) gets its lines,
  meeting there, where they refused the whole mesh. The topology keeps the
  matched sides for the next call.

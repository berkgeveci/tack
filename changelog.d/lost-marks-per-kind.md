### Fixed

- A filter's output from pieces without ghost cells (`split` without ghost
  layers, the split reading of shaped cells and polyhedra) refused every
  reduction, though only the points it makes on shared edges and faces have no
  owner: each cell it made from one input cell keeps that cell's. The loss is now
  recorded per kind, so `integrate` (a contour's area) and statistics of cell
  fields work on such outputs, and point reductions still refuse.
  `extract_surface`'s cells, faces the pieces share, lose their marks too.
  Setting a dataset's `ghosts` to `None` clears the record, as other marks did.

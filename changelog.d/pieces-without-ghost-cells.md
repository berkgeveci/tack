### Added

- `vtk_to_dataset(grid, merge_points=True)` (or a tolerance) merges an
  unstructured grid's coincident points before reading it, as a mesh written
  per processor needs: the polyhedra's faces are matched once, and with
  `polyhedral="split"` every point is owned by one piece. Merging the pieces
  afterwards could leave a point owned twice or by no piece.
- `generate_global_ids` of a filter's outputs from pieces without ghost
  cells (which lost their marks, so reductions refuse them) matches every
  point and cell and marks the owners, so `statistics` and `entropy` then give
  the whole's answer; the refusal names it. Such outputs keep their cells'
  marks from their source cells.

### Fixed

- `generate_global_ids` matches every point, not only boundary points, when
  pieces have ghost cells but no point marks.

### Added

- `subdivide`, and `contour` and `slice` on sub-cells, keep ghost marks: a
  sub-cell takes its parent's, and a lattice point is marked as `carry` marks
  the points it makes, so each piece's cut of a high-order field is its share
  of the whole's. A shared lattice point has the same bits in every piece
  holding it: an edge or face point is computed from that entity's own
  values in a frame its corners' positions fix, and anything else in the cell
  every piece chooses alike (the lowest global id, or the first centroid by
  position). Pieces' surfaces merge with `merge_pieces` into the whole's.

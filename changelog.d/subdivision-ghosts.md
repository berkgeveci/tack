### Added

- `subdivide`, and `contour` and `slice` on sub-cells, keep ghost marks: a
  sub-cell takes its parent's, and a lattice point is marked as `carry` marks
  the points it makes, so each piece's cut of a high-order field is its share
  of the whole's. A shared lattice point has the same bits in every piece
  holding it: an edge or face point is computed from that entity's own
  values in a frame its corners' positions fix, and anything else in the cell
  every piece chooses alike (the lowest global id, or the first centroid by
  position). Pieces' surfaces merge with `merge_pieces` into the whole's.

### Fixed

- `interpolate` places a shared node on a quad face, and an order-3 or higher
  node on an edge, in a frame the entity's corners' positions fix (an edge from
  its first point by position, a face from its first corner toward the first
  of that corner's neighbours), not in its stored point order, which follows a
  piece's point ids. Pieces holding a face computed its nodes from different
  corners, so a curved geometry or field interpolated on each piece differed
  from the whole's in the last bits there (up to 2e-7 in f32). Values may
  differ from before in their last bits.

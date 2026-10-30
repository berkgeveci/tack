### Added

- `interpolate` (and `node_views`) on a curved geometry of another order, on
  hexahedra, quads and the other tensor cells: an `H1` field of order 3 on an
  order-2 geometry, say. Each edge's and quad face's nodes are on the
  geometry's trace there, read from that entity's own geometry values in a frame
  its corners fix, so every piece holding it gets the same bits. It was
  refused; curved simplices and wedges of another order still are.

### Fixed

- An order-2 `H1` field on a geometry curved to order 3 or more had its edge and
  face nodes at the mean of their corners, off the curved cells; they are now
  on the geometry's traces. Node views of a curved geometry and an `L2` field of
  another order no longer fail to build (both defined `ORDER`).

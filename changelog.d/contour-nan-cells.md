### Changed

- `contour` gives no surface, lines or vertices in a cell with a NaN among its
  values: the surface stops at the cells holding numbers. A NaN was below
  every level, and a crossing on an edge to one lay at the edge's finite end,
  a point not in the data; VTK's lies at NaN.

### Fixed

- `contour` of a 2D mesh finds each side's crossings from its two points, not
  from the side's first cell, so a crossing every line needs is always made.

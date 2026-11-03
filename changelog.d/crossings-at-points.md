### Added

- `td.edge_crossing(level, a, b, va, vb, xa, xb)`, the rule every built-in cut
  places a crossing by, for filters of one's own: the ends in the order to
  interpolate from (the end first by position) and the weight, or the end
  itself when the crossing is that point.

### Changed

- `contour` and `slice` make a crossing that lies within half an ulp of a
  point (relative to its edge's largest coordinate) that point, as they did
  when the point's value is the level, so every edge's crossing there is one
  point. A plane a hair from mesh points, such as a symmetry plane's points
  at y = 1e-45, had put several points at one position, and on pieces with
  ghost cells `generate_global_ids` of such slices failed. Slivers whose
  points all lie that close to one point have no area and are dropped:
  Viskores' wedge gyroid gives 49 triangles, where Viskores gives 52.

### Fixed

- Lines cut from surfaces interpolate each crossing from its side's end first
  by position, not from its lower point id, so pieces sharing a side compute
  it to the same bits.

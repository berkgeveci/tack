### Changed

- `extract_extent` of a flat image or rectilinear grid (a 2D grid in Tack)
  takes VTK's extent and rates in three axes, as VTK gives a flat image's
  (`(i0, i1, 0, 0, k0, k1)`), the flat axis's ignored; it took one pair per
  axis the grid has only.

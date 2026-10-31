### Fixed

- Filters no longer store an image's or a rectilinear grid's positions, or
  an implicit field's values, to read them once: `point_elevation`,
  `implicit_values` (so `slice`, `clip`, `extract_geometry` and
  `extract_points` by an implicit function), `carry` of taken and
  interpolated points (`threshold`, `clip`, `extract_cells`, `mask`, ...),
  `warp`, `transform`, `cylindrical`/`spherical`, `glyphs`, `tubes`,
  `feature_edges`, `composite`, the 1D and 2D contours, clipping polyhedra,
  advection and `voxelize` pass them to their kernels as they are. A 256³
  image's positions are 200 MB; `warp` without `scale_by` also no longer
  stores an array of ones. On that image `point_elevation`, `implicit_values`
  and `warp` run 2-5 times faster.

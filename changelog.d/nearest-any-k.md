### Changed

- `PointLocator.nearest`, and with it `outliers`, `point_normals` and
  `ghost_points` by `k`, take any number of neighbours: up to 32 are kept in
  private arrays as before, more in the output's rows. `k` was at most 32
  (PyVista's `point_cloud_neighbors.py` asks for 40).

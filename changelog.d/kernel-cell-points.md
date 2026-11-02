### Added

- `CellLocator` locates in polyhedra that are not star-shaped from the mean of
  their points, as snappyHexMesh's concave cells at walls are: each one's cell
  point moves into its kernel, where it sees all of its faces, given as
  weights on its points so linear fields stay exact
  (`CellLocator.moved_cell_points`). `probe`, `advect` and `streamlines` use
  it. On an AhmedML volume (21.5M cells read as polyhedra), 119 of the 131
  cells the locator refused have a kernel; only cells with none (tangled or
  inside out: 12 there, 10 of them among the file's 508,298 polyhedra) are
  refused, as before.

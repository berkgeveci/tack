### Added

- `CellLocator` locates in polyhedra that are not star-shaped from the mean of
  their points, as snappyHexMesh's concave cells at walls are: each one's cell
  point moves into its kernel, where it sees all of its faces, given as
  weights on its points so linear fields stay exact
  (`CellLocator.moved_cell_points`). `probe`, `advect` and `streamlines` use
  it. On an AhmedML volume (21.5M cells read as polyhedra), 119 of the 131
  cells the locator refused have a kernel.

### Changed

- A polyhedron with no kernel (tangled or inside out, a face whose
  fan folds, no volume) no longer makes `CellLocator`, and with it `probe` and
  advection, refuse the whole mesh: it holds no points, with a warning. A point
  inside it is found in a neighbour it overlaps, or nowhere; a particle reaching
  it stops. The AhmedML volume has 12, wall cells at the nose's corners.

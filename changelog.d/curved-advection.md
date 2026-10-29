### Added

- `advect` and `streamlines` through curved cells (an `H1` geometry above
  order 1) and with a velocity of any order, or in `L2`, on hexahedra,
  voxels, quads, pixels and the order-2 tetrahedra, wedges and triangles.
  The search from the points reads a curved cell and a high-order field
  through its shape group's own view, so it inverts a curved cell by its own
  map and evaluates the field as `probe` does: the paths are an RK4 over
  probed velocities to 1e-12. `probe` on a curved geometry searches from the
  points too, with the same answers as from the cells. Advection refused
  both before. A mesh of one shape compiles its kernels in seconds; a curved
  mesh of several shapes takes minutes on its first call.

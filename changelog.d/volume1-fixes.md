### Added

- `advect(..., step_unit="length")`, as `streamlines` takes it: `step_size` an
  arc length along the path rather than a time, over pieces too.
- `UnstructuredTopology.cell_points()`, each cell's points sorted and unique, as
  polyhedral and polyline topologies give them.

### Fixed

- Point gradients (`gradients(..., at="points")`) were NaN at a corner where a
  cell's Jacobian is singular, as where two of a hexahedron's faces meet flat
  (17 points of the AhmedML mesh). Such a cell has no gradient there -- it
  grows without bound toward the corner -- and is now left out of that point's
  average, the others' mean taken; VTK leaves its inverse Jacobian unset there,
  and Viskores counts the cell but adds nothing.

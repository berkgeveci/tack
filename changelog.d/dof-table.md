### Added

- Element-to-DOF tables: `H1(data, order=p, dofs=table)` numbers a space's values
  by an explicit table, a row per cell giving each node's value index, as MFEM
  and VTK's cell grids store them, instead of by entity. A cell's corners must be
  its points; the other nodes may be numbered freely (VTK's layout, every node a
  point, included). Every filter reads it, a curved geometry may be numbered so,
  and filters' outputs number by entity again.

### Fixed

- `interpolate` on a curved geometry numbered otherwise than the target space
  (by a table) placed nodes at the wrong positions.

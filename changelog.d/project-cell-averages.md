### Added

- `project(..., average="nodes")`: into `Constant` (or order-0 `L2`), the mean
  of the source's own values in each cell (its `NODES`), as
  vtkPointDataToCellData and Viskores' CellAverage take it, for every VTK
  cell type and order.

### Changed

- **Breaking:** `project` into `Constant` (or order-0 `L2`) gives each cell
  the source's average over it, its integral divided by the cell's measure,
  by quadrature through the cell's own map, where it took the mean of the
  cell's corner values. The two agree for a linear field on a tetrahedron, a
  triangle or an image's (or rectilinear grid's) boxes, which still take the
  corners' mean; they differ on pyramids (the apex weighs a quarter of a
  linear field's average, not a fifth), on distorted cells, and for fields of
  higher order, whose nodes other than the corners the corner mean never
  read. Polyhedra, polygons and polylines average a point field over the
  pieces they are cut into. `average="nodes"` gives VTK's mean.

### Added

- `clip` and `threshold` of a curved geometry or a field above order 1, on
  every shape `contour` subdivides: the cells whose field's Bernstein bound
  lies wholly on the kept side stay as they are (curved, their fields of any
  order), and only the others are cut, on linear sub-cells of their lattice
  (`subdivision=`, `tolerance=`, as `contour` takes them).

### Changed

- **Breaking:** `clip` and `threshold` of a curved geometry or a field above
  order 1 return `PartitionedDataSet([whole, part])`, the cells kept whole and
  the cut part, where both went by the cells' corner values alone (and `clip`
  made curved cells straight). Straight cells with order-1 fields, and
  `threshold` by a cell field, still give one `DataSet`.

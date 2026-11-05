### Added

- A field argument may be an object whose `on(data)` gives the `Field` on `data`,
  as a field expression is. Over a `PartitionedDataSet`, `statistics`, `entropy`
  and `integrate` call it once a piece, so an expression reduces over pieces as a
  name does.

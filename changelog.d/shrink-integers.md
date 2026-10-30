### Fixed

- `shrink` keeps integer point fields, each cell's corners copied as they are;
  it failed on them. `project` from order-1 `H1` into order-1 `L2`, a copy,
  takes integers.

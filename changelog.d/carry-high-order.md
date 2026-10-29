### Fixed

- Filters that keep whole cells (`extract_cells`, `threshold`, `split`,
  `remove_ghosts`, and any filter written with `carry` and a `Same` or
  `Take` cell map) keep an `H1` field above order 1, and a curved
  geometry, at its order and nodes. They carried the corner values only, so
  a filter's piece of a curved or high-order mesh came out straight and
  linear. Each output cell's nodes are its source cell's, to the bit.

### Fixed

- `warp` takes a direction or `scale_by` given as values on points when the
  geometry is order-1 `H1`, as `surface_normals` gives its normals; it
  refused them as "not in the geometry's space".

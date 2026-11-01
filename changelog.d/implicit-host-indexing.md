### Added

- Implicit arrays read one value from the host, as a stored `Array` does:
  `values[k]` (a Python number, or a NumPy array of a vector's components;
  negative `k` from the end) and `len(values)`, so `data.geometry.values[0]` works
  on images and rectilinear grids. Each read is one launch; iterating raises, as
  on an `Array`. Underneath, a `@tack.func` may now have a host implementation
  (`@name.host`), which Python calls while kernels inline the device body.

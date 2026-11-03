### Fixed

- A CPU device without f64 (the tests' stand-in for Metal) loaded kernels a
  CPU with f64 had stored in the disk cache, and so ran an f64 kernel Metal
  refuses; its cache target now says it has no f64. Point gradients at
  singular corners are also run on it in the tests, which is where #268's f64
  cast would have been caught.

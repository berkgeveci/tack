### Fixed

- An RK45 particle at a wall the flow turns back from ends there
  (`SPATIAL_BOUNDS`), as `vtkStreamTracer`'s RK45 ends it out of its domain.
  Its stage points swung past the wall at every step size, or its stages'
  directions cancelled, so each step's bisection pushed it out by about an
  ulp, still inside, and it crawled until its steps ran out: 86 of 20,000
  aneurysm paths, on every device, and on a GPU those few threads set every
  launch's length (20,000 seeds took 6.4 s on CUDA, 14 s on HIP and 44 s on
  Level Zero, against 1.4-2.9 s on the CPU). A push out that moves the
  particle less than `min_step` now ends it; on the aneurysm 2 paths still
  run to the step limit, and the CPU run takes 1.4 s instead of 2.3 s. Euler
  and RK4 are unchanged.

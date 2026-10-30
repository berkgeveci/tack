### Added

- `streamlines` and `advect` take `terminal_speed` (in the velocity's unit, or a
  quantity), `vtkStreamTracer`'s TerminalSpeed: a path ends at its first point
  where the speed is at or below it, the seed included, and the particle is
  `TERMINATE` and `SLOW_SPEED` (a new `ParticleStatus` bit, 256). It holds over
  pieces and launches as `max_length` does. Without it a path in a field that
  is near zero but not zero (the carotid's outside its vessel) wandered on to
  its length or step limit. VTK also drops a step whose two ends' average
  speed is at or below the limit, so it may end a path one point sooner.

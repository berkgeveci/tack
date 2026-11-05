### Added

- `RunningStatistics`, `TimeDerivative`, `Advection`, `Pathlines` and
  `Streaklines`, VTK algorithms of Tack's temporal filters, in two modes. In
  situ, each execution takes the step the input holds and gives the output at
  it, stamped with its time; the same step again is not taken twice, and a step
  back in time, a parameter changed or VTK's `NO_PRIOR_TEMPORAL_ACCESS` reset
  starts over. Driving, the filter asks its input for the steps it needs, one
  per pass and never one it has taken: the statistics, particles and paths over
  every step (or those in `time_range`) in one output with no time (time in the
  data: `step` and `time` on the particles, `time` on the paths' points), and
  streaklines and derivatives at the time requested, going on from the steps
  taken. A filter drives where its input has time steps and the pipeline can go
  back in time; `mode=` chooses.

### Fixed

- `Advection` (and `Pathlines`, `Streaklines`) over pieces on MPI ranks waited
  forever where a rank had no particles, released particles at other steps, or
  saw its particles stop at another substep: the probes are collective, and a
  rank skipped them. Every rank now takes part in each.

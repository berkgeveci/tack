### Fixed

- Filters over pieces whose points are marked copies stalled now and then: the
  check whether any mark has a bit (`ghosts._has`, on every `carry`) raised one
  atomic flag for every marked element, so the CPU kept it serial (0.14 s over
  the AhmedML mesh's 22.8M points) and its rechecks fanned it out at 1-1.2 s.
  It is now a flag per chunk of the marks, without atomics: a slice of the
  polyhedra piece takes 0.012 s every call, where it took 0.15 s with spikes
  of 1.2.

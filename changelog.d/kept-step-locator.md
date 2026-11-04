### Changed

- A topology's cell locator serves a geometry of the same positions: where it
  does not serve a dataset, `probe` and advection compare the positions on the
  device and, if they are equal, keep its bins (the search made again from the
  new positions). `Pathlines`, `Streaklines` and `Advection` keep a copy of the
  previous step, whose positions are the step's, and built every mesh's
  locator twice: half of a step's time on the dam break's adaptive meshes
  (1.36 s to about 0.6 for 100 particles over 30 meshes). A simulation that
  hands over new arrays of unchanged positions each step no longer rebuilds
  them either.

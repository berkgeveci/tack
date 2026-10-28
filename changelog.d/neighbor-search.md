### Added

- `advect(..., neighbors=True)` and `streamlines(..., neighbors=True)`: a stage
  whose particle has left the cell it was last found in tries that cell's face
  neighbours before the locator's bins, taking one, as the last cell is taken,
  only where it holds the point well inside, so the paths are the bins' bit
  for bit. On the CPU, streamlines on the aneurysm (8.7M tetrahedra) took 1.45
  to 2.04x less time (20,000 seeds: 1116 to 546 ms) and on the ONERA M6 (46.9M
  cells) 1.63 to 1.90x (16,384 seeds: 2067 to 1086 ms). About one location in
  seven missed the last cell, and 43-49% (aneurysm) and 78% (M6) of the misses
  were in a face neighbour. The neighbours are a table of six ids a cell
  (`CellLocator.cell_neighbors()`), built from the faces on first use and kept
  with the locator: 0.9 s and 210 MB on the aneurysm, 4.1 s and 1.1 GB on the
  M6, so the walk is off by default. Pyramids, polyhedra and surfaces search
  the bins as before.

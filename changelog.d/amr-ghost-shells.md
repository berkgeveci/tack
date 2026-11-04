### Fixed

- `ghost_cells` of an `AMRDataSet` looks for sources only in each grown box's
  shell (its ghost cells, and its own points on its boundary), not in every
  cell and point it holds, places each box's values on the device, and looks
  each box's device up once: on AMR-Wind's levels 0 and 1 (6,436 boxes) it
  went from 7.6 s to 3.6, with every value the same to the bit, and on all
  five levels (15,598 boxes, 183M cells) it takes 20 s, `average_down` 9.

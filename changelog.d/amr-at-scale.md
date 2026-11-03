### Fixed

- AMR hierarchies of thousands of boxes took minutes or more: making an
  `AMRDataSet` compared every box with every other in Python, and
  `ghost_cells` and `average_down` built their sources box by box and filled
  through a plan with a pair for every two boxes that touch (AMR-Wind's 6,436
  boxes: 326,480 launches a fill). The marks are computed a level at a time,
  boxes that meet are found by a sort and sweep, the sources by kernels over a
  level's boxes at once, and a fill runs once per device's boxes. On AMR-Wind's
  levels 0 and 1 (6,436 boxes, CPU, VTK master), making the dataset from VTK's
  went from 129 s to 3.2, `ghost_cells` from more than 40 minutes to 7.6 s,
  and `average_down` from not finishing to 2.9 s, with every value the same to
  the bit; all five levels (15,598 boxes, 183M cells) take 8.2, 30 and 11 s.
- `vtk_to_dataset` of a `vtkOverlappingAMR` gives every box the hierarchy's
  origin, not one chosen from the box's own bounds, and places each box as it
  converts it.
- `average_down` of an image box weighs its cells by the product of its spacing,
  every cell alike, not by each cell's measure from its corners, which could
  differ in the last bits.

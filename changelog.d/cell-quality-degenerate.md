### Fixed

- `cell_quality` scores a degenerate cell (an edge of no length, or no area or
  volume) 0 in its scaled Jacobian and Jacobian ratio, as Verdict does for
  triangles, quads and tetrahedra; it gave the largest float, of either sign.
- `cell_quality` measures a polygonal topology's triangles and quads as those
  shapes, not by fans from their centers (an equilateral triangle scored 0.577
  beside a pentagon), and a polygon's fan triangles' condition number is
  Verdict's triangle's; it was twice that.

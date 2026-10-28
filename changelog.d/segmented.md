### Added

- Segmented operations in `tack.algorithms`: `segment_reduce` (sum, min, max, mean),
  `segment_scan` (inclusive and exclusive sums, running min and max),
  `segment_broadcast` and `segment_ids`, over flat values and CSR offsets,
  empty segments kept -- each cell's points, each point's cells, sparse rows.
  A segment of up to 256 elements is one thread's; a longer one goes by
  chunks of 256, a segment over several chunks combined a level up, so one
  enormous segment is a few launches, not one thread's walk. On an M1 Max's
  GPU a 10M-element segment reduces in 4 ms (a thread per segment: 1.2 s), a
  skewed mix of 1M short segments and one of 5M in 3.7 ms (590 ms), and 2.5M
  segments of about 4 in 0.66 ms (0.59). No atomics in the arithmetic, so
  float sums are the same bits on every backend; `ordered=True` sums each
  segment in element order, as `reduce_by_key` does. Empty segments take
  `empty=` (0, NaN for a mean, or the type's extremes for min and max), or
  keep their output with `empty="keep"`; vectors reduce componentwise; debug
  mode checks the offsets.

### Changed

- Averages over each point's cells -- `gradients(at="points")`, the
  cell-to-point `project` and `surface_normals` -- are segmented means over
  the point links (`segment_reduce(..., op="mean", empty="keep")`), the values
  read through the links, not gathered. A point of up to 256 cells sums them
  in the links' order as before, so ordinary meshes give the same bits; a
  point of more is summed by chunks, where one thread walked its links. At
  the apex of 403,200 tetrahedra on an M1 Max's GPU: point gradients 294 to
  5.7 ms, cell to point 271 to 3.4. On the CPU, where one core walks such a
  point quickly, about the same (5M tetrahedra 14.8 to 14.1 ms; the apex 1.7
  to 2.4).

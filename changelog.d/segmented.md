### Added

- Segmented operations in `tack.algorithms`: `segment_reduce` (sum, min, max),
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
  `empty=` (0, or the type's extremes for min and max); vectors reduce
  componentwise; debug mode checks the offsets.

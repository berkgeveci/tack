### Fixed

- `contour`, `slice` and `clip` interpolate each crossing from its edge's end
  first by position (x, then y, then z; the lower id only between coincident
  points), not from its lower point id. Ids are a piece's own, so two pieces
  holding an edge could interpolate its crossing from opposite ends: with an
  f32 field the copies differed in their last bits (up to 4e-8 on a wavelet),
  and `merge_pieces` of ghosted unstructured pieces' isosurfaces failed
  ("marked duplicates but no piece holds them unmarked") for want of a
  tolerance. Every piece now computes a shared crossing, and the values
  carried there, to the same bits, from the same end ghost marks already
  give it its owner by; within one mesh, as before, every cell sharing an
  edge agrees. Output points and their order are unchanged; their last bits
  may differ from before.

### Added

- A gallery in the Visualization Guide: contour, slice, clip, threshold,
  surfaces with feature edges, cell quality, gradients, probe, subdivide and
  pieces with ghost cells on polyhedra,
  tetrahedra, an order-4 field on straight hexahedra, an order-3 field on
  curved ones and a mixed mesh, each picture a script in `docs/gallery/` (Tack makes what is
  drawn, VTK draws linear polygons and lines), rebuilt by
  `docs/gallery/build.py`.
- `extract_edges(..., subdivision=n)`: each edge as `n` line cells through points
  on a curved geometry's trace, so a mesh's outline shows its cells as they are.
- `gradients(data, field, at=L2(...))`: the gradient at each cell's nodes of an
  `L2` space, exact there, so a high-order field's gradient keeps its order's
  detail, which averaging to the points loses.
- `surface_normals(..., weights="area")`: point normals with each cell's normal
  weighted by its area, so a cut surface's slivers do not swing them.

### Fixed

- `contour` with `subdivision=` on polyhedra, polygons or polylines sent them to
  the sub-cell path, which failed; they are cut as they are, as `clip` and
  `threshold` cut them.
- `as_polygons` of a polygonal dataset failed; it comes back as it is.
- `ghost_cells` and `merge_pieces` made a curved geometry straight with no word
  (they move positions at the points); they refuse it.

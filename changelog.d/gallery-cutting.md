### Added

- A gallery in the Visualization Guide: contour, slice and clip on polyhedra,
  tetrahedra, an order-4 field on straight hexahedra and an order-3 field on
  curved ones, each picture a script in `docs/gallery/` (Tack makes what is
  drawn, VTK draws linear polygons and lines), rebuilt by
  `docs/gallery/build.py`.
- `extract_edges(..., subdivision=n)`: each edge as `n` line cells through points
  on a curved geometry's trace, so a mesh's outline shows its cells as they are.
- `surface_normals(..., weights="area")`: point normals with each cell's normal
  weighted by its area, so a cut surface's slivers do not swing them.

### Fixed

- `contour` with `subdivision=` on polyhedra, polygons or polylines sent them to
  the sub-cell path, which failed; they are cut as they are, as `clip` and
  `threshold` cut them.
- `as_polygons` of a polygonal dataset failed; it comes back as it is.

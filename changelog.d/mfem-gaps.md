### Changed

- `H1` and `L2` without a `basis` take the mesh's own family above order 2:
  equispaced where the mesh has triangles, tetrahedra or wedges (the one family
  they have), Gauss-Lobatto otherwise. The default was refused on simplices.
- `extract_edges` of an `L2` geometry draws each cell's own edges on points of
  their own along the cell's map (`subdivision` segments each): its cells need not
  meet. It was refused.

- `integrate` takes point data on polyhedra, polygons and polylines, exactly:
  the values are linear on their sub-tetrahedra, triangles and segments. It
  took cell data alone.
- A curved `L2` geometry on triangles, tetrahedra and wedges (a mesh displaced
  by a DG field, a periodic mesh of order 2 or more) maps, measures and inverts
  its cells as its `H1` twin (`_L2SimplexGeometry`, `_L2WedgeGeometry`, over
  the `_SimplexGeometry` and `_WedgeGeometry` the `H1` ones now share). It was
  refused.
- `subdivide`, and the cuts that subdivide (`contour`, `slice`, `clip`,
  `threshold`), take meshes of several families of one dimension (hexahedra,
  wedges and tetrahedra, as MFEM's mixed Fichera corner): each family is refined
  alone and the parts joined at the lattice points they share. Their cuts'
  crossings are placed linearly on the sub-cells (exact crossings stay a
  one-family feature). They were refused.

### Fixed

- A curved cell is inverted to rounding: two Newton steps past VTK's stopping
  rule (`CURVED_NEWTON_POLISH`). Probes of MFEM's curved meshes agreed with MFEM
  to 3e-9; now to 1e-15.
- `extract_edges(..., subdivision=)` placed a curved edge's points at f32
  fractions of it (`(k + 0.0) / level`), about 1e-8 off the curve in f64.

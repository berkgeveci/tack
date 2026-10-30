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

### Fixed

- A curved cell is inverted to rounding: two Newton steps past VTK's stopping
  rule (`CURVED_NEWTON_POLISH`). Probes of MFEM's curved meshes agreed with MFEM
  to 3e-9; now to 1e-15.
- `extract_edges(..., subdivision=)` placed a curved edge's points at f32
  fractions of it (`(k + 0.0) / level`), about 1e-8 off the curve in f64.

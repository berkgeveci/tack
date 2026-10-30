### Added

- Serendipity: `H1(data, order=2, basis="serendipity")` on quads, pixels,
  hexahedra, voxels and wedges (the full order-2 space on triangles and
  tetrahedra): the corners' and edges' values alone, the quad faces' centres and
  the cells' insides determined from them. Every filter reads it as the full
  order-2 space, with no kernels of its own; a geometry may be serendipity too.
- VTK's classic quadratic cells, both ways: `vtk_to_dataset` reads
  `VTK_QUADRATIC_QUAD`, `_HEXAHEDRON` and `_WEDGE` (quad8, hex20, wedge15) into
  serendipity, `VTK_QUADRATIC_TRIANGLE` and `_TETRA` into the full order-2 space;
  `dataset_to_vtk(..., higher_order="quadratic")` writes them (`True` chooses them
  for a dataset with serendipity fields).

### Fixed

- Order-2 Bezier cells read from VTK took the order-2 Lagrange view's node
  numbering for a Bernstein space's.

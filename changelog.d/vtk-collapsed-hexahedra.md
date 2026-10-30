### Fixed

- `vtk_to_dataset` reads hexahedra with corners repeated as finite-element
  files write other shapes (Tecplot's bricks, degenerate Exodus and Patran
  hexahedra) as those shapes: wedges, pyramids and tetrahedra. As hexahedra,
  each one's collapsed face, an edge, was claimed by every cell around it, and
  every face-based algorithm refused the mesh.

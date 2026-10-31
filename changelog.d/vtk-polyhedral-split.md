### Added

- `vtk_to_dataset(grid, polyhedral="split")`: a VTK grid mixing polyhedra with
  shaped cells as a `PartitionedDataSet` of two pieces over the same points --
  the shaped cells an `UnstructuredTopology`, the polyhedra a
  `PolyhedralTopology` -- sharing positions and point fields, each point owned
  by one piece (the other marks it `DUPLICATEPOINT`), global ids the grid's
  point and cell ids. A CFD mesh of 20.9M hexahedra, 67K wedges and 508K
  polyhedra reads in 4.6 s and 14.7 GB (`copy=False`), where reading every cell
  as a polyhedron did not finish in 64 GB.

### Fixed

- Reading hexahedra no longer sorts every hexahedron's corners to look for
  collapsed ones (Tecplot's bricks, degenerate Exodus hexahedra): only those
  with a collapsed edge are sorted, about 1.2 s and 2.6 GB less on 21M
  hexahedra.

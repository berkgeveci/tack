### Added

- `integrate(data, fields=None)`: a dataset's measure and its fields' integrals
  (`Integrals`), as VTK's vtkIntegrateAttributes, by Gauss quadrature in each
  cell's reference element through its own map (collapsed onto simplices and
  wedges; `integrate.quadrature`), so curved cells and fields of any order
  integrate exactly to rounding; owned cells only, and over a
  `PartitionedDataSet` every piece on every rank; the cells of the mesh's
  dimension only, as VTK's (a volume leaves out its surfaces, lines and
  vertices). Polyhedra take cell data.

### Fixed

- `cell_geometry` of a curved geometry (`H1` above order 1, a curved `L2` one)
  measured each cell by its corners; it measures it by quadrature through its
  map (a sphere's area was 3-6% short).

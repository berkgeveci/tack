### Changed

- Handing an unstructured grid between VTK and Tack costs less:
  `vtk_to_dataset` finds the cell types a grid holds without sorting them
  (`np.unique` was 8 ms for 1.3M cells), and `dataset_to_vtk` sets the cells
  without VTK's scan of every type for a polyhedron (4 ms). Together most of
  what a GPU pipeline spent between its filters.

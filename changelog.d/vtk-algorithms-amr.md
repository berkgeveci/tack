### Added

- The VTK algorithms take an AMR hierarchy (`vtkOverlappingAMR`) whole, as one
  `AMRDataSet`, where the Tack filter takes one: `Probe` (as its source),
  `Statistics`, `Entropy`, `Integrate`, `Voxelize` and `GhostCells`, and the new
  `AverageDown` (`td.average_down`, its input left as it was). A region that a
  finer box covers, and a point that a sibling box holds, counts once.
  `GhostCells` and `AverageDown` give a `vtkOverlappingAMR` of the input's boxes,
  and the next AMR filter takes the hierarchy as it was made, ghost layers too.
  The others run box by box, as Tack's filters do on AMR, each level's boxes as
  pieces and the cells finer boxes cover left out: a field filter gives the
  hierarchy with its fields, one that makes a dataset a
  `vtkPartitionedDataSetCollection` with a partitioned dataset per level.
  `GenerateGlobalIds`, `Streamlines` and `Advect` refuse a hierarchy. Over MPI
  ranks each rank holds its own blocks, as VTK spreads a hierarchy.
- `vtk_to_dataset(..., world=)`: the ranks a `vtkOverlappingAMR`'s blocks are
  spread over, a `World` or an mpi4py communicator, by default VTK's (`world()`).

### Fixed

- A `vtkOverlappingAMR` given to a VTK algorithm gave an empty hierarchy, with no
  error: VTK makes a hierarchy a partitioned collection, which they filtered as a
  tree into an output of the hierarchy's type that could not hold it.
- `vtk_to_dataset` of a `vtkOverlappingAMR` spread over MPI ranks marked each
  rank's boxes as if its own were the whole hierarchy; it now converts in VTK's
  world unless given one.

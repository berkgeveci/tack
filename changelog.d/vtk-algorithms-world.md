### Added

- Tack's VTK algorithms work over MPI ranks: the filters whose answer needs every
  piece (`GhostCells`, `GenerateGlobalIds`, `MergePieces`, and the reductions)
  take the other ranks' pieces, the ranks taken from VTK's global controller as
  VTK's parallel filters take them (its communicator made mpi4py's by
  `vtkMPI4PyCommunicator`; `pvbatch` sets one), or given as `world=` (an mpi4py
  communicator or a `World`), as with VTK's wheels, which have no MPI.
  `tack.interop.vtk.world()` gives VTK's processes as a `tack.communication.World`.

### Fixed

- `MergePieces` over MPI ranks gave the ranks other than the root nothing to
  convert, and failed there; they get an empty dataset of the output's type.

### Changed

- Tack's VTK algorithms (`tack.interop.vtk.algorithms`) keep one Tack dataset
  per VTK dataset, by its address, modification time and time step, dropped
  when the VTK dataset is deleted. Filters that read one input, or one filter
  run again after a parameter changed, share it and what it derives from its
  topology (locators, faces, point links), which every run rebuilt before: a
  second `Streamlines` on 2.6M tetrahedra took 13-86 ms against 380-540. A
  field filter's (`Gradients`, `Elevation`, `CellQuality`) output keeps its
  input's topology, so the filter after it does too.

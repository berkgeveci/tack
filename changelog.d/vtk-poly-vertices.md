### Fixed

- `vtk_to_dataset` reads a polydata's poly vertices (a vert cell of several
  points, as `vtkTableToPolyData` and `vtkParticleReader` make) as one vertex
  per point, each taking the poly vertex's cell data, as strips become
  triangles; they were refused. A VTK cell type Tack has no shape for is
  refused as such, naming the higher-order cells that convert only when the
  type is one.

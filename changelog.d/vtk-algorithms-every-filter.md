### Added

- A VTK algorithm (`tack.interop.vtk.algorithms`) for every `tack.dataset` filter
  but the summaries and the finite-volume operators that take values on faces,
  named as the filter in CamelCase: `ExtractCells`, `ExtractGeometry`,
  `ExtractExtent`, `Mask`, `Slice`, `BoundaryFaces`, `ExtractEdges`,
  `FeatureEdges`, `Tubes`, `Glyphs`, `CheckWinding`, `MergePoints`, `Triangulate`,
  `AsPolyhedra`, `AsPolygons`, `Shrink`, `Subdivide`, `Transform` (also of a
  `vtkTransform` or `vtkMatrix4x4`), `Warp`, `Cylindrical`, `Spherical`,
  `CellGeometry`, `FaceGeometry`, `CellCenters`, `FaceCenters`, `EdgeLengths`,
  `ImplicitValues`, `CellIds`, `PointIds`, `FlowQuantities`, `Project`,
  `ValuesAtCenters`, `Interpolate`, `Magnitude`, `Dot`, `Cross`, `Composite`,
  `LogValues`, `Convert`, `Jump`, `UpwindFlux`, `Voxelize`, `Advect`,
  `ExtractPoints`, `ThresholdPoints`, `MaskPoints`, `PointNormals`, `Outliers`,
  `VoxelDownsample`, `Deposit`, `SphDensity`, `SphInterpolate`, `HaloCatalog`,
  `Split`, `GhostPoints` and `RemoveGhosts`. Values on faces or edges come out as
  the faces' polygons or the edges' lines with the values as cell data; a grid a
  filter samples onto (`Voxelize`, `Deposit`) may be a `vtkImageData`.
- `Streamlines` and `Advect` take their seeds from the points of the data on
  input port 1 (`SetSourceConnection`), as `vtkStreamTracer` does, when `seeds`
  is not given; `SphInterpolate` takes its particles there, as
  `vtkSPHInterpolator` does.

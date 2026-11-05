### Changed

- **Breaking:** the VTK algorithms of points, lines and surfaces give `vtkPolyData`,
  as VTK's filters do, where they gave a `vtkUnstructuredGrid`: `Contour`,
  `Slice`, `SlicePlane`, `ExtractSurface`, `BoundaryFaces`, `ExtractEdges`,
  `FeatureEdges`, `SurfaceNormals`, `Tubes`, `Glyphs`, `Triangulate`,
  `AsPolygons`, the face filters, `Streamlines`, `Advect`, the temporal particle
  filters and the point-cloud filters. They now feed `vtkPolyDataMapper` and VTK's
  polygonal filters (`vtkCleanPolyData`, `vtkStripper`, `vtkDecimatePro`), which
  refused them. `Clip` and `MergePoints` give `vtkPolyData` for a `vtkPolyData`
  input, as `vtkClipPolyData` and `vtkCleanPolyData`, and the filters that keep
  their input's kind (`Warp`, `Transform`, `Probe`, ...) keep a polygonal one. The
  cells go into the polydata's vertices, lines and polygons, shared where they are
  all of one kind. Found running VTK's Python tests on Tack.

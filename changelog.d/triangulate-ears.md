### Fixed

- `triangulate` cut a polygon into a fan from its first corner, which covers a
  concave polygon twice in part and leaves part uncovered (a U-shaped polygon of
  area 6 gave triangles of area 9). It now cuts ears in the polygon's plane, as
  VTK's `vtkTriangleFilter`: `n - 2` triangles that cover the polygon once, each
  wound as it is. Found running VTK's Python tests on Tack (the concave polygons of
  `vtkDiscreteFlyingEdges2D` and `vtkImageToPolyDataFilter`).

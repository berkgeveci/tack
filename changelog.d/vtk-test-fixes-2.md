### Fixed

- `vtk_to_dataset` marks a cell `HIDDENCELL` where one of its points is
  `HIDDENPOINT`: VTK sees a cell only where it and all its points are visible (a
  structured grid blanked by points), and Tack's filters, which leave out hidden
  cells, contoured and cut through blanked points. The point marks are kept.
- The VTK algorithm `ConnectedComponents` refused its own output on `vtkPolyData`
  (it gave an unstructured grid where it promised its input's type): its arrays go
  onto a copy of the input, as VTK's `vtkConnectivityFilter` keeps it.
- `project` of a polygon (polyhedron, polyline) with no points gave `inf`, having
  read past the cell: it is `NaN`, the cell having no value. These three found
  running VTK's Python tests on Tack.

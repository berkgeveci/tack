### Added

- `Contour` and `Threshold` take a vector field's `component` (the first by
  default, as VTK's filters; `Threshold` also `"magnitude"`), as a view of the
  vector's values: they refused a vector field. Found running VTK's Python tests
  on Tack (`multipleComponentContour.py`).

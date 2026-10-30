### Fixed

- `outliers`' docstring no longer says its statistical test is VTK's: it marks
  points far above the mean distance only, as PCL; `vtkStatisticalOutlierRemoval`
  also removes those far below it.

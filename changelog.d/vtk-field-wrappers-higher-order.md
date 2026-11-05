### Fixed

- The VTK algorithms that add fields (`Elevation`, `Gradients`, `Project`, ... and
  `RunningStatistics`, `TimeDerivative`) on VTK's higher-order cells wrote Tack's
  values straight into VTK's point data: an order-2 field's in Tack's node order,
  not VTK's (`Elevation` came out scrambled, and its `Gradients` wrong), and values
  on Tack's points, the corners, for every VTK point, edge middles included (a
  1193-point grid got an array of 1193 tuples of 1227 components). Each VTK point
  now takes its node's value: the field put into the space the import made
  (`project`, exact for an `H1` field of that order or below and for values on the
  corners), its nodes matched to VTK's points as the import matched them. An
  integer field on the points is refused there, having no value at an edge's
  middle. Found running VTK's `quadraticQuadDeriv.py` on Tack.

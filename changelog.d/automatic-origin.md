### Changed

- `vtk_to_dataset` converting to `f32` without an `origin` chooses one where
  the data are more than 100 times farther from zero than they are wide: the
  lower corner of VTK's bounds, rounded down, the same for a composite's
  pieces. Far points kept 0.2 m steps on a survey's coordinates; they keep
  their precision. `point_cloud` chooses its origin for `f64` positions by the
  same rule (`automatic_origin`): a small cloud about zero got (-1, -1, -1),
  coarsening its values.

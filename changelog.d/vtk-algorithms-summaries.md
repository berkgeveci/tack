### Added

- `Statistics`, `Entropy` and `Integrate`, VTK algorithms of `td.statistics`,
  `td.entropy` and `td.integrate` that reduce all their input (every dataset of a
  partitioned dataset or a tree, on every MPI rank) to one answer, as VTK's
  `vtkDescriptiveStatistics` and `vtkIntegrateAttributes` do. `Statistics` and
  `Entropy` give a `vtkTable`, a row per field and a vector field's components one
  by one; `Integrate` gives `vtkIntegrateAttributes`' one vertex at the centroid,
  each field's integral on it and the measure as `Length`, `Area` or `Volume`.

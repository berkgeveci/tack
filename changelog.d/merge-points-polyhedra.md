### Added

- `merge_points` takes polyhedral, polygonal and polyline topologies. A
  polyhedral mesh's faces are matched again on the merged points, so the copies
  of a face that pieces each stored once, with one cell, become one face with
  both: OpenFOAM's processor pieces put together without merging (an AhmedML
  volume: 1,561,917 copied points of 24,358,265, 1,494,132 faces stored twice)
  merge in 13 s into the mesh their solver ran on, whose boundary is the walls
  and the far field (40,700 faces not at rest, not 3,028,964) and whose body is
  one piece, not 116. Copies wound the same way, or a face with two cells on a
  side, are refused.

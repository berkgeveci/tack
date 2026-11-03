### Added

- `extract_surface` of a `PartitionedDataSet` gives the whole's surface, a
  piece each: pieces without ghost cells take the faces between them for
  boundary, so the faces of every piece's surface are matched over the pieces
  by their points and those two pieces hold are left out; the surface's points
  are owned once. On pieces with ghost cells it is each piece's own surface.

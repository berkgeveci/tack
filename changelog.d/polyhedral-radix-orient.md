### Added

- `vtk_to_dataset` finds tetrahedra, hexahedra, wedges and pyramids that are
  inside out (corners in the reverse of VTK's order, as some converters write
  prisms: negative volumes, inward faces) on every read -- a kernel screens
  every cell, the volume decides -- and warns, naming `orient=True`, which now
  turns them outward too. A tangled cell, negative at some corners but not
  inside out, is left. A CFD mesh's 66,606 inside-out wedges took 0.3 s to find
  among 21M cells.

### Fixed

- Reading as polyhedra (`polyhedral=True`) is about four times faster again:
  triangles and quads are matched by their sorted points packed into two
  64-bit keys and Tack's parallel radix sort, where `np.lexsort` took 68 s on
  128M quads, and their face lists are built by columns, a few million cells at
  a time, with 32-bit point ids where they fit. 21M hexahedra, 67K wedges and
  508K polyhedra: 27 s and 19 GB, with `orient=True` for its wedges, which also
  no longer falls back to the cell-by-cell walk when the shaped cells were what
  disagreed.

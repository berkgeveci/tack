### Changed

- Advection tries the polyhedron a particle was last found in before the
  bins, as it tries solid cells with a reference element: taken where one of
  its sub-tetrahedra holds the point well inside, so the paths are the same
  bit for bit. On the AhmedML mesh's 21.5M polyhedra, 2,000 streamlines take
  0.66 s on the CPU (0.76 before) and 3.0 s on Metal (6.0).

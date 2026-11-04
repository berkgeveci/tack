### Fixed

- A particle whose step leaves the mesh is pushed just outside it (an Euler
  step from the last point inside by the bisection's bracket, grown until no
  cell holds it), not by the distance to the boundary again, which Viskores'
  `SmallStep` does and Tack did, counting the distance once in the particle's
  time and path length. On one mesh it lengthened only a path's last segment;
  between pieces without ghost cells the path went on from there: on the Ford
  DrivAer's split pieces 59 of 200 streamlines ran past `max_length` (by up to
  0.31 m over 23 hand-offs) and one was pushed out of the mesh across a thin
  layer, 0.48 m into its 13.84.
- RK45's end at a wall, a push out that moves the particle less than
  `min_step`, measured the push after it was taken (always 0), so every RK45
  push out that stayed in the mesh ended the particle; it measures the move.
- Over pieces without ghost cells, a particle pushed out of one piece into a
  sliver no piece holds -- pieces whose shared faces are not one surface, a
  hexahedron's bilinear face against a polyhedron's fanned one in a
  `polyhedral="split"` reading -- goes on from where the push would have ended
  going the way to the boundary again, if a piece holds that, with the move in
  its time, length and stretch; it stopped there, outside.

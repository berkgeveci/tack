### Fixed

- Advection stopped a particle at zero velocity wherever its speed was below
  3.5e-4 in the data's unit (a squared speed at the float's machine epsilon),
  so every streamline of the SciVis 2021 mantle (1e-9 m/s) ended at its seed.
  Zero velocity is now a squared speed at or below 1e-37; slower flows stop by
  `terminal_speed`, as vtkStreamTracer's do. The mantle's 500 streamlines are
  VTK's (300,500 points, lengths within 1.3e-5).
- A step leaving the mesh is bisected to the boundary, and the bisection's
  bracket was absolute (ten machine epsilons): on long steps in f32 (an ulp of
  50 is 3.8e-6) it never got that small, and the particle bisected forever.
  The bisection is now bounded at 64 halvings.

### Added

- `warp` by a displacement of higher order than the geometry (a finite-element
  solution, as MFEM's examples displace their meshes) lifts the geometry into
  its space, so the output is curved; a 2-vector displacement (a 2D problem's)
  moves x and y.
- The MFEM interop reads meshes whose nodes are in MFEM's older collections
  ("Linear", "Quadratic", "QuadraticPos", "Cubic", as `escher-p3.mesh`), and
  periodic meshes, whose nodes MFEM keeps per element, with an `L2` geometry.

### Fixed

- `jump` of a field of order 2 or more took each side's value from the face's
  corner values (`traces`) and their linear mean; it evaluates each side's own
  basis at the face's center (DG jumps against MFEM's to 1e-16).
- `mfem_to_dataset` of a 2D curved mesh refused its two-component nodes; they
  get z = 0.
- An `L2` geometry above order 1 (a mesh displaced by a DG field, a periodic
  mesh of order 2) was read as order-1 corners, each cell's first values, so
  probes, centres and everything else were wrong. On tensor shapes it is now a
  curved geometry (`views._L2TensorGeometry`, either layout); on simplices and
  wedges it is refused.
- `subdivide` (and the cuts that subdivide) of an `L2` geometry shared lattice
  points through the topology, so on a periodic mesh, whose seam's cells share
  points but sit apart, sub-cells stretched across the mesh; each cell's lattice
  points are now its own.

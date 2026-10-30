### Added

- A field's nodes on a curved simplex or wedge geometry of another order (an
  order-4 field on quadratic tetrahedra, VTK's quadratic cells among them) are
  placed where the geometry puts them: on edges and quad faces its trace there,
  as on hexahedra since #217, and on triangle faces its trace in the face's
  canonical frame (its corners by position) evaluated in the geometry's
  Lagrange or Bernstein basis (`views._CurvedTriangleNodePosition`), so every
  piece holding a face gets the same bits. They were refused.

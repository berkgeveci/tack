"""Earlier visualization code, kept for reference and superseded by ``tack.dataset``.

- ``tack.deprecated.algorithms``: the first vis worklets -- flying edges on
  uniform grids, triangle normals, cell-to-point averaging on hexahedra, AMR
  blanking -- each on its own array conventions.
- ``tack.deprecated.fe``: Lagrange bases, geometry maps and DOF accessors for
  quads and hexahedra.

``tack.dataset`` does all of this on one dataset model: ``contour`` on any
topology (a uniform grid's included), ``to_points``, the spaces and their
views. Nothing new is added here; code here may go in a later release.
"""

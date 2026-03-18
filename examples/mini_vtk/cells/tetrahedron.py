"""Tetrahedron cell type — barycentric interpolation."""

import pgc


@pgc.data_oriented
class Tetrahedron:
    """4-node tetrahedron with linear (barycentric) shape functions.

    Parametric space: (r, s, t) where r,s,t >= 0 and r+s+t <= 1.
    Center at (0.25, 0.25, 0.25).

    Vertex ordering:
      0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (0,0,1)
    """

    def __init__(self):
        self.num_points = 4

    @pgc.func
    def center(self, dim):
        return 0.25

    @pgc.func
    def weight(self, vertex, r, s, t):
        """Barycentric shape function for a given vertex at (r, s, t)."""
        # N0 = 1 - r - s - t, N1 = r, N2 = s, N3 = t
        w = 1.0 - r - s - t
        if vertex == 1:
            w = r
        if vertex == 2:
            w = s
        if vertex == 3:
            w = t
        return w

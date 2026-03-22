"""Hexahedron cell type -- trilinear interpolation."""

import pgc


@pgc.data_oriented
class Hexahedron:
    """8-node hexahedron with trilinear shape functions.

    Parametric space: (r, s, t) in [0, 1]^3.
    Center at (0.5, 0.5, 0.5).

    Vertex ordering:
      0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)
      4: (0,0,1)  5: (1,0,1)  6: (0,1,1)  7: (1,1,1)
    """

    def __init__(self):
        self.num_points = 8

    @pgc.func
    def center(self, dim):
        return 0.5

    @pgc.func
    def weight(self, vertex, r, s, t):
        """Trilinear shape function for a given vertex at (r, s, t)."""
        # N_v = (1-r or r) * (1-s or s) * (1-t or t)
        # Bit decomposition: vertex & 1 -> r, vertex & 2 -> s, vertex & 4 -> t
        wr = r
        if vertex % 2 == 0:
            wr = 1.0 - r
        ws = s
        if (vertex // 2) % 2 == 0:
            ws = 1.0 - s
        wt = t
        if vertex // 4 == 0:
            wt = 1.0 - t
        return wr * ws * wt

"""Wedge (pentahedron) cell type -- prismatic interpolation."""

import tack


@tack.data_oriented
class Wedge:
    """6-node wedge with linear-triangular x linear shape functions.

    Parametric space: (r, s) are barycentric in the triangle face,
    t in [0, 1] along the prism axis.
    Center at (1/3, 1/3, 0.5).

    Vertex ordering:
      Bottom triangle: 0:(0,0,0) 1:(1,0,0) 2:(0,1,0)
      Top triangle:    3:(0,0,1) 4:(1,0,1) 5:(0,1,1)
    """

    def __init__(self):
        self.num_points = 6

    @tack.func
    def center(self, dim):
        result = 0.5
        if dim == 0:
            result = 0.3333333333
        if dim == 1:
            result = 0.3333333333
        return result

    @tack.func
    def weight(self, vertex, r, s, t):
        """Shape function for a given vertex at (r, s, t).

        Bottom face (t=0): vertices 0,1,2 with triangular weights * (1-t)
        Top face    (t=1): vertices 3,4,5 with triangular weights * t
        """
        # Triangular weight: w0 = 1-r-s, w1 = r, w2 = s
        tri = 1.0 - r - s
        if vertex == 1:
            tri = r
        if vertex == 2:
            tri = s
        if vertex == 4:
            tri = r
        if vertex == 5:
            tri = s
        if vertex == 3:
            tri = 1.0 - r - s

        # Axial weight: bottom (0,1,2) use (1-t), top (3,4,5) use t
        ax = 1.0 - t
        if vertex >= 3:
            ax = t

        return tri * ax

"""Structured (rectilinear) cell set — connectivity from grid dimensions."""

import pgc


@pgc.data_oriented
class CellSetStructured3D:
    """Structured hex mesh: zero connectivity storage.

    Grid of nx*ny*nz cells with (nx+1)*(ny+1)*(nz+1) points.
    Point ordering is x-fastest: point(i,j,k) = k*(nx+1)*(ny+1) + j*(nx+1) + i.
    """

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_plus1 = nx + 1
        self.nxy = nx * ny
        self.nxy_plus1 = (nx + 1) * (ny + 1)

    @pgc.func
    def get_cell_points(self, cell_id):
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy

        base = ck * self.nxy_plus1 + cj * self.nx_plus1 + ci

        p0 = base
        p1 = base + 1
        p2 = base + self.nx_plus1
        p3 = base + self.nx_plus1 + 1
        p4 = base + self.nxy_plus1
        p5 = base + self.nxy_plus1 + 1
        p6 = base + self.nxy_plus1 + self.nx_plus1
        p7 = base + self.nxy_plus1 + self.nx_plus1 + 1
        return p0, p1, p2, p3, p4, p5, p6, p7

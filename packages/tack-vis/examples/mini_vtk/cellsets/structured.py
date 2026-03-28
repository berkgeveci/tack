"""Structured (rectilinear) cell set -- connectivity from grid dimensions."""

import tack


@tack.data_oriented
class CellSetStructured3D:
    """Structured hex mesh: zero connectivity storage.

    Grid of nx*ny*nz cells with (nx+1)*(ny+1)*(nz+1) points.
    Point ordering is x-fastest: point(i,j,k) = k*(nx+1)*(ny+1) + j*(nx+1) + i.

    Uses the generic cell set interface:
      - points_per_cell: compile-time constant (8 for hex)
      - get_point_id(cell_id, local_idx): returns the global point index
    """

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nx_plus1 = nx + 1
        self.nxy = nx * ny
        self.nxy_plus1 = (nx + 1) * (ny + 1)
        self.points_per_cell = 8

    @tack.func
    def get_point_id(self, cell_id, local_idx):
        ci = cell_id % self.nx
        cj = (cell_id // self.nx) % self.ny
        ck = cell_id // self.nxy
        base = ck * self.nxy_plus1 + cj * self.nx_plus1 + ci

        # Local vertex ordering for hex:
        #   0: (i,   j,   k  )    4: (i,   j,   k+1)
        #   1: (i+1, j,   k  )    5: (i+1, j,   k+1)
        #   2: (i,   j+1, k  )    6: (i,   j+1, k+1)
        #   3: (i+1, j+1, k  )    7: (i+1, j+1, k+1)
        result = base
        if local_idx == 1:
            result = base + 1
        if local_idx == 2:
            result = base + self.nx_plus1
        if local_idx == 3:
            result = base + self.nx_plus1 + 1
        if local_idx == 4:
            result = base + self.nxy_plus1
        if local_idx == 5:
            result = base + self.nxy_plus1 + 1
        if local_idx == 6:
            result = base + self.nxy_plus1 + self.nx_plus1
        if local_idx == 7:
            result = base + self.nxy_plus1 + self.nx_plus1 + 1
        return result

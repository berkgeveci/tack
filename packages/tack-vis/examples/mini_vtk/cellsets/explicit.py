"""Explicit cell set -- connectivity stored in a flat field."""

import tack


@tack.data_oriented
class CellSetExplicit:
    """Explicit cell set with fixed points-per-cell.

    Connectivity is a flat i32 field: [c0p0, ..., c0pN, c1p0, ..., c1pN, ...]
    where N = points_per_cell.

    Works for any cell type (hex=8, tet=4, wedge=6, etc.) -- the same
    kernel code compiles differently for each points_per_cell value via
    the template system.

    Uses the generic cell set interface:
      - points_per_cell: compile-time constant
      - get_point_id(cell_id, local_idx): returns the global point index
    """

    def __init__(self, connectivity, points_per_cell):
        self.connectivity = connectivity
        self.points_per_cell = points_per_cell

    @tack.func
    def get_point_id(self, cell_id, local_idx):
        return self.connectivity[cell_id * self.points_per_cell + local_idx]


def from_structured(cell_set_structured, n_cells):
    """Build explicit connectivity from a structured cell set."""
    ppc = cell_set_structured.points_per_cell

    @tack.kernel
    def _expand(struct: tack.template(), conn, n):
        for c in range(n):
            for v in range(struct.points_per_cell):
                conn[c * struct.points_per_cell + v] = struct.get_point_id(c, v)

    conn = tack.field(dtype=tack.i32, shape=(n_cells * ppc,))
    _expand(cell_set_structured, conn, n_cells)
    return CellSetExplicit(conn, ppc)

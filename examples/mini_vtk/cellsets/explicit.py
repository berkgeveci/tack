"""Explicit cell set — connectivity stored in a flat field."""

import numpy as np
import pgc


@pgc.data_oriented
class CellSetExplicitHex:
    """Explicit hex mesh: 8 point IDs per cell in a flat i32 field.

    Memory: [c0p0, c0p1, ..., c0p7, c1p0, ..., c1p7, ...]
    """

    def __init__(self, connectivity):
        self.connectivity = connectivity

    @pgc.func
    def get_cell_points(self, cell_id):
        base = cell_id * 8
        p0 = self.connectivity[base]
        p1 = self.connectivity[base + 1]
        p2 = self.connectivity[base + 2]
        p3 = self.connectivity[base + 3]
        p4 = self.connectivity[base + 4]
        p5 = self.connectivity[base + 5]
        p6 = self.connectivity[base + 6]
        p7 = self.connectivity[base + 7]
        return p0, p1, p2, p3, p4, p5, p6, p7


def from_structured(cell_set_structured, n_cells):
    """Build explicit connectivity from a structured cell set.

    Useful for benchmarking explicit vs structured performance.
    """
    @pgc.kernel
    def _expand(struct: pgc.template(), conn, n):
        for c in range(n):
            p0, p1, p2, p3, p4, p5, p6, p7 = struct.get_cell_points(c)
            base = c * 8
            conn[base] = p0
            conn[base + 1] = p1
            conn[base + 2] = p2
            conn[base + 3] = p3
            conn[base + 4] = p4
            conn[base + 5] = p5
            conn[base + 6] = p6
            conn[base + 7] = p7

    conn = pgc.field(dtype=pgc.i32, shape=(n_cells * 8,))
    _expand(cell_set_structured, conn, n_cells)
    return CellSetExplicitHex(conn)

"""GeometryMap — physical ↔ parametric coordinate mapping.

Linear geometry maps (order 1) use direct vertex interpolation.
High-order geometry maps compose a FieldAccessor + ElementBasis
on the mesh coordinate DOFs (Mesh::Nodes in MFEM).
"""

import numpy as np

import tack

# ── Linear (order 1) maps ──────────────────────────────────────────

@tack.data_oriented
class LinearQuadMap:
    """Bilinear mapping for order-1 quadrilateral elements.

    Uses 4 vertex coordinates directly — no basis evaluation needed.

    Vertex ordering (matches MFEM):
      0: (0,0)  1: (1,0)  2: (1,1)  3: (0,1)
    """

    def __init__(self, vx, vy, connectivity, conn_offsets):
        """Create from Tack fields.

        Args:
            vx, vy: tack.field of vertex coordinates.
            connectivity: tack.field(i32) of element-to-vertex indices.
            conn_offsets: tack.field(i32) of per-element offsets.
        """
        self.vx = vx
        self.vy = vy
        self.connectivity = connectivity
        self.conn_offsets = conn_offsets

    @tack.func
    def physical_x(self, elem, r, s):
        c = self.conn_offsets[elem]
        w0 = (1.0 - r) * (1.0 - s)
        w1 = r * (1.0 - s)
        w2 = r * s
        w3 = (1.0 - r) * s
        return (w0 * self.vx[self.connectivity[c]]
                + w1 * self.vx[self.connectivity[c + 1]]
                + w2 * self.vx[self.connectivity[c + 2]]
                + w3 * self.vx[self.connectivity[c + 3]])

    @tack.func
    def physical_y(self, elem, r, s):
        c = self.conn_offsets[elem]
        w0 = (1.0 - r) * (1.0 - s)
        w1 = r * (1.0 - s)
        w2 = r * s
        w3 = (1.0 - r) * s
        return (w0 * self.vy[self.connectivity[c]]
                + w1 * self.vy[self.connectivity[c + 1]]
                + w2 * self.vy[self.connectivity[c + 2]]
                + w3 * self.vy[self.connectivity[c + 3]])

    @tack.func
    def jacobian_component(self, elem, r, s, row, col):
        """Jacobian J[row][col] where row=0→x,1→y and col=0→dr,1→ds."""
        c = self.conn_offsets[elem]
        if row == 0:
            p0 = self.vx[self.connectivity[c]]
            p1 = self.vx[self.connectivity[c + 1]]
            p2 = self.vx[self.connectivity[c + 2]]
            p3 = self.vx[self.connectivity[c + 3]]
        else:
            p0 = self.vy[self.connectivity[c]]
            p1 = self.vy[self.connectivity[c + 1]]
            p2 = self.vy[self.connectivity[c + 2]]
            p3 = self.vy[self.connectivity[c + 3]]
        if col == 0:
            # d/dr
            return -(1.0 - s) * p0 + (1.0 - s) * p1 + s * p2 - s * p3
        # d/ds
        return -(1.0 - r) * p0 - r * p1 + r * p2 + (1.0 - r) * p3


@tack.data_oriented
class LinearHexMap:
    """Trilinear mapping for order-1 hexahedral elements.

    Vertex ordering (matches MFEM/MC convention):
      0: (0,0,0)  1: (1,0,0)  2: (1,1,0)  3: (0,1,0)
      4: (0,0,1)  5: (1,0,1)  6: (1,1,1)  7: (0,1,1)
    """

    def __init__(self, vx, vy, vz, connectivity, conn_offsets):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.connectivity = connectivity
        self.conn_offsets = conn_offsets

    @tack.func
    def _trilinear(self, elem, r, s, t, coords):
        c = self.conn_offsets[elem]
        w0 = (1-r)*(1-s)*(1-t)
        w1 = r*(1-s)*(1-t)
        w2 = r*s*(1-t)
        w3 = (1-r)*s*(1-t)
        w4 = (1-r)*(1-s)*t
        w5 = r*(1-s)*t
        w6 = r*s*t
        w7 = (1-r)*s*t
        return (w0 * coords[self.connectivity[c]]
                + w1 * coords[self.connectivity[c + 1]]
                + w2 * coords[self.connectivity[c + 2]]
                + w3 * coords[self.connectivity[c + 3]]
                + w4 * coords[self.connectivity[c + 4]]
                + w5 * coords[self.connectivity[c + 5]]
                + w6 * coords[self.connectivity[c + 6]]
                + w7 * coords[self.connectivity[c + 7]])

    @tack.func
    def physical_x(self, elem, r, s, t):
        return self._trilinear(elem, r, s, t, self.vx)

    @tack.func
    def physical_y(self, elem, r, s, t):
        return self._trilinear(elem, r, s, t, self.vy)

    @tack.func
    def physical_z(self, elem, r, s, t):
        return self._trilinear(elem, r, s, t, self.vz)


# ── Construction helpers ────────────────────────────────────────────

def linear_quad_map_from_numpy(vx_np, vy_np, conn_np, conn_off_np,
                               np_fp=np.float64):
    """Create LinearQuadMap from numpy arrays."""
    return LinearQuadMap(
        vx=tack.field_like(vx_np.astype(np_fp)),
        vy=tack.field_like(vy_np.astype(np_fp)),
        connectivity=tack.field_like(conn_np.astype(np.int32)),
        conn_offsets=tack.field_like(conn_off_np.astype(np.int32)))


def linear_hex_map_from_numpy(vx_np, vy_np, vz_np, conn_np, conn_off_np,
                              np_fp=np.float64):
    """Create LinearHexMap from numpy arrays."""
    return LinearHexMap(
        vx=tack.field_like(vx_np.astype(np_fp)),
        vy=tack.field_like(vy_np.astype(np_fp)),
        vz=tack.field_like(vz_np.astype(np_fp)),
        connectivity=tack.field_like(conn_np.astype(np.int32)),
        conn_offsets=tack.field_like(conn_off_np.astype(np.int32)))

"""ElementBasis — evaluate FE basis functions at parametric coordinates.

Each basis class is @tack.data_oriented and provides:
- evaluate(j, r, s, ...): value of basis function j
- evaluate_dr(j, r, s, ...): derivative w.r.t. r
- evaluate_ds(j, r, s, ...): derivative w.r.t. s
- evaluate_dt(j, r, s, t): derivative w.r.t. t (3D only)

Tensor-product bases (QuadBasis, HexBasis) factor into 1D Lagrange
products for efficiency.
"""

import numpy as np

import tack

# ── Host-side helpers ───────────────────────────────────────────────

def _lagrange_1d_host(nodes, i, t):
    """Host-side 1D Lagrange evaluation (for precomputing B matrices)."""
    n = len(nodes)
    val = 1.0
    for j in range(n):
        if j != i:
            val *= (t - nodes[j]) / (nodes[i] - nodes[j])
    return val


def extract_gl_nodes_1d(ref_rx, ref_ry, ndof):
    """Extract sorted unique 1D GL nodes from 2D ref positions.

    Returns (gl_1d, tp_permutation) where tp_permutation[mfem_dof]
    gives the tensor-product index.
    """
    gl_1d = np.sort(np.unique(np.round(ref_rx, 10)))
    n1d = len(gl_1d)
    assert n1d * n1d == ndof, (
        f"Expected {n1d}^2={n1d*n1d} DOFs, got {ndof}")

    tp_perm = np.zeros(ndof, dtype=np.int32)
    for k in range(ndof):
        i = np.argmin(np.abs(gl_1d - ref_rx[k]))
        j = np.argmin(np.abs(gl_1d - ref_ry[k]))
        tp_perm[k] = j * n1d + i
    return gl_1d, tp_perm


def extract_gl_nodes_1d_3d(ref_rx, ref_ry, ref_rz, ndof):
    """Extract 1D GL nodes from 3D ref positions (hex elements)."""
    gl_1d = np.sort(np.unique(np.round(ref_rx, 10)))
    n1d = len(gl_1d)
    assert n1d ** 3 == ndof, (
        f"Expected {n1d}^3={n1d**3} DOFs, got {ndof}")

    tp_perm = np.zeros(ndof, dtype=np.int32)
    for k in range(ndof):
        i = np.argmin(np.abs(gl_1d - ref_rx[k]))
        j = np.argmin(np.abs(gl_1d - ref_ry[k]))
        kk = np.argmin(np.abs(gl_1d - ref_rz[k]))
        tp_perm[k] = kk * n1d * n1d + j * n1d + i
    return gl_1d, tp_perm


def precompute_basis_matrix_2d(basis, n_sub):
    """Precompute basis values at (n_sub+1)^2 subdivision points.

    Returns B of shape ((n_sub+1)^2, ndof).
    """
    n1d = basis.n1d
    ndof = n1d * n1d
    nrow = n_sub + 1
    n_pts = nrow * nrow
    gl = basis._gl_np

    B = np.zeros((n_pts, ndof))
    for sj in range(nrow):
        s = sj / n_sub
        ls = np.array([_lagrange_1d_host(gl, j, s) for j in range(n1d)])
        for si in range(nrow):
            r = si / n_sub
            lr = np.array([_lagrange_1d_host(gl, i, r)
                           for i in range(n1d)])
            pt = sj * nrow + si
            for j in range(n1d):
                for i in range(n1d):
                    B[pt, j * n1d + i] = lr[i] * ls[j]
    return B


def precompute_basis_matrix_3d(basis, n_sub):
    """Precompute basis values at (n_sub+1)^3 subdivision points (hex)."""
    n1d = basis.n1d
    ndof = n1d ** 3
    nrow = n_sub + 1
    n_pts = nrow ** 3
    gl = basis._gl_np

    B = np.zeros((n_pts, ndof))
    for sk in range(nrow):
        t = sk / n_sub
        lt = np.array([_lagrange_1d_host(gl, k, t)
                       for k in range(n1d)])
        for sj in range(nrow):
            s = sj / n_sub
            ls = np.array([_lagrange_1d_host(gl, j, s)
                           for j in range(n1d)])
            for si in range(nrow):
                r = si / n_sub
                lr = np.array([_lagrange_1d_host(gl, i, r)
                               for i in range(n1d)])
                pt = sk * nrow * nrow + sj * nrow + si
                for k in range(n1d):
                    for j in range(n1d):
                        for i in range(n1d):
                            dof = k * n1d * n1d + j * n1d + i
                            B[pt, dof] = lr[i] * ls[j] * lt[k]
    return B


# ── Quad basis ──────────────────────────────────────────────────────

@tack.data_oriented
class QuadBasis:
    """Tensor-product Lagrange basis on [0,1]^2.

    DOFs are in tensor-product order (row-major):
    dof j = (j // n1d, j % n1d) in (s-row, r-col) indices.

    If MFEM uses a different DOF ordering, the FieldAccessor should
    reorder DOFs to tensor-product order.
    """

    def __init__(self, gl_nodes_1d_np, np_fp=np.float64):
        self._gl_np = gl_nodes_1d_np.copy()
        self.gl_nodes_1d = tack.field_like(
            gl_nodes_1d_np.astype(np_fp))
        self.n1d = len(gl_nodes_1d_np)

    @tack.func
    def _l1d(self, i, t):
        """1D Lagrange basis function i at t."""
        val = 1.0
        for j in range(self.n1d):
            if j != i:
                val *= ((t - self.gl_nodes_1d[j])
                        / (self.gl_nodes_1d[i] - self.gl_nodes_1d[j]))
        return val

    @tack.func
    def _l1d_deriv(self, i, t):
        """Derivative of 1D Lagrange basis function i at t."""
        result = 0.0
        for k in range(self.n1d):
            if k != i:
                term = 1.0 / (self.gl_nodes_1d[i] - self.gl_nodes_1d[k])
                for j in range(self.n1d):
                    if j != i and j != k:
                        term *= ((t - self.gl_nodes_1d[j])
                                 / (self.gl_nodes_1d[i]
                                    - self.gl_nodes_1d[j]))
                result += term
        return result

    @tack.func
    def ndof(self):
        return self.n1d * self.n1d

    @tack.func
    def evaluate(self, j, r, s):
        """Value of basis function j at (r, s)."""
        ji = j % self.n1d
        jj = j // self.n1d
        return self._l1d(ji, r) * self._l1d(jj, s)

    @tack.func
    def evaluate_dr(self, j, r, s):
        """d(phi_j)/dr at (r, s)."""
        ji = j % self.n1d
        jj = j // self.n1d
        return self._l1d_deriv(ji, r) * self._l1d(jj, s)

    @tack.func
    def evaluate_ds(self, j, r, s):
        """d(phi_j)/ds at (r, s)."""
        ji = j % self.n1d
        jj = j // self.n1d
        return self._l1d(ji, r) * self._l1d_deriv(jj, s)


# ── Hex basis ───────────────────────────────────────────────────────

@tack.data_oriented
class HexBasis:
    """Tensor-product Lagrange basis on [0,1]^3.

    DOFs in tensor-product order: dof = k*n1d^2 + j*n1d + i.
    """

    def __init__(self, gl_nodes_1d_np, np_fp=np.float64):
        self._gl_np = gl_nodes_1d_np.copy()
        self.gl_nodes_1d = tack.field_like(
            gl_nodes_1d_np.astype(np_fp))
        self.n1d = len(gl_nodes_1d_np)

    @tack.func
    def _l1d(self, i, t):
        val = 1.0
        for j in range(self.n1d):
            if j != i:
                val *= ((t - self.gl_nodes_1d[j])
                        / (self.gl_nodes_1d[i] - self.gl_nodes_1d[j]))
        return val

    @tack.func
    def _l1d_deriv(self, i, t):
        result = 0.0
        for k in range(self.n1d):
            if k != i:
                term = 1.0 / (self.gl_nodes_1d[i] - self.gl_nodes_1d[k])
                for j in range(self.n1d):
                    if j != i and j != k:
                        term *= ((t - self.gl_nodes_1d[j])
                                 / (self.gl_nodes_1d[i]
                                    - self.gl_nodes_1d[j]))
                result += term
        return result

    @tack.func
    def ndof(self):
        return self.n1d * self.n1d * self.n1d

    @tack.func
    def evaluate(self, j, r, s, t):
        ji = j % self.n1d
        jj = (j // self.n1d) % self.n1d
        jk = j // (self.n1d * self.n1d)
        return self._l1d(ji, r) * self._l1d(jj, s) * self._l1d(jk, t)

    @tack.func
    def evaluate_dr(self, j, r, s, t):
        ji = j % self.n1d
        jj = (j // self.n1d) % self.n1d
        jk = j // (self.n1d * self.n1d)
        return (self._l1d_deriv(ji, r)
                * self._l1d(jj, s)
                * self._l1d(jk, t))

    @tack.func
    def evaluate_ds(self, j, r, s, t):
        ji = j % self.n1d
        jj = (j // self.n1d) % self.n1d
        jk = j // (self.n1d * self.n1d)
        return (self._l1d(ji, r)
                * self._l1d_deriv(jj, s)
                * self._l1d(jk, t))

    @tack.func
    def evaluate_dt(self, j, r, s, t):
        ji = j % self.n1d
        jj = (j // self.n1d) % self.n1d
        jk = j // (self.n1d * self.n1d)
        return (self._l1d(ji, r)
                * self._l1d(jj, s)
                * self._l1d_deriv(jk, t))


# Public aliases for standalone use
def lagrange_1d(nodes, i, t, n):
    """Host-side 1D Lagrange evaluation."""
    return _lagrange_1d_host(nodes, i, t)


def lagrange_1d_deriv(nodes, i, t, n):
    """Host-side 1D Lagrange derivative."""
    nd = len(nodes) if hasattr(nodes, '__len__') else n
    result = 0.0
    for k in range(nd):
        if k != i:
            term = 1.0 / (nodes[i] - nodes[k])
            for j in range(nd):
                if j != i and j != k:
                    term *= (t - nodes[j]) / (nodes[i] - nodes[j])
            result += term
    return result

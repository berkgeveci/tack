"""Tests for tack.fe — finite element basis, accessor, geometry."""

import numpy as np
import tack
from tack.fe.basis import (
    QuadBasis, HexBasis,
    extract_gl_nodes_1d, extract_gl_nodes_1d_3d,
    precompute_basis_matrix_2d,
)
from tack.fe.accessor import ContiguousDofs, contiguous_from_numpy
from tack.fe.geometry import LinearQuadMap, linear_quad_map_from_numpy

import pytest

# FE tests use f64 — run on backends that support it
_f64_backends = []
for _arch in ["cpu", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        from tack.runtime.dispatch import get_backend as _get_backend
        _be = _get_backend()
        if getattr(_be, 'supports_f64', True):
            _f64_backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass

np_fp = np.float64


@pytest.fixture(autouse=True, params=_f64_backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


def test_lagrange_1d_cardinal():
    """Basis functions should be 1 at their own node, 0 at others."""
    gl = np.array([0.0, 0.5, 1.0])
    basis = QuadBasis(gl, np_fp=np_fp)

    @tack.kernel
    def check_cardinal(b: tack.template(), out):
        # Evaluate basis function i at node j → should be delta_ij
        n = b.n1d
        for i in range(n):
            for j in range(n):
                r = b.gl_nodes_1d[j]
                val = tack.fe.basis.lagrange_1d(b.gl_nodes_1d, i, r, n)
                out[i * n + j] = val

    # Can't call tack.func from kernel with module path.
    # Instead, test via the QuadBasis.evaluate at corner nodes.
    @tack.kernel
    def check_quad_cardinal(b: tack.template(), out, ndof):
        for dof in range(ndof):
            n = b.n1d
            i = dof % n
            j = dof // n
            r = b.gl_nodes_1d[i]
            s = b.gl_nodes_1d[j]
            for k in range(ndof):
                val = b.evaluate(k, r, s)
                out[dof * ndof + k] = val

    n1d = 3
    ndof = n1d * n1d
    out = tack.field(dtype=tack.f32 if np_fp == np.float32 else tack.f64,
                    shape=(ndof * ndof,))
    check_quad_cardinal(basis, out, ndof)
    result = out.to_numpy().reshape(ndof, ndof)

    # Should be identity matrix
    for i in range(ndof):
        for j in range(ndof):
            expected = 1.0 if i == j else 0.0
            assert abs(result[i, j] - expected) < 1e-5, (
                f"Cardinal property failed: B[{i},{j}] = {result[i,j]}, "
                f"expected {expected}")
    print("  PASS: quad basis cardinal property")


def test_quad_partition_of_unity():
    """Sum of all basis functions at any point should be 1."""
    gl = np.array([0.0, 0.5, 1.0])
    basis = QuadBasis(gl, np_fp=np_fp)
    ndof = 9

    @tack.kernel
    def check_partition(b: tack.template(), out, ndof):
        for idx in range(25):
            r = float(idx % 5) * 0.25
            s = float(idx // 5) * 0.25
            total = 0.0
            for j in range(ndof):
                total += b.evaluate(j, r, s)
            out[idx] = total

    out = tack.field(dtype=tack.f32 if np_fp == np.float32 else tack.f64,
                    shape=(25,))
    check_partition(basis, out, ndof)
    result = out.to_numpy()

    for i in range(25):
        assert abs(result[i] - 1.0) < 1e-5, (
            f"Partition of unity failed at point {i}: sum = {result[i]}")
    print("  PASS: quad basis partition of unity")


def test_contiguous_accessor():
    """ContiguousDofs should read correct DOF values."""
    # 3 elements, 4 DOFs each
    dofs = np.array([10, 20, 30, 40,
                     50, 60, 70, 80,
                     90, 100, 110, 120], dtype=np.float64)
    offsets = np.array([0, 4, 8, 12], dtype=np.int32)
    accessor = contiguous_from_numpy(dofs, offsets, np_fp=np_fp)

    @tack.kernel
    def read_dofs(acc: tack.template(), out):
        for i in range(1):
            out[0] = acc.get_dof(1, 2)
            out[1] = acc.get_dof(2, 0)

    fp = tack.f32 if np_fp == np.float32 else tack.f64
    out = tack.field(dtype=fp, shape=(2,))
    read_dofs(accessor, out)
    result = out.to_numpy()

    assert abs(result[0] - 70.0) < 1e-5
    assert abs(result[1] - 90.0) < 1e-5
    print("  PASS: contiguous DOF accessor")


def test_linear_quad_map():
    """LinearQuadMap should interpolate vertex coordinates."""
    # Unit square element
    vx = np.array([0.0, 2.0, 2.0, 0.0])
    vy = np.array([0.0, 0.0, 1.0, 1.0])
    conn = np.array([0, 1, 2, 3], dtype=np.int32)
    conn_off = np.array([0, 4], dtype=np.int32)
    gmap = linear_quad_map_from_numpy(vx, vy, conn, conn_off, np_fp=np_fp)

    @tack.kernel
    def eval_map(g: tack.template(), out):
        for i in range(1):
            out[0] = g.physical_x(0, 0.5, 0.5)
            out[1] = g.physical_y(0, 0.5, 0.5)
            out[2] = g.physical_x(0, 0.0, 0.0)
            out[3] = g.physical_y(0, 0.0, 0.0)
            out[4] = g.physical_x(0, 1.0, 1.0)
            out[5] = g.physical_y(0, 1.0, 1.0)

    fp = tack.f32 if np_fp == np.float32 else tack.f64
    out = tack.field(dtype=fp, shape=(6,))
    eval_map(gmap, out)
    result = out.to_numpy()

    assert abs(result[0] - 1.0) < 1e-5
    assert abs(result[1] - 0.5) < 1e-5
    assert abs(result[2] - 0.0) < 1e-5
    assert abs(result[3] - 0.0) < 1e-5
    assert abs(result[4] - 2.0) < 1e-5
    assert abs(result[5] - 1.0) < 1e-5
    print("  PASS: linear quad geometry map")


def test_basis_matrix_matches_evaluate():
    """Precomputed B matrix should match on-the-fly evaluate()."""
    gl = np.array([0.0, 0.5, 1.0])
    basis = QuadBasis(gl, np_fp=np_fp)
    B = precompute_basis_matrix_2d(basis, n_sub=4)

    # Check a few entries against direct evaluation
    n1d = 3
    ndof = n1d * n1d
    for sj in range(5):
        for si in range(5):
            r = si / 4.0
            s = sj / 4.0
            pt = sj * 5 + si
            for j in range(ndof):
                ji = j % n1d
                jj = j // n1d
                # Direct 1D evaluation on host
                lr = 1.0
                for k in range(n1d):
                    if k != ji:
                        lr *= (r - gl[k]) / (gl[ji] - gl[k])
                ls = 1.0
                for k in range(n1d):
                    if k != jj:
                        ls *= (s - gl[k]) / (gl[jj] - gl[k])
                expected = lr * ls
                assert abs(B[pt, j] - expected) < 1e-10, (
                    f"B[{pt},{j}] = {B[pt,j]}, expected {expected}")
    print("  PASS: precomputed basis matrix matches evaluate")


def test_dof_reordering():
    """extract_gl_nodes_1d should produce correct permutation."""
    # MFEM H1 order 2 quad: corners, edge mids, center
    rx = np.array([0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5])
    ry = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.5, 1.0, 0.5, 0.5])
    gl_1d, tp_perm = extract_gl_nodes_1d(rx, ry, 9)

    assert np.allclose(gl_1d, [0.0, 0.5, 1.0])
    # tp_perm[mfem_dof] = tensor_product_index
    # MFEM dof 0 at (0,0) → tp index 0
    assert tp_perm[0] == 0
    # MFEM dof 1 at (1,0) → tp index 2
    assert tp_perm[1] == 2
    # MFEM dof 2 at (1,1) → tp index 8
    assert tp_perm[2] == 8
    # MFEM dof 8 at (0.5,0.5) → tp index 4
    assert tp_perm[8] == 4
    print("  PASS: DOF reordering permutation")


if __name__ == "__main__":
    print("Testing tack.fe...")
    test_lagrange_1d_cardinal()
    test_quad_partition_of_unity()
    test_contiguous_accessor()
    test_linear_quad_map()
    test_basis_matrix_matches_evaluate()
    test_dof_reordering()
    print("All tests passed.")

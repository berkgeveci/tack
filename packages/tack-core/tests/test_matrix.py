"""tack.Matrix: small fixed-size matrices, scalarized like vectors."""

import numpy as np
import pytest

import tack
from tack.lang.source_validation import UnsupportedSyntaxError

N = 5
_RNG = np.random.default_rng(11)
A2 = (_RNG.standard_normal((N, 2, 2)) + 2.0 * np.eye(2)).astype(np.float32)
B2 = _RNG.standard_normal((N, 2, 2)).astype(np.float32)
A3 = (_RNG.standard_normal((N, 3, 3)) + 3.0 * np.eye(3)).astype(np.float32)
B3 = _RNG.standard_normal((N, 3, 3)).astype(np.float32)
V2 = _RNG.standard_normal((N, 2)).astype(np.float32)
V3 = _RNG.standard_normal((N, 3)).astype(np.float32)
R23 = _RNG.standard_normal((N, 2, 3)).astype(np.float32)


def _matrices(values):
    f = tack.Matrix.field(values.shape[1], values.shape[2], dtype=tack.f32, shape=(len(values),))
    f.from_numpy(values)
    return f


def _vectors(values):
    f = tack.Vector.field(values.shape[1], dtype=tack.f32, shape=(len(values),))
    f.from_numpy(values)
    return f


def _like(values):
    return _matrices(np.zeros_like(values))


def _scalars():
    return tack.field(dtype=tack.f32, shape=(N,))


@tack.kernel
def products(a, b, r, v, w, ab, ar, av, wa, ww, n):
    for i in range(n):
        ab[i] = a[i] @ b[i]
        ar[i] = a[i] @ r[i]                 # 2x2 @ 2x3
        av[i] = a[i] @ v[i]                 # a column on the right
        wa[i] = w[i] @ a[i]                 # a row on the left
        ww[i] = v[i] @ w[i]                 # two vectors: their dot product


def test_matrix_products(backend):
    ab, ar = _like(A2), _like(R23)
    av, wa, ww = _vectors(np.zeros_like(V2)), _vectors(np.zeros_like(V2)), _scalars()
    w = V2[::-1].copy()
    products(_matrices(A2), _matrices(B2), _matrices(R23), _vectors(V2), _vectors(w),
             ab, ar, av, wa, ww, N)
    np.testing.assert_allclose(ab.to_numpy(vectors=True), A2 @ B2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ar.to_numpy(vectors=True), A2 @ R23, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(av.to_numpy(vectors=True), np.einsum('nij,nj->ni', A2, V2),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(wa.to_numpy(vectors=True), np.einsum('ni,nij->nj', w, A2),
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ww.to_numpy(), (V2 * w).sum(1), rtol=1e-5, atol=1e-6)


@tack.kernel
def decompose(a2, a3, r, inv2, inv3, rt, dets, traces, n):
    for i in range(n):
        m = a3[i]
        inv2[i] = a2[i].inverse()
        inv3[i] = m.inverse()
        rt[i] = r[i].transpose()
        dets[i] = a2[i].determinant() + m.determinant()
        traces[i] = a2[i].trace() + m.trace()


def test_transpose_trace_determinant_inverse(backend):
    inv2, inv3 = _like(A2), _like(A3)
    rt = _matrices(np.zeros((N, 3, 2), np.float32))
    dets, traces = _scalars(), _scalars()
    decompose(_matrices(A2), _matrices(A3), _matrices(R23), inv2, inv3, rt, dets, traces, N)
    np.testing.assert_allclose(inv2.to_numpy(vectors=True), np.linalg.inv(A2.astype(np.float64)),
                               rtol=2e-4, atol=1e-5)
    np.testing.assert_allclose(inv3.to_numpy(vectors=True), np.linalg.inv(A3.astype(np.float64)),
                               rtol=2e-4, atol=1e-5)
    np.testing.assert_array_equal(rt.to_numpy(vectors=True), R23.transpose(0, 2, 1))
    want = np.linalg.det(A2.astype(np.float64)) + np.linalg.det(A3.astype(np.float64))
    np.testing.assert_allclose(dets.to_numpy(), want, rtol=2e-4)
    np.testing.assert_allclose(traces.to_numpy(), np.trace(A2, axis1=1, axis2=2)
                               + np.trace(A3, axis1=1, axis2=2), rtol=1e-5)


@tack.func
def deformation_update(F, C, dt):
    return (tack.Matrix.identity(2) + dt * C) @ F


@tack.func
def polar_parts(M):
    """A matrix and a scalar back from a device function."""
    S = (M + M.transpose()) * 0.5
    return S, M.trace()


@tack.kernel
def construct_and_call(a, b, v, built, updated, sym, outer, traces, n):
    for i in range(n):
        p = v[i]
        m = tack.Matrix([[p[0], 2.0], [3, p[1]]])
        rows = tack.Matrix([p, p * 2.0])
        built[i] = m * rows - tack.Matrix.identity(2)        # '*' is entry by entry
        updated[i] = deformation_update(a[i], b[i], 0.5)
        s, t = polar_parts(a[i])
        sym[i] = s
        traces[i] = t
        outer[i] = p.outer_product(tack.Vector([1.0, -1.0]))


def test_constructors_device_functions_and_entrywise_arithmetic(backend):
    built, updated, sym, outer = (_like(A2) for _ in range(4))
    traces = _scalars()
    construct_and_call(_matrices(A2), _matrices(B2), _vectors(V2), built, updated, sym, outer,
                       traces, N)
    x, y = V2[:, 0], V2[:, 1]
    m = np.stack([np.stack([x, np.full(N, 2.0, np.float32)], 1),
                  np.stack([np.full(N, 3.0, np.float32), y], 1)], 1)
    rows = np.stack([V2, V2 * 2], 1)
    np.testing.assert_allclose(built.to_numpy(vectors=True), m * rows - np.eye(2), rtol=1e-6)
    np.testing.assert_allclose(updated.to_numpy(vectors=True), (np.eye(2) + 0.5 * B2) @ A2,
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sym.to_numpy(vectors=True), (A2 + A2.transpose(0, 2, 1)) * 0.5,
                               rtol=1e-6)
    np.testing.assert_allclose(traces.to_numpy(), np.trace(A2, axis1=1, axis2=2), rtol=1e-6)
    np.testing.assert_array_equal(outer.to_numpy(vectors=True),
                                  V2[:, :, None] * np.float32([1.0, -1.0])[None, None, :])


@tack.kernel
def entries(a, sel, out, picked, n):
    for i in range(n):
        m = a[i]
        m[0, 1] = -m[1, 0]
        m[1, 1] += 10.0
        k = sel[i]
        m[k, 2] = 100.0                    # a runtime row
        a[i][2, 0] = 7.0                   # an entry of a field element
        a[i][k, 1] *= 2.0
        out[i] = m
        picked[i] = m[k, k] + a[i][1, 2] + (m @ m)[0, 0]


def test_matrix_entries_read_and_written(backend):
    a = _matrices(A3)
    sel = tack.field(dtype=tack.i32, shape=(N,))
    rows = np.array([0, 1, 2, 1, 0], np.int32)
    sel.from_numpy(rows)
    out, picked = _like(A3), _scalars()
    entries(a, sel, out, picked, N)
    m = A3.copy()
    m[:, 0, 1] = -A3[:, 1, 0]
    m[:, 1, 1] += 10
    m[np.arange(N), rows, 2] = 100
    stored = A3.copy()
    stored[:, 2, 0] = 7
    stored[np.arange(N), rows, 1] *= 2
    np.testing.assert_array_equal(out.to_numpy(vectors=True), m)
    np.testing.assert_array_equal(a.to_numpy(vectors=True), stored)
    want = m[np.arange(N), rows, rows] + stored[:, 1, 2] + (m @ m)[:, 0, 0]
    np.testing.assert_allclose(picked.to_numpy(), want, rtol=1e-5)


@tack.kernel
def accumulate(a, total, n):
    for i in range(n):
        tack.atomic_add(total, 0, a[i])
        a[i] += tack.Matrix.identity(2)
        a[i] *= 2.0


def test_matrix_fields_in_atomics_and_augmented_stores(backend):
    whole = np.round(A2 * 8) / 8          # sums of these are exact in f32
    a = _matrices(whole)
    total = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(1,))
    total.fill(0.0)
    accumulate(a, total, N)
    np.testing.assert_array_equal(total.to_numpy(vectors=True)[0], whole.sum(0))
    np.testing.assert_array_equal(a.to_numpy(vectors=True), (whole + np.eye(2, dtype=np.float32)) * 2)


def test_matrix_field_numpy_shapes():
    f = tack.Matrix.field(2, 3, dtype=tack.f32, shape=(4, 2))
    values = np.arange(48, dtype=np.float32).reshape(4, 2, 2, 3)
    f.from_numpy(values)
    assert f.to_numpy().shape == (48,)
    np.testing.assert_array_equal(f.to_numpy(vectors=True), values)
    with pytest.raises(ValueError, match="at most 4x4"):
        tack.Matrix.field(5, 2)


def _product_shape_mismatch():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            out[i] = m[i] @ tack.Vector([1.0, 2.0, 3.0])
    return bad


def _scaled_with_matmul():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            m[i] = m[i] @ 2.0
    return bad


def _one_index():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            a = m[i]
            out[i] = tack.Vector([a[0], a[1]])
    return bad


def _vector_into_matrix_field():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            m[i] = tack.Vector([1.0, 2.0, 3.0, 4.0])
    return bad


def _matrix_plus_vector():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            m[i] = m[i] + tack.Vector([1.0, 2.0, 3.0, 4.0])
    return bad


def _inverse_of_a_rectangle():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            r = tack.Matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            m[i] = (r @ r.transpose()) + r.inverse()
    return bad


def _named_component():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            out[i] = v[i] * m[i].x
    return bad


def _ragged_rows():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            m[i] = tack.Matrix([[1.0, 2.0], [3.0]])
    return bad


def _matrix_as_a_condition():
    @tack.kernel
    def bad(m, v, out, n):
        for i in range(n):
            if m[i]:
                out[i] = v[i]
    return bad


@pytest.mark.parametrize("define, message", [
    (_product_shape_mismatch, "multiplies a 2x2 matrix by a 3-vector"),
    (_scaled_with_matmul, r"use '\*' to scale by a scalar"),
    (_one_index, r"of a 2x2 matrix needs a row and a column: m\[i, j\]"),
    (_vector_into_matrix_field, "stores a 4-vector into a field of 2x2 matrix elements"),
    (_matrix_plus_vector, "combines a 2x2 matrix and 4-vector"),
    (_inverse_of_a_rectangle, r"inverse\(\) needs a square matrix, not a 2x3 matrix"),
    (_named_component, r"'x' names a component of a vector; a matrix entry is m\[i, j\]"),
    (_ragged_rows, "has rows of different lengths"),
    (_matrix_as_a_condition, "has a 2x2 matrix as its condition"),
])
def test_mismatched_matrix_forms_are_rejected(define, message):
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"m": (2, 2), "v": 2, "out": 2})


@tack.func
def _stretch_and_volume(F, dt):
    grown = F * (1.0 + dt)
    return grown, grown.determinant()


@tack.kernel
def unpack_matrix_into_fields(F, J, n):
    for p in range(n):
        F[p], J[p] = _stretch_and_volume(F[p], 0.5)


def test_matrix_unpacked_into_a_field_element_keeps_its_shape(backend):
    """A matrix among several results, unpacked straight into a matrix
    field: the values are evaluated into temporaries before the first
    store, and must still be a matrix afterwards."""
    F = _matrices(A2)
    J = _scalars()
    unpack_matrix_into_fields(F, J, N)
    grown = A2 * np.float32(1.5)
    np.testing.assert_allclose(F.to_numpy(vectors=True), grown, rtol=1e-6)
    np.testing.assert_allclose(J.to_numpy(), np.linalg.det(grown.astype(np.float64)), rtol=1e-5)

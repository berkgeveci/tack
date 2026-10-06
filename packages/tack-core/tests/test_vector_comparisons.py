"""Elementwise comparison of vectors, ``any``/``all``, and ``tack.select``."""

import numpy as np
import pytest

import tack


@tack.func
def _inside_box(p):
    inside = -0.1 <= p <= 1.1                    # a chained comparison, per component
    return all(inside), tack.select(inside, p, 0.0)


@tack.kernel
def comparisons(pos, out, flags, n):
    for i in range(n):
        p = pos[i]
        m = p > 0.5                              # a mask: a vector of 0/1
        out[i, 0] = m.sum()
        out[i, 1] = 1.0 if any(p > 0.9) else 0.0
        ok, clipped = _inside_box(p)
        out[i, 2] = clipped.sum() + (10.0 if ok else 0.0)
        s = tack.select(p < 0.5, -1.0, 1.0)      # scalar arms broadcast
        out[i, 3] = s.x * 1 + s.y * 2 + s.z * 4
        both = (p > 0.2) and (p < 0.8)           # and/or act per component on masks
        out[i, 4] = both.sum() + (not m).sum() * 100
        w = tack.select(p.x > 0.5, p, [0.0, 0.0, 0.0])   # a scalar mask picks whole vectors
        out[i, 5] = w.sum()
        out[i, 6] = (p == pos[i]).sum() + (p != pos[i]).sum() * 10
        if all(p >= 0.0):
            flags[i] = 1


def test_vector_comparisons_select_any_and_all(backend):
    points = np.array([[0.1, 0.6, 0.95], [-0.2, 0.5, 0.3], [0.7, 0.75, 0.3], [1.2, 0.0, 0.5]],
                      np.float32)
    n = len(points)
    pos = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    pos.from_numpy(points)
    out = tack.field(dtype=tack.f32, shape=(n, 7))
    flags = tack.field(dtype=tack.i32, shape=(n,))
    comparisons(pos, out, flags, n)
    want = []
    for p in points:
        inside = (p >= -0.1) & (p <= 1.1)
        m = p > 0.5
        want.append([m.sum(), float((p > 0.9).any()),
                     np.where(inside, p, 0).sum() + (10 if inside.all() else 0),
                     np.where(p < 0.5, -1, 1) @ [1, 2, 4],
                     ((p > 0.2) & (p < 0.8)).sum() + (~m).sum() * 100,
                     (p if p[0] > 0.5 else np.zeros(3)).sum(), 3.0])
    np.testing.assert_allclose(out.to_numpy(), want)
    np.testing.assert_array_equal(flags.to_numpy(), (points >= 0).all(axis=1))


@tack.kernel
def matrix_masks(m, out, n):
    for i in range(n):
        a = m[i]
        positive = a > 0.0
        out[i] = positive.sum() + tack.select(positive, a, -a).sum()   # the sum of |a|


def test_matrices_compare_like_vectors(backend):
    values = np.array([[[1.0, -2.0], [-3.0, 4.0]], [[0.0, 0.0], [0.0, 5.0]]], np.float32)
    m = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(2,))
    m.from_numpy(values)
    out = tack.field(dtype=tack.f32, shape=(2,))
    matrix_masks(m, out, 2)
    np.testing.assert_allclose(out.to_numpy(), [2 + 10, 1 + 5])


@tack.kernel
def links_are_all_evaluated(pos, hits, out, n):
    for i in range(n):
        p = pos[i]
        m = (p.x < 0.0) <= p < 2.0                        # a scalar link beside vector ones
        out[i] = m.sum()
        tack.atomic_add(hits, 0, 1)


def test_every_link_of_a_vector_chain_is_evaluated(backend):
    pos = tack.Vector.field(2, dtype=tack.f32, shape=(2,))
    pos.from_numpy(np.array([[0.5, 3.0], [-1.0, 1.0]], np.float32))
    hits = tack.field(dtype=tack.i32, shape=(1,))
    out = tack.field(dtype=tack.f32, shape=(2,))
    links_are_all_evaluated(pos, hits, out, 2)
    # [0.5, 3.0]: 0 <= p and p < 2 -> [1, 0]; [-1, 1]: 1 <= p and p < 2 -> [0, 1]
    np.testing.assert_array_equal(out.to_numpy(), [1, 1])


def _vector_as_if_condition():
    @tack.kernel
    def bad(vf, out, n):
        for i in range(n):
            if vf[i] > 0.0:
                out[i] = 1.0
    return bad


def _vector_as_while_condition():
    @tack.kernel
    def bad(vf, out, n):
        for i in range(n):
            v = vf[i]
            while v < 1.0:
                v += 1.0
    return bad


def _any_of_a_scalar():
    @tack.kernel
    def bad(vf, out, n):
        for i in range(n):
            out[i] = 1.0 if any(out[i] > 0.0) else 0.0
    return bad


def _mismatched_mask():
    @tack.kernel
    def bad(vf, out, n):
        for i in range(n):
            out[i] = tack.select(vf[i] > 0.0, [1.0, 2.0], 0.0).sum()
    return bad


def _mismatched_comparison():
    @tack.kernel
    def bad(vf, out, n):
        for i in range(n):
            out[i] = (vf[i] > [1.0, 2.0]).sum()
    return bad


@pytest.mark.parametrize("define, message", [
    (_vector_as_if_condition, "has a 3-vector as its condition; reduce it with any"),
    (_vector_as_while_condition, "while .* has a 3-vector as its condition"),
    (_any_of_a_scalar, "a scalar is already a truth value"),
    (_mismatched_mask, "combines a 2-vector and 3-vector"),
    (_mismatched_comparison, "combines a 2-vector and 3-vector"),
])
def test_misused_comparisons_are_rejected(define, message):
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3})


def test_select_is_kernel_only():
    with pytest.raises(RuntimeError, match="inside a @tack.kernel"):
        tack.select(True, 1.0, 2.0)

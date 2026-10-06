"""Tests for new Tack features: @tack.func, ndrange, multi-dim indexing,
field[None], Vector types, and vector methods."""

import numpy as np
import pytest

import tack

# Build list of available backends


# ─── @tack.func inlining ───────────────────────────────────────────────

@tack.func
def add_one(x):
    return x + 1.0


@tack.func
def lerp(a, b, t):
    return a + t * (b - a)


@tack.kernel
def use_func(x, out):
    for i in range(x.shape[0]):
        out[i] = add_one(x[i])


@tack.kernel
def use_lerp(a, b, out):
    for i in range(a.shape[0]):
        out[i] = lerp(a[i], b[i], 0.5)


def test_func_inline_basic(backend):
    """@tack.func with simple return value."""
    n = 100
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    use_func(x, out)
    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 1.0
    assert np.allclose(result, expected)


def test_func_inline_multi_arg(backend):
    """@tack.func with multiple arguments."""
    n = 100
    a = tack.field(dtype=tack.f32, shape=(n,))
    b = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    a.from_numpy(np.zeros(n, dtype=np.float32))
    b.from_numpy(np.ones(n, dtype=np.float32) * 10.0)
    use_lerp(a, b, out)
    result = out.to_numpy()
    expected = np.ones(n, dtype=np.float32) * 5.0
    assert np.allclose(result, expected)


@tack.func
def square(x):
    return x * x


@tack.kernel
def use_nested_func(x, out):
    for i in range(x.shape[0]):
        out[i] = add_one(square(x[i]))


def test_func_inline_nested(backend):
    """Nested @tack.func calls."""
    n = 50
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    use_nested_func(x, out)
    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) ** 2 + 1.0
    assert np.allclose(result, expected)


# ─── field[None] (scalar fields) ──────────────────────────────────────

@tack.kernel
def scale_field(x, factor, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor[0]


def test_scalar_field(backend):
    """Scalar field (1-element) used as a parameter."""
    n = 100
    x = tack.field(dtype=tack.f32, shape=(n,))
    factor = tack.field(dtype=tack.f32, shape=(1,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    factor.from_numpy(np.array([2.5], dtype=np.float32))
    scale_field(x, factor, out)
    result = out.to_numpy()
    assert np.allclose(result, 7.5)


# ─── ndrange (2D parallel iteration) ──────────────────────────────────

@tack.kernel
def fill_2d(out, w_field):
    w = w_field[0]
    for i, j in tack.ndrange(4, 4):
        out[i * 4 + j] = float(i * 10 + j)


def test_ndrange_2d(backend):
    """2D parallel iteration with ndrange."""
    out = tack.field(dtype=tack.f32, shape=(16,))
    w_field = tack.field(dtype=tack.f32, shape=(1,))
    w_field.from_numpy(np.array([4.0], dtype=np.float32))
    fill_2d(out, w_field)
    result = out.to_numpy()
    expected = np.array([i * 10 + j for i in range(4) for j in range(4)],
                        dtype=np.float32)
    assert np.allclose(result, expected)


# ─── Vector types and operations ──────────────────────────────────────

@tack.kernel
def vec_add(ax, ay, az, bx, by, bz, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        b = tack.Vector([bx[i], by[i], bz[i]])
        c = a + b
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_add(backend):
    """Vector addition with component extraction."""
    n = 100
    fields = []
    for _ in range(9):
        fields.append(tack.field(dtype=tack.f32, shape=(n,)))
    ax, ay, az, bx, by, bz, cx, cy, cz = fields

    np.random.seed(42)
    ax.from_numpy(np.random.randn(n).astype(np.float32))
    ay.from_numpy(np.random.randn(n).astype(np.float32))
    az.from_numpy(np.random.randn(n).astype(np.float32))
    bx.from_numpy(np.random.randn(n).astype(np.float32))
    by.from_numpy(np.random.randn(n).astype(np.float32))
    bz.from_numpy(np.random.randn(n).astype(np.float32))

    vec_add(ax, ay, az, bx, by, bz, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), ax.to_numpy() + bx.to_numpy())
    assert np.allclose(cy.to_numpy(), ay.to_numpy() + by.to_numpy())
    assert np.allclose(cz.to_numpy(), az.to_numpy() + bz.to_numpy())


@tack.kernel
def vec_scalar_mul(ax, ay, az, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        c = 2.0 * a
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_scalar_mul(backend):
    """Scalar * vector multiplication."""
    n = 50
    ax = tack.field(dtype=tack.f32, shape=(n,))
    ay = tack.field(dtype=tack.f32, shape=(n,))
    az = tack.field(dtype=tack.f32, shape=(n,))
    cx = tack.field(dtype=tack.f32, shape=(n,))
    cy = tack.field(dtype=tack.f32, shape=(n,))
    cz = tack.field(dtype=tack.f32, shape=(n,))

    ax.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    ay.from_numpy(np.ones(n, dtype=np.float32) * 4.0)
    az.from_numpy(np.ones(n, dtype=np.float32) * 5.0)

    vec_scalar_mul(ax, ay, az, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), 6.0)
    assert np.allclose(cy.to_numpy(), 8.0)
    assert np.allclose(cz.to_numpy(), 10.0)


# ─── Vector methods ───────────────────────────────────────────────────

@tack.kernel
def vec_dot(ax, ay, az, bx, by, bz, out):
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        b = tack.Vector([bx[i], by[i], bz[i]])
        out[i] = a.dot(b)


def test_vector_dot(backend):
    """Vector dot product."""
    n = 50
    ax = tack.field(dtype=tack.f32, shape=(n,))
    ay = tack.field(dtype=tack.f32, shape=(n,))
    az = tack.field(dtype=tack.f32, shape=(n,))
    bx = tack.field(dtype=tack.f32, shape=(n,))
    by = tack.field(dtype=tack.f32, shape=(n,))
    bz = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    ax.from_numpy(np.ones(n, dtype=np.float32))
    ay.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    az.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    bx.from_numpy(np.ones(n, dtype=np.float32) * 4.0)
    by.from_numpy(np.ones(n, dtype=np.float32) * 5.0)
    bz.from_numpy(np.ones(n, dtype=np.float32) * 6.0)

    vec_dot(ax, ay, az, bx, by, bz, out)

    # dot = 1*4 + 2*5 + 3*6 = 32
    assert np.allclose(out.to_numpy(), 32.0)


@tack.kernel
def vec_cross(ax, ay, az, bx, by, bz, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        b = tack.Vector([bx[i], by[i], bz[i]])
        c = a.cross(b)
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_cross(backend):
    """Vector cross product."""
    n = 1
    ax = tack.field(dtype=tack.f32, shape=(n,))
    ay = tack.field(dtype=tack.f32, shape=(n,))
    az = tack.field(dtype=tack.f32, shape=(n,))
    bx = tack.field(dtype=tack.f32, shape=(n,))
    by = tack.field(dtype=tack.f32, shape=(n,))
    bz = tack.field(dtype=tack.f32, shape=(n,))
    cx = tack.field(dtype=tack.f32, shape=(n,))
    cy = tack.field(dtype=tack.f32, shape=(n,))
    cz = tack.field(dtype=tack.f32, shape=(n,))

    # i × j = k
    ax.from_numpy(np.array([1.0], dtype=np.float32))
    ay.from_numpy(np.array([0.0], dtype=np.float32))
    az.from_numpy(np.array([0.0], dtype=np.float32))
    bx.from_numpy(np.array([0.0], dtype=np.float32))
    by.from_numpy(np.array([1.0], dtype=np.float32))
    bz.from_numpy(np.array([0.0], dtype=np.float32))

    vec_cross(ax, ay, az, bx, by, bz, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), 0.0)
    assert np.allclose(cy.to_numpy(), 0.0)
    assert np.allclose(cz.to_numpy(), 1.0)


@tack.kernel
def vec_normalize(ax, ay, az, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = tack.Vector([ax[i], ay[i], az[i]])
        n = a.normalized()
        cx[i] = n[0]
        cy[i] = n[1]
        cz[i] = n[2]


def test_vector_normalized(backend):
    """Vector normalization."""
    n = 1
    ax = tack.field(dtype=tack.f32, shape=(n,))
    ay = tack.field(dtype=tack.f32, shape=(n,))
    az = tack.field(dtype=tack.f32, shape=(n,))
    cx = tack.field(dtype=tack.f32, shape=(n,))
    cy = tack.field(dtype=tack.f32, shape=(n,))
    cz = tack.field(dtype=tack.f32, shape=(n,))

    ax.from_numpy(np.array([3.0], dtype=np.float32))
    ay.from_numpy(np.array([4.0], dtype=np.float32))
    az.from_numpy(np.array([0.0], dtype=np.float32))

    vec_normalize(ax, ay, az, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), 0.6, atol=1e-5)
    assert np.allclose(cy.to_numpy(), 0.8, atol=1e-5)
    assert np.allclose(cz.to_numpy(), 0.0, atol=1e-5)


# ─── @tack.func with vectors ──────────────────────────────────────────

@tack.func
def vec_scale(vx, vy, vz, s):
    rx = vx * s
    ry = vy * s
    rz = vz * s
    return rx


@tack.kernel
def use_func_with_locals(x, out):
    for i in range(x.shape[0]):
        out[i] = vec_scale(x[i], x[i], x[i], 3.0)


def test_func_with_multiple_locals(backend):
    """@tack.func with multiple local variables."""
    n = 50
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    use_func_with_locals(x, out)
    result = out.to_numpy()
    assert np.allclose(result, 6.0)


# ─── Multi-dimensional field indexing ─────────────────────────────────

@tack.kernel
def fill_2d_field(out):
    for i, j in tack.ndrange(4, 3):
        out[i, j] = float(i * 10 + j)


def test_multidim_field_indexing(backend):
    """Multi-dimensional field indexing: field[i, j]."""
    out = tack.field(dtype=tack.f32, shape=(4, 3))
    fill_2d_field(out)
    result = out.to_numpy()
    expected = np.array([[i * 10 + j for j in range(3)] for i in range(4)],
                        dtype=np.float32)
    assert np.allclose(result, expected)


# ─── Vector field load/store ──────────────────────────────────────────

@tack.kernel
def vec_field_load(vf, out_x, out_y, out_z):
    for i in range(out_x.shape[0]):
        v = vf[i]
        out_x[i] = v[0]
        out_y[i] = v[1]
        out_z[i] = v[2]


def test_vector_field_load(backend):
    """Load vectors from a vector field."""
    n = 10
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    out_x = tack.field(dtype=tack.f32, shape=(n,))
    out_y = tack.field(dtype=tack.f32, shape=(n,))
    out_z = tack.field(dtype=tack.f32, shape=(n,))

    data = np.zeros(n * 3, dtype=np.float32)
    for i in range(n):
        data[i * 3 + 0] = float(i)
        data[i * 3 + 1] = float(i * 10)
        data[i * 3 + 2] = float(i * 100)
    vf.from_numpy(data)

    vec_field_load(vf, out_x, out_y, out_z)

    assert np.allclose(out_x.to_numpy(), np.arange(n, dtype=np.float32))
    assert np.allclose(out_y.to_numpy(), np.arange(n, dtype=np.float32) * 10)
    assert np.allclose(out_z.to_numpy(), np.arange(n, dtype=np.float32) * 100)


@tack.func
def _element(vf, i):
    return vf[i]


@tack.func
def _element_2d(vf, i, j):
    return vf[i, j]


@tack.func
def _guarded_element(vf, i, n):
    # Two returns, one a literal vector: both must carry the width.
    if i < 0 or i >= n:
        return tack.Vector([0.0, 0.0, 0.0])
    return vf[i]


@tack.func
def _blend(vf, i, n):
    a = _guarded_element(vf, i, n)
    b = _guarded_element(vf, i + 1, n)
    return a + (b - a) * 0.5


@tack.kernel
def vec_field_through_func(vf, out, blended, n):
    for i in range(n):
        out[i] = _element(vf, i)
        blended[i] = _blend(vf, i, n)


@tack.kernel
def vec_field_2d_through_func(vf, out, w, h):
    for i, j in tack.ndrange(w, h):
        out[i, j] = _element_2d(vf, i, j)


def test_vector_field_element_returned_from_func(backend):
    """A device function given a vector field loads whole vectors from it.

    The inliner propagated vector-variable and texture metadata to renamed
    parameters but not vector-field metadata, so `return vf[i]` lowered to
    one scalar load and the store wrote one component."""
    n = 6
    data = np.arange(n * 3, dtype=np.float32) + 1
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vf.from_numpy(data)
    out = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    blended = tack.Vector.field(3, dtype=tack.f32, shape=(n,))

    vec_field_through_func(vf, out, blended, n)

    v = data.reshape(n, 3)
    np.testing.assert_array_equal(out.to_numpy().reshape(n, 3), v)
    nxt = np.vstack([v[1:], np.zeros((1, 3), np.float32)])
    np.testing.assert_array_equal(blended.to_numpy().reshape(n, 3), v + (nxt - v) * 0.5)


def test_vector_field_2d_element_returned_from_func(backend):
    w, h = 4, 3
    data = np.arange(w * h * 2, dtype=np.float32) + 1
    vf = tack.Vector.field(2, dtype=tack.f32, shape=(w, h))
    vf.from_numpy(data)
    out = tack.Vector.field(2, dtype=tack.f32, shape=(w, h))

    vec_field_2d_through_func(vf, out, w, h)

    np.testing.assert_array_equal(out.to_numpy(), data)


@tack.kernel
def vec_field_store(in_x, in_y, in_z, vf):
    for i in range(in_x.shape[0]):
        v = tack.Vector([in_x[i], in_y[i], in_z[i]])
        vf[i] = v


def test_vector_field_store(backend):
    """Store vectors to a vector field."""
    n = 10
    in_x = tack.field(dtype=tack.f32, shape=(n,))
    in_y = tack.field(dtype=tack.f32, shape=(n,))
    in_z = tack.field(dtype=tack.f32, shape=(n,))
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))

    in_x.from_numpy(np.arange(n, dtype=np.float32))
    in_y.from_numpy(np.arange(n, dtype=np.float32) * 10)
    in_z.from_numpy(np.arange(n, dtype=np.float32) * 100)

    vec_field_store(in_x, in_y, in_z, vf)

    data = vf.to_numpy()
    for i in range(n):
        assert data[i * 3 + 0] == float(i)
        assert data[i * 3 + 1] == float(i * 10)
        assert data[i * 3 + 2] == float(i * 100)


@tack.kernel
def scalar_vec_field_load(cam, out_x, out_y, out_z):
    for i in range(out_x.shape[0]):
        c = cam[None]
        out_x[i] = c[0]
        out_y[i] = c[1]
        out_z[i] = c[2]


def test_scalar_vector_field(backend):
    """Scalar vector field (shape=()) load with field[None]."""
    cam = tack.Vector.field(3, dtype=tack.f32, shape=())
    cam.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    n = 5
    out_x = tack.field(dtype=tack.f32, shape=(n,))
    out_y = tack.field(dtype=tack.f32, shape=(n,))
    out_z = tack.field(dtype=tack.f32, shape=(n,))

    scalar_vec_field_load(cam, out_x, out_y, out_z)

    assert np.allclose(out_x.to_numpy(), 1.0)
    assert np.allclose(out_y.to_numpy(), 2.0)
    assert np.allclose(out_z.to_numpy(), 3.0)


# ─── Multi-return from @tack.func ─────────────────────────────────────

@tack.func
def min_max(a, b):
    lo = min(a, b)
    hi = max(a, b)
    return lo, hi


@tack.kernel
def use_multi_return(x, y, lo_out, hi_out):
    for i in range(x.shape[0]):
        lo, hi = min_max(x[i], y[i])
        lo_out[i] = lo
        hi_out[i] = hi


def test_multi_return_func(backend):
    """@tack.func returning a tuple."""
    n = 50
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    lo_out = tack.field(dtype=tack.f32, shape=(n,))
    hi_out = tack.field(dtype=tack.f32, shape=(n,))

    np.random.seed(42)
    xn = np.random.randn(n).astype(np.float32)
    yn = np.random.randn(n).astype(np.float32)
    x.from_numpy(xn)
    y.from_numpy(yn)

    use_multi_return(x, y, lo_out, hi_out)

    assert np.allclose(lo_out.to_numpy(), np.minimum(xn, yn))
    assert np.allclose(hi_out.to_numpy(), np.maximum(xn, yn))


# ─── Tuple unpacking / swap ──────────────────────────────────────────

@tack.kernel
def sort_pair(x, y):
    for i in range(x.shape[0]):
        a = x[i]
        b = y[i]
        if a > b:
            a, b = b, a
        x[i] = a
        y[i] = b


def test_tuple_swap(backend):
    """Tuple swap: a, b = b, a."""
    n = 100
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))

    np.random.seed(42)
    xn = np.random.randn(n).astype(np.float32)
    yn = np.random.randn(n).astype(np.float32)
    x.from_numpy(xn)
    y.from_numpy(yn)

    sort_pair(x, y)

    assert np.allclose(x.to_numpy(), np.minimum(xn, yn))
    assert np.allclose(y.to_numpy(), np.maximum(xn, yn))


# --- Vector components by runtime index, chained subscripts, component stores ---

@tack.kernel
def vec_component_by_runtime_index(vf, sel, out, n):
    for i in range(n):
        v = vf[i]
        out[i] = v[sel[i]]


@tack.kernel
def vec_chained_subscripts(vf, sel, out, n):
    for i in range(n):
        out[i] = vf[i][1] * 10.0 + vf[i][sel[i]]


@tack.func
def _pair(x):
    return tack.Vector([x, x * 2.0])


@tack.kernel
def vec_call_component(out, n):
    for i in range(n):
        out[i] = _pair(float(i))[1]


@tack.kernel
def vec_component_stores(vf, sel, out, n):
    for i in range(n):
        v = vf[i]
        v[1] = -1.0
        v[sel[i]] = 100.0
        v[0] += 0.5
        out[i] = v


def test_vector_component_by_runtime_index(backend):
    """`v[k]` with a runtime k selects the component; past the last, the last.
    Used to fail IR verification with "expected expr node"."""
    n = 4
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vf.from_numpy(np.arange(12, dtype=np.float32))
    sel = tack.field(dtype=tack.i32, shape=(n,))
    sel.from_numpy(np.array([0, 1, 2, 7], np.int32))
    out = tack.field(dtype=tack.f32, shape=(n,))
    vec_component_by_runtime_index(vf, sel, out, n)
    np.testing.assert_array_equal(out.to_numpy(), [0.0, 4.0, 8.0, 11.0])


def test_vector_chained_subscripts(backend):
    """`vf[i][c]` without naming the vector, with a literal or runtime c."""
    n = 4
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vf.from_numpy(np.arange(12, dtype=np.float32))
    sel = tack.field(dtype=tack.i32, shape=(n,))
    sel.from_numpy(np.array([0, 1, 2, 7], np.int32))
    out = tack.field(dtype=tack.f32, shape=(n,))
    vec_chained_subscripts(vf, sel, out, n)
    np.testing.assert_array_equal(out.to_numpy(), [10.0, 44.0, 78.0, 111.0])
    vec_call_component(out, n)
    np.testing.assert_array_equal(out.to_numpy(), [0.0, 2.0, 4.0, 6.0])


def test_vector_component_stores(backend):
    """`v[c] = x`, `v[k] = x` and `v[c] += x` on a vector variable; a store
    past the last component changes nothing."""
    n = 4
    vf = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vf.from_numpy(np.arange(12, dtype=np.float32))
    sel = tack.field(dtype=tack.i32, shape=(n,))
    sel.from_numpy(np.array([0, 1, 2, 7], np.int32))
    out = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vec_component_stores(vf, sel, out, n)
    np.testing.assert_array_equal(out.to_numpy().reshape(n, 3), [
        [100.5, -1.0, 2.0], [3.5, 100.0, 5.0], [6.5, -1.0, 100.0], [9.5, -1.0, 11.0]])


def test_vector_component_out_of_range_literal_is_rejected():
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match="component 3 of a 3-vector is out of range"):
        @tack.kernel
        def bad(vf, out, n):
            for i in range(n):
                out[i] = vf[i][3]
        bad.get_ir(vector_fields={"vf": 3})


def test_subscript_of_a_scalar_value_is_rejected():
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match="indexes a scalar value"):
        @tack.kernel
        def bad(x, out, n):
            for i in range(n):
                out[i] = x[i][0]
        bad.get_ir()


# --- A vector assignment evaluates its whole right side first ---

@tack.kernel
def vec_assign_from_itself(a, b, crossed, rotated, augmented, n):
    for i in range(n):
        v = a[i]
        v = v.cross(b[i])
        crossed[i] = v
        r = a[i]
        r = tack.Vector([r[1], r[2], r[0]])
        rotated[i] = r
        g = a[i]
        g += g.cross(b[i])
        augmented[i] = g


@tack.kernel
def vec_field_store_from_itself(a, b, c, n):
    for i in range(n):
        a[i] = a[i].cross(b[i])
        c[i] = tack.Vector([c[i][1], c[i][2], c[i][0]])


@tack.kernel
def vec_field_store_through_alias(src, dst, n):
    for i in range(n):
        v = src[i]
        dst[i] = tack.Vector([v[2], src[i][0], src[i][1]])


@tack.kernel
def vec_unpack_from_itself(a, out, n):
    for i in range(n):
        p = a[i]
        x = p[0]
        y = p[1]
        x, y = tack.Vector([x, y]) * 2.0 + tack.Vector([y, x])
        out[i] = x * 10.0 + y


def _vec3_field(values):
    f = tack.Vector.field(3, dtype=tack.f32, shape=(len(values),))
    f.from_numpy(np.ascontiguousarray(values, np.float32).reshape(-1))
    return f


_VEC_A = np.array([[1, 2, 3], [-4, 5, 6], [7, -8, 9], [0.5, 0.25, -2]], np.float32)
_VEC_B = np.array([[2, -1, 4], [3, 3, -5], [-6, 1, 2], [8, -3, 0.5]], np.float32)


def test_vector_variable_assigned_from_itself(backend):
    """`v = v.cross(w)`, a component rotation and `v += v.cross(w)` read the
    old components. They were assigned one at a time, so later components
    read the earlier ones already overwritten."""
    n = len(_VEC_A)
    outs = [tack.Vector.field(3, dtype=tack.f32, shape=(n,)) for _ in range(3)]
    vec_assign_from_itself(_vec3_field(_VEC_A), _vec3_field(_VEC_B), *outs, n)
    crossed, rotated, augmented = (o.to_numpy().reshape(n, 3) for o in outs)
    np.testing.assert_array_equal(crossed, np.cross(_VEC_A, _VEC_B))
    np.testing.assert_array_equal(rotated, _VEC_A[:, [1, 2, 0]])
    np.testing.assert_array_equal(augmented, _VEC_A + np.cross(_VEC_A, _VEC_B))


def test_vector_field_element_stored_from_itself(backend):
    """A store to a vector field element loads its right side before the
    first component is written."""
    n = len(_VEC_A)
    a, c = _vec3_field(_VEC_A), _vec3_field(_VEC_B)
    vec_field_store_from_itself(a, _vec3_field(_VEC_B), c, n)
    np.testing.assert_array_equal(a.to_numpy().reshape(n, 3), np.cross(_VEC_A, _VEC_B))
    np.testing.assert_array_equal(c.to_numpy().reshape(n, 3), _VEC_B[:, [1, 2, 0]])


def test_vector_field_store_through_an_alias(backend):
    """The same holds when source and destination are one field passed
    twice: the kernel cannot see that the names share storage."""
    n = len(_VEC_A)
    f = _vec3_field(_VEC_A)
    vec_field_store_through_alias(f, f, n)
    np.testing.assert_array_equal(f.to_numpy().reshape(n, 3), _VEC_A[:, [2, 0, 1]])


def test_tuple_unpacking_a_vector_built_from_its_targets(backend):
    n = len(_VEC_A)
    out = tack.field(dtype=tack.f32, shape=(n,))
    vec_unpack_from_itself(_vec3_field(_VEC_A), out, n)
    x, y = _VEC_A[:, 0], _VEC_A[:, 1]
    np.testing.assert_array_equal(out.to_numpy(), (2 * x + y) * 10 + (2 * y + x))


# --- Math builtins, casts and conditionals apply to each component ---

@tack.kernel
def vec_math_builtins(a, rounded, clamped, powered, n):
    for i in range(n):
        v = a[i]
        rounded[i] = floor(v) + sqrt(abs(v)) + ceil(v)
        clamped[i] = min(max(v, -1.0), tack.Vector([0.5, 1.5, 2.5])) + max(0, v)
        powered[i] = pow(abs(v), 2.0) + atan2(v, 1.0)


@tack.kernel
def vec_casts(a, out, n):
    for i in range(n):
        whole = int(a[i])
        out[i] = float(whole) * 0.5 + tack.f32(whole)


@tack.func
def _twice(v):
    return v * 2.0


@tack.kernel
def vec_conditionals(a, plain, inlined, mixed, n):
    for i in range(n):
        v = a[i]
        plain[i] = v if v[0] > 0.0 else -v
        inlined[i] = _twice(v) if v[1] > 0.0 else v
        mixed[i] = v if v[2] > 0.0 else 0.0


_VEC_M = (np.arange(12, dtype=np.float32).reshape(4, 3) - 4.5) * np.float32(0.75)


def test_math_builtins_apply_to_each_component(backend):
    """`min`, `max`, `abs`, `floor`, `sqrt`, ... on a vector, with a scalar
    repeated for every component. They used to fail IR verification."""
    n = len(_VEC_M)
    outs = [tack.Vector.field(3, dtype=tack.f32, shape=(n,)) for _ in range(3)]
    vec_math_builtins(_vec3_field(_VEC_M), *outs, n)
    rounded, clamped, powered = (o.to_numpy().reshape(n, 3) for o in outs)
    m = _VEC_M
    np.testing.assert_allclose(
        rounded, np.floor(m) + np.sqrt(np.abs(m)) + np.ceil(m), rtol=1e-6)
    np.testing.assert_array_equal(
        clamped, np.minimum(np.maximum(m, -1), np.float32([0.5, 1.5, 2.5])) + np.maximum(0, m))
    np.testing.assert_allclose(powered, m * m + np.arctan2(m, 1.0), rtol=1e-5, atol=1e-6)


def test_casts_apply_to_each_component(backend):
    n = len(_VEC_M)
    out = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vec_casts(_vec3_field(_VEC_M), out, n)
    np.testing.assert_array_equal(out.to_numpy().reshape(n, 3), np.trunc(_VEC_M) * 1.5)


def test_conditional_expression_with_vector_arms(backend):
    """One condition selects whole vectors; an arm that is an inlined call
    runs only when selected; a scalar arm is repeated."""
    n = len(_VEC_M)
    outs = [tack.Vector.field(3, dtype=tack.f32, shape=(n,)) for _ in range(3)]
    vec_conditionals(_vec3_field(_VEC_M), *outs, n)
    plain, inlined, mixed = (o.to_numpy().reshape(n, 3) for o in outs)
    m = _VEC_M
    np.testing.assert_array_equal(plain, np.where(m[:, :1] > 0, m, -m))
    np.testing.assert_array_equal(inlined, np.where(m[:, 1:2] > 0, 2 * m, m))
    np.testing.assert_array_equal(mixed, np.where(m[:, 2:] > 0, m, 0))


# --- Components by name, components of field elements, augmented element stores ---

@tack.kernel
def vec_named_components(a, out, n):
    for i in range(n):
        v = a[i]
        v.x = -v.y
        v.z += a[i].x
        out[i] = v


@tack.kernel
def vec_field_element_components(a, b, sel, n):
    for i in range(n):
        a[i][1] = 7.0
        a[i][2] += a[i][0]
        a[i].x = float(i)
        b[i].y -= 1.0
        b[i][sel[i]] = 50.0
        b[i][sel[i]] += float(i)


@tack.kernel
def vec_field_augmented(a, b, n):
    for i in range(n):
        a[i] -= 0.5 * tack.Vector([a[i][1], a[i][2], a[i][0]])
        b[i] *= 2.0
        b[i] += a[i]


@tack.kernel
def field_indexed_by_vector(grid, out, n):
    for i in range(n):
        cell = tack.Vector([i, 2 - i % 3])
        out[i] = grid[cell] + grid[cell + 1]


def test_vector_components_by_name(backend):
    """`v.x` is `v[0]`, readable on any vector value and assignable on a
    vector variable. Reads used to fail IR verification."""
    n = len(_VEC_M)
    out = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    vec_named_components(_vec3_field(_VEC_M), out, n)
    want = _VEC_M.copy()
    want[:, 0] = -_VEC_M[:, 1]
    want[:, 2] += _VEC_M[:, 0]
    np.testing.assert_array_equal(out.to_numpy().reshape(n, 3), want)


def test_components_of_a_vector_field_element(backend):
    """`vf[i][c] = x`, `vf[i][c] += x` and `vf[i].y = x` store one component;
    a runtime component past the last stores nothing."""
    n = len(_VEC_M)
    a, b = _vec3_field(_VEC_M), _vec3_field(_VEC_M)
    sel = tack.field(dtype=tack.i32, shape=(n,))
    sel.from_numpy(np.array([0, 1, 2, 7], np.int32))
    vec_field_element_components(a, b, sel, n)
    want_a = _VEC_M.copy()
    want_a[:, 1] = 7
    want_a[:, 2] += _VEC_M[:, 0]
    want_a[:, 0] = np.arange(n)
    want_b = _VEC_M.copy()
    want_b[:, 1] -= 1
    for i, c in enumerate([0, 1, 2]):
        want_b[i, c] = 50.0 + i
    np.testing.assert_array_equal(a.to_numpy().reshape(n, 3), want_a)
    np.testing.assert_array_equal(b.to_numpy().reshape(n, 3), want_b)


def test_augmented_store_to_a_vector_field_element(backend):
    """`vf[i] -= vec` and `vf[i] *= scalar`; the first reads the element it
    updates. Used to raise AttributeError during lowering."""
    n = len(_VEC_M)
    a, b = _vec3_field(_VEC_M), _vec3_field(_VEC_A)
    vec_field_augmented(a, b, n)
    want_a = _VEC_M - np.float32(0.5) * _VEC_M[:, [1, 2, 0]]
    np.testing.assert_array_equal(a.to_numpy().reshape(n, 3), want_a)
    np.testing.assert_array_equal(b.to_numpy().reshape(n, 3), _VEC_A * 2 + want_a)


def test_field_indexed_by_a_vector(backend):
    """A vector index supplies one dimension per component: g[I] is g[I[0], I[1]]."""
    n = 3
    values = np.arange(20, dtype=np.float32).reshape(4, 5)
    grid = tack.field(dtype=tack.f32, shape=values.shape)
    grid.from_numpy(values)
    out = tack.field(dtype=tack.f32, shape=(n,))
    field_indexed_by_vector(grid, out, n)
    rows, cols = np.arange(n), 2 - np.arange(n) % 3
    np.testing.assert_array_equal(out.to_numpy(), values[rows, cols] + values[rows + 1, cols + 1])


def _vector_width_mismatch():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            out[i] = max(vf[i], tack.Vector([1.0, 2.0]))
    return bad


def _component_name_past_the_width():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            v = vf[i]
            out[i] = v * v.w
    return bad


def _swizzle_store():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            v = vf[i]
            v.xy = 1.0
            out[i] = v
    return bad


def _vector_into_a_component():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            vf[i][0] = vf[i]
    return bad


def _vector_into_a_scalar_element():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] += vf[i]
    return bad


def _augmented_width_mismatch():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            vf[i] += tack.Vector([1.0, 2.0])
    return bad


@pytest.mark.parametrize("define, message", [
    (_vector_width_mismatch, "combines a 2-vector and 3-vector"),
    (_component_name_past_the_width, "'w' is not a component of a 3-vector"),
    (_swizzle_store, "'xy' is not a component of a 3-vector"),
    (_vector_into_a_component, "a component of a vector must be a scalar"),
    (_vector_into_a_scalar_element, "combines a scalar target with a vector"),
    (_augmented_width_mismatch, "combines a 2-vector and 3-vector"),
])
def test_mismatched_vector_forms_are_rejected(define, message):
    """Each names the kernel and source position instead of failing IR
    verification with "expected expr node"."""
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3, "out": 3})


# --- Tuple assignment to subscripts and components ---

@tack.func
def _sum_and_difference(a, b):
    return a + b, a - b


@tack.kernel
def tuple_assign_to_subscripts(pos, heading, lo, hi, n):
    for i in range(n):
        p, h = pos[i], heading[i]
        p += tack.Vector([h, -h, 1.0])
        pos[i], heading[i] = p, h * 2.0
        lo[i], hi[i] = hi[i], lo[i]
        lo[i], hi[i] = _sum_and_difference(lo[i], hi[i])


@tack.kernel
def tuple_assign_to_components(a, n):
    for i in range(n):
        v = a[i]
        v.x, v[1] = v.y, v.x
        a[i][2], a[i].x = v.x, v.y
        a[i].y = v.z


def test_tuple_assignment_to_subscripts(backend):
    """`x[i], v[i] = p, q` assigns field elements, whole vectors included,
    after evaluating the whole right side; a swap of two elements works.
    Only plain names were accepted as targets."""
    n = len(_VEC_M)
    pos = _vec3_field(_VEC_M)
    heading, lo, hi = (tack.field(dtype=tack.f32, shape=(n,)) for _ in range(3))
    h = np.arange(n, dtype=np.float32) + 1
    low, high = h * 3, h * 5
    heading.from_numpy(h)
    lo.from_numpy(low)
    hi.from_numpy(high)
    tuple_assign_to_subscripts(pos, heading, lo, hi, n)
    np.testing.assert_array_equal(
        pos.to_numpy().reshape(n, 3), _VEC_M + np.stack([h, -h, np.ones(n, np.float32)], 1))
    np.testing.assert_array_equal(heading.to_numpy(), h * 2)
    np.testing.assert_array_equal(lo.to_numpy(), high + low)
    np.testing.assert_array_equal(hi.to_numpy(), high - low)


def test_tuple_assignment_to_components(backend):
    n = len(_VEC_M)
    a = _vec3_field(_VEC_M)
    tuple_assign_to_components(a, n)
    m = _VEC_M
    # v becomes (y, x, z); then a[i] = (v.y, v.z, v.x) = (x, z, y)
    np.testing.assert_array_equal(a.to_numpy().reshape(n, 3), m[:, [0, 2, 1]])


# --- Atomics with an index per dimension, and vector values ---

@tack.kernel
def atomic_scatter(cells, grid, layers, momentum, bias, n):
    for i in range(n):
        cell = cells[i]
        tack.atomic_add(grid, (cell[0], cell[1]), 1.0)
        tack.atomic_add(layers, (1, cell), 2.0)
        tack.atomic_max(layers, (0, cell[0], cell[1]), float(i))
        tack.atomic_add(momentum, cell, tack.Vector([1.0, float(i)]))
        tack.atomic_add(momentum, (cell[0], 0), tack.Vector([0.0, 1.0]) + bias[0])


def test_atomics_take_an_index_per_dimension(backend):
    """A tuple index, with a vector supplying several dimensions, replaces
    hand-linearized indices; a vector value updates every component of a
    vector field element. A tuple index failed IR verification."""
    n, g = 4000, 4
    rng = np.random.default_rng(3)
    where = rng.integers(0, g, size=(n, 2)).astype(np.int32)
    where[:, 1] = np.maximum(where[:, 1], 1)       # keep column 0 for the last atomic
    bias = tack.Vector.field(2, dtype=tack.f32, shape=(1,))
    bias.from_numpy(np.array([0.5, 0.25], np.float32))
    cells = tack.Vector.field(2, dtype=tack.i32, shape=(n,))
    cells.from_numpy(where.reshape(-1))
    grid = tack.field(dtype=tack.f32, shape=(g, g))
    layers = tack.field(dtype=tack.f32, shape=(2, g, g))
    momentum = tack.Vector.field(2, dtype=tack.f32, shape=(g, g))
    for f in (grid, layers, momentum):
        f.fill(0.0)
    atomic_scatter(cells, grid, layers, momentum, bias, n)

    counts = np.zeros((g, g), np.float32)
    np.add.at(counts, (where[:, 0], where[:, 1]), 1)
    latest = np.zeros((g, g), np.float32)
    np.maximum.at(latest, (where[:, 0], where[:, 1]), np.arange(n, dtype=np.float32))
    sums = np.zeros((g, g, 2), np.float32)
    np.add.at(sums, (where[:, 0], where[:, 1]), np.stack([np.ones(n), np.arange(n)], 1))
    np.add.at(sums, (where[:, 0], 0), np.array([0.5, 1.25]))
    np.testing.assert_array_equal(grid.to_numpy(), counts)
    np.testing.assert_array_equal(layers.to_numpy()[1], counts * 2)
    np.testing.assert_array_equal(layers.to_numpy()[0], latest)
    np.testing.assert_array_equal(momentum.to_numpy().reshape(g, g, 2), sums)


def _atomic_vector_to_scalars():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            tack.atomic_add(s, i, vf[i])
    return bad


def _atomic_scalar_to_vector_element():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            tack.atomic_add(vf, (i, 0), 1.0)
    return bad


def _atomic_width_mismatch():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            tack.atomic_add(vf, i, tack.Vector([1.0, 2.0]))
    return bad


@pytest.mark.parametrize("define, message", [
    (_atomic_vector_to_scalars, "gives a vector to a field of scalars"),
    (_atomic_scalar_to_vector_element, "names an element of a field of 3-vectors"),
    (_atomic_width_mismatch, "gives a 2-vector to a field of 3-vectors"),
])
def test_mismatched_atomics_are_rejected(define, message):
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3, "out": 3})


# --- Several return values with vectors among them; min/max of several; vector reductions ---

@tack.func
def _closest_hit(v, scale):
    closest = v.norm() * scale
    normal = tack.Vector([0.0, 0.0, 0.0])
    color = tack.Vector([0.0, 0.0, 0.0])
    if closest > 2.0:
        normal = v.normalized()
        color = abs(v) * 0.5
    return closest, normal, color


@tack.kernel
def tuple_return_with_vectors(a, shaded, normals, n):
    for i in range(n):
        closest, normal, color = _closest_hit(a[i], 1.5)
        shaded[i] = normal * closest + color
        closest, normals[i], color = _closest_hit(a[i], 1.5)


@tack.kernel
def vec_reductions(a, out, safe, n):
    for i in range(n):
        v = a[i]
        out[i] = tack.Vector([min(v[0], v[1], v[2]), max(v[0], 0.5, v[2], v[1]),
                              v.max() + v.min() + v.sum()])
        safe[i] = v.normalized(1e-3) + (v * 0.0).normalized(1e-3) + max(v, 0.0, -v) * 0.0


def test_device_function_returns_vectors_among_several_values(backend):
    """`return closest, normal, color` with vector elements. The vector
    slots were never bound, and lowering failed on the unbound name."""
    n = len(_VEC_M)
    shaded, normals = (tack.Vector.field(3, dtype=tack.f32, shape=(n,)) for _ in range(2))
    tuple_return_with_vectors(_vec3_field(_VEC_M), shaded, normals, n)
    m = _VEC_M.astype(np.float64)
    length = np.linalg.norm(m, axis=1, keepdims=True)
    far = length * 1.5 > 2.0
    normal = np.where(far, m / length, 0.0)
    want = normal * length * 1.5 + np.where(far, np.abs(m) * 0.5, 0.0)
    np.testing.assert_allclose(shaded.to_numpy().reshape(n, 3), want, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(normals.to_numpy().reshape(n, 3), normal, rtol=1e-6, atol=1e-7)


def test_min_max_of_several_values_and_vector_reductions(backend):
    """`min(a, b, c)` as in Python; `v.min()`, `v.max()` and `v.sum()` over a
    vector's components; `normalized(eps)` for a vector that may be zero."""
    n = len(_VEC_M)
    out, safe = (tack.Vector.field(3, dtype=tack.f32, shape=(n,)) for _ in range(2))
    vec_reductions(_vec3_field(_VEC_M), out, safe, n)
    m = _VEC_M
    want = np.stack([m.min(1), np.maximum(m.max(1), 0.5), m.max(1) + m.min(1) + m.sum(1)], 1)
    np.testing.assert_allclose(out.to_numpy().reshape(n, 3), want, rtol=1e-6)
    length = np.linalg.norm(m.astype(np.float64), axis=1, keepdims=True)
    np.testing.assert_allclose(safe.to_numpy().reshape(n, 3), m / (length + 1e-3), rtol=1e-6)


def _several_values_bound_to_one_name():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            hit = _closest_hit(vf[i], 1.0)
            out[i] = vf[i]
    return bad


def _field_reduced_inside_a_kernel():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] = s.max()
    return bad


@pytest.mark.parametrize("define, message", [
    (_several_values_bound_to_one_name, "binds one target to several values"),
    (_field_reduced_inside_a_kernel, "a field is reduced from Python"),
])
def test_misused_results_and_reductions_are_rejected(define, message):
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3, "out": 3})


# --- ndrange with (start, end) ranges ---

@tack.kernel
def ndrange_interior(a, n, m):
    for i, j in tack.ndrange((1, n - 1), (1, m - 1)):
        a[i, j] = i * 10 + j


@tack.kernel
def ndrange_mixed(a, n, lo, hi, d):
    for i, j, k in tack.ndrange(n, (lo, hi), d):
        a[i, j, k] += 1


@tack.kernel
def ndrange_two_reversed(a, n):
    for i, j in tack.ndrange((3, 1), (4, 2)):
        a[i, j] += 1


@tack.kernel
def ndrange_inner(a, n, m):
    for i in range(n):
        for j, k in tack.ndrange((2, m), (i, i + 2)):
            a[j, k] += 1


def _zeros_i32(shape):
    f = tack.field(dtype=tack.i32, shape=shape)
    f.fill(0)
    return f


def test_ndrange_takes_start_end_ranges(backend):
    """`ndrange((1, n - 1), (1, m - 1))` visits the interior; a size and a
    range can be mixed."""
    a = _zeros_i32((5, 7))
    ndrange_interior(a, 5, 7)
    want = np.zeros((5, 7), np.int32)
    rows, cols = np.mgrid[1:4, 1:6]
    want[1:4, 1:6] = rows * 10 + cols
    np.testing.assert_array_equal(a.to_numpy(), want)

    b = _zeros_i32((3, 6, 2))
    ndrange_mixed(b, 3, 2, 5, 2)
    want = np.zeros((3, 6, 2), np.int32)
    want[:, 2:5, :] = 1
    np.testing.assert_array_equal(b.to_numpy(), want)


@pytest.mark.parametrize("lo, hi", [(4, 4), (5, 2)])
def test_empty_or_reversed_ndrange_range_runs_nothing(backend, lo, hi):
    b = _zeros_i32((3, 6, 2))
    ndrange_mixed(b, 3, lo, hi, 2)
    assert not b.to_numpy().any()


def test_two_reversed_ndrange_ranges_run_nothing(backend):
    """Two negative extents must not multiply into a positive count."""
    a = _zeros_i32((5, 5))
    ndrange_two_reversed(a, 5)
    assert not a.to_numpy().any()


def test_sequential_ndrange_ranges_may_use_the_outer_index(backend):
    a = _zeros_i32((5, 8))
    ndrange_inner(a, 3, 5)
    want = np.zeros((5, 8), np.int32)
    for i in range(3):
        want[2:5, i:i + 2] += 1
    np.testing.assert_array_equal(a.to_numpy(), want)


# --- Vector fields to and from NumPy, one row per vector ---

def test_vector_field_takes_and_returns_one_row_per_vector(backend):
    """`from_numpy` accepts (*shape, n) as well as the flat storage shape;
    `to_numpy(vectors=True)` returns (*shape, n)."""
    values = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    f = tack.Vector.field(3, dtype=tack.f32, shape=(4, 2))
    f.from_numpy(values)
    np.testing.assert_array_equal(f.to_numpy(), values.reshape(-1))
    np.testing.assert_array_equal(f.to_numpy(vectors=True), values)
    f.from_numpy(values.reshape(-1) * 2)
    np.testing.assert_array_equal(f.to_numpy(vectors=True), values * 2)
    with pytest.raises(ValueError, match=r"\(4, 2, 3\) or flat \(24,\)"):
        f.from_numpy(values.reshape(8, 3))
    scalar = tack.field(dtype=tack.f32, shape=(4,))
    scalar.fill(1.0)
    assert scalar.to_numpy(vectors=True).shape == (4,)


# --- What a subscript of a vector field accepts ---

@tack.kernel
def vec_field_scalar_store(a, n):
    for i in range(n):
        a[i] = float(i) + 0.5


def test_scalar_stored_to_a_vector_field_element_sets_every_component(backend):
    """`vf[i] = s` names element i, as a load of `vf[i]` does. It used to
    write the single component at flat index i."""
    n = 4
    a = _vec3_field(_VEC_M)
    vec_field_scalar_store(a, n)
    want = np.repeat(np.arange(n, dtype=np.float32) + 0.5, 3).reshape(n, 3)
    np.testing.assert_array_equal(a.to_numpy(vectors=True), want)


def _vector_into_scalar_storage():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] = vf[i]
    return bad


def _vector_into_another_width():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            vf[i] = tack.Vector([1.0, 2.0])
    return bad


@pytest.mark.parametrize("define, message", [
    (_vector_into_scalar_storage, "stores a 3-vector into an element of scalar storage"),
    (_vector_into_another_width, "stores a 2-vector into a field of 3-vectors"),
])
def test_vector_stores_must_match_their_field(define, message):
    """Both used to write components at offsets computed from the value's
    width, into whatever was there."""
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3, "out": 3})


def test_vector_field_element_index_is_64_bit():
    """The element index is scaled by the width in i64: an i32 product
    wraps for a field of more than 2**31 components."""
    @tack.kernel
    def load(vf, out, where):
        for i in range(1):
            out[i] = vf[where][1]
    vf = tack.Vector.field(2, dtype=tack.i8, shape=(4,))
    out = tack.field(dtype=tack.i8, shape=(1,))
    text = tack.inspect(load, vf, out, 3, mode="ir")
    assert "Cast(where, i64)" in text and "i32" not in text


# --- A vector or tuple where one value is required ---

@tack.func
def _truthy(x):
    return x and 1


def _vector_compared():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] = vf[i] < 1.0
    return bad


def _vector_as_a_condition():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            if vf[i]:
                s[i] = 1.0
    return bad


def _vector_as_a_loop_bound():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            for k in range(vf[i]):
                s[i] += 1.0
    return bad


def _vector_printed():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            print("v", vf[i])
    return bad


def _vector_in_a_device_function():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] = _truthy(vf[i])
    return bad


def _tuple_stored():
    @tack.kernel
    def bad(vf, s, out, n):
        for i in range(n):
            s[i] = (1.0, 2.0)
    return bad


@pytest.mark.parametrize("define, message", [
    (_vector_compared, r"Kernel 'bad': statement at line 4, column \d+ uses a 3-vector"),
    (_vector_as_a_condition, "uses a 3-vector where a single value is required"),
    (_vector_as_a_loop_bound, "uses a 3-vector where a single value is required"),
    (_vector_printed, "uses a 3-vector where a single value is required"),
    (_vector_in_a_device_function,
     r"Device function '_truthy' \(inlined into kernel 'bad'\): statement at line 3"),
    (_tuple_stored, "stores a tuple of 2 values into an element of scalar storage"),
])
def test_vector_where_one_value_is_required_is_diagnosed(define, message):
    """Operations that do not map over a vector's components used to leave
    the vector in the IR, where the verifier reported "expected expr node"
    with a tree path. Any such statement now gets its source position."""
    from tack.lang.source_validation import UnsupportedSyntaxError
    with pytest.raises(UnsupportedSyntaxError, match=message):
        define().get_ir(vector_fields={"vf": 3, "out": 3})

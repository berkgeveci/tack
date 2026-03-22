"""Tests for new PGC features: @pgc.func, ndrange, multi-dim indexing,
field[None], Vector types, and vector methods."""

import numpy as np
import pytest
import pgc
from pgc.lang.ast_transform import transform_kernel
from pgc.lang import ir

# Build list of available backends
_backends = []
for _arch in ["cpu", "metal"]:
    try:
        pgc.init(arch=getattr(pgc, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    pgc.init(arch=getattr(pgc, request.param))
    return request.param


# ─── @pgc.func inlining ───────────────────────────────────────────────

@pgc.func
def add_one(x):
    return x + 1.0


@pgc.func
def lerp(a, b, t):
    return a + t * (b - a)


@pgc.kernel
def use_func(x, out):
    for i in range(x.shape[0]):
        out[i] = add_one(x[i])


@pgc.kernel
def use_lerp(a, b, out):
    for i in range(a.shape[0]):
        out[i] = lerp(a[i], b[i], 0.5)


def test_func_inline_basic(backend):
    """@pgc.func with simple return value."""
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    use_func(x, out)
    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 1.0
    assert np.allclose(result, expected)


def test_func_inline_multi_arg(backend):
    """@pgc.func with multiple arguments."""
    n = 100
    a = pgc.field(dtype=pgc.f32, shape=(n,))
    b = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    a.from_numpy(np.zeros(n, dtype=np.float32))
    b.from_numpy(np.ones(n, dtype=np.float32) * 10.0)
    use_lerp(a, b, out)
    result = out.to_numpy()
    expected = np.ones(n, dtype=np.float32) * 5.0
    assert np.allclose(result, expected)


@pgc.func
def square(x):
    return x * x


@pgc.kernel
def use_nested_func(x, out):
    for i in range(x.shape[0]):
        out[i] = add_one(square(x[i]))


def test_func_inline_nested(backend):
    """Nested @pgc.func calls."""
    n = 50
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    use_nested_func(x, out)
    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) ** 2 + 1.0
    assert np.allclose(result, expected)


# ─── field[None] (scalar fields) ──────────────────────────────────────

@pgc.kernel
def scale_field(x, factor, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * factor[0]


def test_scalar_field(backend):
    """Scalar field (1-element) used as a parameter."""
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    factor = pgc.field(dtype=pgc.f32, shape=(1,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    factor.from_numpy(np.array([2.5], dtype=np.float32))
    scale_field(x, factor, out)
    result = out.to_numpy()
    assert np.allclose(result, 7.5)


# ─── ndrange (2D parallel iteration) ──────────────────────────────────

@pgc.kernel
def fill_2d(out, w_field):
    w = w_field[0]
    for i, j in pgc.ndrange(4, 4):
        out[i * 4 + j] = float(i * 10 + j)


def test_ndrange_2d(backend):
    """2D parallel iteration with ndrange."""
    out = pgc.field(dtype=pgc.f32, shape=(16,))
    w_field = pgc.field(dtype=pgc.f32, shape=(1,))
    w_field.from_numpy(np.array([4.0], dtype=np.float32))
    fill_2d(out, w_field)
    result = out.to_numpy()
    expected = np.array([i * 10 + j for i in range(4) for j in range(4)],
                        dtype=np.float32)
    assert np.allclose(result, expected)


# ─── Vector types and operations ──────────────────────────────────────

@pgc.kernel
def vec_add(ax, ay, az, bx, by, bz, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        b = pgc.Vector([bx[i], by[i], bz[i]])
        c = a + b
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_add(backend):
    """Vector addition with component extraction."""
    n = 100
    fields = []
    for _ in range(9):
        fields.append(pgc.field(dtype=pgc.f32, shape=(n,)))
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


@pgc.kernel
def vec_scalar_mul(ax, ay, az, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        c = 2.0 * a
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_scalar_mul(backend):
    """Scalar * vector multiplication."""
    n = 50
    ax = pgc.field(dtype=pgc.f32, shape=(n,))
    ay = pgc.field(dtype=pgc.f32, shape=(n,))
    az = pgc.field(dtype=pgc.f32, shape=(n,))
    cx = pgc.field(dtype=pgc.f32, shape=(n,))
    cy = pgc.field(dtype=pgc.f32, shape=(n,))
    cz = pgc.field(dtype=pgc.f32, shape=(n,))

    ax.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    ay.from_numpy(np.ones(n, dtype=np.float32) * 4.0)
    az.from_numpy(np.ones(n, dtype=np.float32) * 5.0)

    vec_scalar_mul(ax, ay, az, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), 6.0)
    assert np.allclose(cy.to_numpy(), 8.0)
    assert np.allclose(cz.to_numpy(), 10.0)


# ─── Vector methods ───────────────────────────────────────────────────

@pgc.kernel
def vec_dot(ax, ay, az, bx, by, bz, out):
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        b = pgc.Vector([bx[i], by[i], bz[i]])
        out[i] = a.dot(b)


def test_vector_dot(backend):
    """Vector dot product."""
    n = 50
    ax = pgc.field(dtype=pgc.f32, shape=(n,))
    ay = pgc.field(dtype=pgc.f32, shape=(n,))
    az = pgc.field(dtype=pgc.f32, shape=(n,))
    bx = pgc.field(dtype=pgc.f32, shape=(n,))
    by = pgc.field(dtype=pgc.f32, shape=(n,))
    bz = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    ax.from_numpy(np.ones(n, dtype=np.float32))
    ay.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    az.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    bx.from_numpy(np.ones(n, dtype=np.float32) * 4.0)
    by.from_numpy(np.ones(n, dtype=np.float32) * 5.0)
    bz.from_numpy(np.ones(n, dtype=np.float32) * 6.0)

    vec_dot(ax, ay, az, bx, by, bz, out)

    # dot = 1*4 + 2*5 + 3*6 = 32
    assert np.allclose(out.to_numpy(), 32.0)


@pgc.kernel
def vec_cross(ax, ay, az, bx, by, bz, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        b = pgc.Vector([bx[i], by[i], bz[i]])
        c = a.cross(b)
        cx[i] = c[0]
        cy[i] = c[1]
        cz[i] = c[2]


def test_vector_cross(backend):
    """Vector cross product."""
    n = 1
    ax = pgc.field(dtype=pgc.f32, shape=(n,))
    ay = pgc.field(dtype=pgc.f32, shape=(n,))
    az = pgc.field(dtype=pgc.f32, shape=(n,))
    bx = pgc.field(dtype=pgc.f32, shape=(n,))
    by = pgc.field(dtype=pgc.f32, shape=(n,))
    bz = pgc.field(dtype=pgc.f32, shape=(n,))
    cx = pgc.field(dtype=pgc.f32, shape=(n,))
    cy = pgc.field(dtype=pgc.f32, shape=(n,))
    cz = pgc.field(dtype=pgc.f32, shape=(n,))

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


@pgc.kernel
def vec_normalize(ax, ay, az, cx, cy, cz):
    for i in range(ax.shape[0]):
        a = pgc.Vector([ax[i], ay[i], az[i]])
        n = a.normalized()
        cx[i] = n[0]
        cy[i] = n[1]
        cz[i] = n[2]


def test_vector_normalized(backend):
    """Vector normalization."""
    n = 1
    ax = pgc.field(dtype=pgc.f32, shape=(n,))
    ay = pgc.field(dtype=pgc.f32, shape=(n,))
    az = pgc.field(dtype=pgc.f32, shape=(n,))
    cx = pgc.field(dtype=pgc.f32, shape=(n,))
    cy = pgc.field(dtype=pgc.f32, shape=(n,))
    cz = pgc.field(dtype=pgc.f32, shape=(n,))

    ax.from_numpy(np.array([3.0], dtype=np.float32))
    ay.from_numpy(np.array([4.0], dtype=np.float32))
    az.from_numpy(np.array([0.0], dtype=np.float32))

    vec_normalize(ax, ay, az, cx, cy, cz)

    assert np.allclose(cx.to_numpy(), 0.6, atol=1e-5)
    assert np.allclose(cy.to_numpy(), 0.8, atol=1e-5)
    assert np.allclose(cz.to_numpy(), 0.0, atol=1e-5)


# ─── @pgc.func with vectors ──────────────────────────────────────────

@pgc.func
def vec_scale(vx, vy, vz, s):
    rx = vx * s
    ry = vy * s
    rz = vz * s
    return rx


@pgc.kernel
def use_func_with_locals(x, out):
    for i in range(x.shape[0]):
        out[i] = vec_scale(x[i], x[i], x[i], 3.0)


def test_func_with_multiple_locals(backend):
    """@pgc.func with multiple local variables."""
    n = 50
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    use_func_with_locals(x, out)
    result = out.to_numpy()
    assert np.allclose(result, 6.0)


# ─── Multi-dimensional field indexing ─────────────────────────────────

@pgc.kernel
def fill_2d_field(out):
    for i, j in pgc.ndrange(4, 3):
        out[i, j] = float(i * 10 + j)


def test_multidim_field_indexing(backend):
    """Multi-dimensional field indexing: field[i, j]."""
    out = pgc.field(dtype=pgc.f32, shape=(4, 3))
    fill_2d_field(out)
    result = out.to_numpy()
    expected = np.array([[i * 10 + j for j in range(3)] for i in range(4)],
                        dtype=np.float32)
    assert np.allclose(result, expected)


# ─── Vector field load/store ──────────────────────────────────────────

@pgc.kernel
def vec_field_load(vf, out_x, out_y, out_z):
    for i in range(out_x.shape[0]):
        v = vf[i]
        out_x[i] = v[0]
        out_y[i] = v[1]
        out_z[i] = v[2]


def test_vector_field_load(backend):
    """Load vectors from a vector field."""
    n = 10
    vf = pgc.Vector.field(3, dtype=pgc.f32, shape=(n,))
    out_x = pgc.field(dtype=pgc.f32, shape=(n,))
    out_y = pgc.field(dtype=pgc.f32, shape=(n,))
    out_z = pgc.field(dtype=pgc.f32, shape=(n,))

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


@pgc.kernel
def vec_field_store(in_x, in_y, in_z, vf):
    for i in range(in_x.shape[0]):
        v = pgc.Vector([in_x[i], in_y[i], in_z[i]])
        vf[i] = v


def test_vector_field_store(backend):
    """Store vectors to a vector field."""
    n = 10
    in_x = pgc.field(dtype=pgc.f32, shape=(n,))
    in_y = pgc.field(dtype=pgc.f32, shape=(n,))
    in_z = pgc.field(dtype=pgc.f32, shape=(n,))
    vf = pgc.Vector.field(3, dtype=pgc.f32, shape=(n,))

    in_x.from_numpy(np.arange(n, dtype=np.float32))
    in_y.from_numpy(np.arange(n, dtype=np.float32) * 10)
    in_z.from_numpy(np.arange(n, dtype=np.float32) * 100)

    vec_field_store(in_x, in_y, in_z, vf)

    data = vf.to_numpy()
    for i in range(n):
        assert data[i * 3 + 0] == float(i)
        assert data[i * 3 + 1] == float(i * 10)
        assert data[i * 3 + 2] == float(i * 100)


@pgc.kernel
def scalar_vec_field_load(cam, out_x, out_y, out_z):
    for i in range(out_x.shape[0]):
        c = cam[None]
        out_x[i] = c[0]
        out_y[i] = c[1]
        out_z[i] = c[2]


def test_scalar_vector_field(backend):
    """Scalar vector field (shape=()) load with field[None]."""
    cam = pgc.Vector.field(3, dtype=pgc.f32, shape=())
    cam.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    n = 5
    out_x = pgc.field(dtype=pgc.f32, shape=(n,))
    out_y = pgc.field(dtype=pgc.f32, shape=(n,))
    out_z = pgc.field(dtype=pgc.f32, shape=(n,))

    scalar_vec_field_load(cam, out_x, out_y, out_z)

    assert np.allclose(out_x.to_numpy(), 1.0)
    assert np.allclose(out_y.to_numpy(), 2.0)
    assert np.allclose(out_z.to_numpy(), 3.0)


# ─── Multi-return from @pgc.func ─────────────────────────────────────

@pgc.func
def min_max(a, b):
    lo = min(a, b)
    hi = max(a, b)
    return lo, hi


@pgc.kernel
def use_multi_return(x, y, lo_out, hi_out):
    for i in range(x.shape[0]):
        lo, hi = min_max(x[i], y[i])
        lo_out[i] = lo
        hi_out[i] = hi


def test_multi_return_func(backend):
    """@pgc.func returning a tuple."""
    n = 50
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    lo_out = pgc.field(dtype=pgc.f32, shape=(n,))
    hi_out = pgc.field(dtype=pgc.f32, shape=(n,))

    np.random.seed(42)
    xn = np.random.randn(n).astype(np.float32)
    yn = np.random.randn(n).astype(np.float32)
    x.from_numpy(xn)
    y.from_numpy(yn)

    use_multi_return(x, y, lo_out, hi_out)

    assert np.allclose(lo_out.to_numpy(), np.minimum(xn, yn))
    assert np.allclose(hi_out.to_numpy(), np.maximum(xn, yn))


# ─── Tuple unpacking / swap ──────────────────────────────────────────

@pgc.kernel
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
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))

    np.random.seed(42)
    xn = np.random.randn(n).astype(np.float32)
    yn = np.random.randn(n).astype(np.float32)
    x.from_numpy(xn)
    y.from_numpy(yn)

    sort_pair(x, y)

    assert np.allclose(x.to_numpy(), np.minimum(xn, yn))
    assert np.allclose(y.to_numpy(), np.maximum(xn, yn))

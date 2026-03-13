"""Tests for new PGC features: @pgc.func, ndrange, multi-dim indexing,
field[None], Vector types, and vector methods."""

import numpy as np
import pgc
from pgc.lang.ast_transform import transform_kernel
from pgc.lang import ir


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


def test_func_inline_basic():
    """@pgc.func with simple return value."""
    pgc.init(arch=pgc.cpu)
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    use_func(x, out)
    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 1.0
    assert np.allclose(result, expected)


def test_func_inline_multi_arg():
    """@pgc.func with multiple arguments."""
    pgc.init(arch=pgc.cpu)
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


def test_func_inline_nested():
    """Nested @pgc.func calls."""
    pgc.init(arch=pgc.cpu)
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


def test_scalar_field():
    """Scalar field (1-element) used as a parameter."""
    pgc.init(arch=pgc.cpu)
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


def test_ndrange_2d():
    """2D parallel iteration with ndrange."""
    pgc.init(arch=pgc.cpu)
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


def test_vector_add():
    """Vector addition with component extraction."""
    pgc.init(arch=pgc.cpu)
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


def test_vector_scalar_mul():
    """Scalar * vector multiplication."""
    pgc.init(arch=pgc.cpu)
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


def test_vector_dot():
    """Vector dot product."""
    pgc.init(arch=pgc.cpu)
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


def test_vector_cross():
    """Vector cross product."""
    pgc.init(arch=pgc.cpu)
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


def test_vector_normalized():
    """Vector normalization."""
    pgc.init(arch=pgc.cpu)
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


def test_func_with_multiple_locals():
    """@pgc.func with multiple local variables."""
    pgc.init(arch=pgc.cpu)
    n = 50
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32) * 2.0)
    use_func_with_locals(x, out)
    result = out.to_numpy()
    assert np.allclose(result, 6.0)

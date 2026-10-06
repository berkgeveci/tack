"""tack.constant: named values a kernel may read from its defining scope."""

import math

import numpy as np
import pytest

import tack
from tack.lang.constant import FloatConstant, IntConstant

COUNT = tack.constant(8)
DT = tack.constant(0.25)
GRAVITY = tack.constant(-9.8)
STEPS = tack.constant(3)
MULTIPLIER = tack.constant(747796405, tack.u32)
INCREMENT = tack.constant(2891336453, tack.u32)
TENTH = tack.constant(0.1, tack.f64)
WIDE = tack.constant(5, tack.i64)
plain_python_value = 3.0


@tack.func
def fall(v, dt):
    return v + GRAVITY * dt


@tack.kernel
def ramp(out):
    for i in range(COUNT):
        out[i] = i * DT + GRAVITY


@tack.kernel
def integrate(out, n):
    for i in range(n):
        v = 0.0
        for _ in range(STEPS):
            v = fall(v, DT)
        out[i] = v


@tack.kernel
def hash_step(src, out, n):
    for i in range(n):
        out[i] = src[i] * MULTIPLIER + INCREMENT


@tack.kernel
def exact_double(out, n):
    for i in range(n):
        out[i] = TENTH * 3.0


@tack.kernel
def typed_bound(out):
    for i in range(WIDE):
        out[i] = WIDE + i


@tack.kernel
def circle(out, n):
    for i in range(n):
        out[i] = 2.0 * math.pi * i + math.tau + math.e


@tack.kernel
def shadowed(out, n):
    for i in range(n):
        DT = 2.0
        out[i] = DT


def test_constants_read_as_their_values(backend):
    """A module constant is the literal it names, in a kernel, as a loop
    bound, and inside a device function."""
    out = tack.field(dtype=tack.f32, shape=(COUNT,))
    ramp(out)
    np.testing.assert_allclose(out.to_numpy(), np.arange(8) * 0.25 - 9.8, rtol=1e-6)
    integrate(out, 4)
    np.testing.assert_allclose(out.to_numpy()[:4], -9.8 * 0.25 * 3, rtol=1e-6)


def test_typed_integer_constants_keep_their_width(backend):
    """u32 times a plain literal is i64 arithmetic; times a u32 constant it
    wraps at 32 bits, which a hash needs."""
    values = np.array([1, 7, 123456789, 4000000000], np.uint32)
    src = tack.field(dtype=tack.u32, shape=(4,))
    out = tack.field(dtype=tack.u32, shape=(4,))
    src.from_numpy(values)
    hash_step(src, out, 4)
    with np.errstate(over="ignore"):
        want = values * np.uint32(747796405) + np.uint32(2891336453)
    np.testing.assert_array_equal(out.to_numpy(), want)


def test_typed_constant_as_the_parallel_bound(backend):
    out = tack.field(dtype=tack.i64, shape=(5,))
    typed_bound(out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(5) + 5)


def test_typed_float_constant_is_exact_in_f64(backend):
    if not tack.runtime.dispatch.get_backend().supports_f64:
        pytest.skip("backend has no f64")
    out = tack.field(dtype=tack.f64, shape=(2,))
    exact_double(out, 2)
    np.testing.assert_array_equal(out.to_numpy(), np.full(2, 0.1 * 3.0))


def test_math_module_constants(backend):
    """`math.pi`, `math.e` and `math.tau` read as their values; they used
    to be an unbound name."""
    out = tack.field(dtype=tack.f32, shape=(4,))
    circle(out, 4)
    want = 2.0 * math.pi * np.arange(4) + math.tau + math.e
    np.testing.assert_allclose(out.to_numpy(), want, rtol=1e-6)


@tack.kernel
def scaled_rows(grid, out, n, scale):
    for i, j in tack.ndrange(n, n):
        out[i, j] = grid[i, j] * scale + grid.shape[1]


def test_constants_as_field_shapes_and_arguments(backend):
    """A constant used as a field's shape is baked into indexing as a
    number, and one passed as an argument is an ordinary scalar. The
    shape once reached generated code as the text "tack.constant(8)"."""
    grid = tack.field(dtype=tack.f32, shape=(COUNT, COUNT))
    out = tack.field(dtype=tack.f32, shape=(COUNT, COUNT))
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    grid.from_numpy(values)
    scaled_rows(grid, out, COUNT, DT)
    np.testing.assert_array_equal(out.to_numpy(), values * 0.25 + 8)


def test_a_local_shadows_a_constant(backend):
    out = tack.field(dtype=tack.f32, shape=(2,))
    shadowed(out, 2)
    np.testing.assert_array_equal(out.to_numpy(), [2.0, 2.0])


def test_constants_are_ordinary_numbers_on_the_host():
    assert isinstance(DT, float) and isinstance(DT, FloatConstant)
    assert isinstance(COUNT, int) and isinstance(COUNT, IntConstant)
    assert DT * 2 == 0.5 and COUNT + 1 == 9 and list(range(COUNT))[-1] == 7
    assert np.float32(DT) == np.float32(0.25)
    # They print as the numbers they are: generated code formats them.
    assert repr(MULTIPLIER) == "747796405" and str(DT) == "0.25"
    assert MULTIPLIER.dtype is tack.u32 and DT.dtype is None
    # arithmetic gives plain numbers; a derived constant is declared again
    assert type(DT * 2) is float
    assert tack.constant(True) == 1 and tack.constant(np.float32(0.5)) == 0.5
    assert tack.constant(3, tack.f32) == 3.0 and isinstance(tack.constant(3, tack.f32), float)


@pytest.mark.parametrize("value, dtype, error", [
    (2**40, tack.i32, ValueError),
    (-1, tack.u8, ValueError),
    (2**64, None, ValueError),
    (1.5, tack.i32, TypeError),
    ("x", None, TypeError),
    (1.0, "f32", TypeError),
])
def test_unrepresentable_constants_are_rejected(value, dtype, error):
    with pytest.raises(error):
        tack.constant(value, dtype)


def test_plain_python_values_are_still_not_captured():
    """The contract is unchanged for everything that is not a tack.constant,
    and the error says how to declare one."""
    @tack.kernel
    def bad(out, n):
        for i in range(n):
            out[i] = plain_python_value
    with pytest.raises(NameError, match=r"plain_python_value = tack\.constant\(\.\.\.\)"):
        bad.get_ir()


# --- Vector and matrix constants ---

SUN = tack.constant((0.5, 0.25, 0.0))
ROT = tack.constant(((0.0, -1.0), (1.0, 0.0)))
CELL = tack.constant((2, 3), tack.i32)
HASH = tack.constant((747796405, 2891336453), tack.u32)


@tack.func
def _toward_sun(p):
    return (SUN - p).norm()


@tack.kernel
def vector_constants(pos, grid, out, n):
    for i in range(n):
        p = pos[i]
        r = p - SUN                                 # an operand
        q = ROT @ [p.x, p.y]                        # a matrix constant
        out[i] = (r.norm() + SUN.y + SUN[2] + ROT[1, 0] + q.x + q.y
                  + grid[CELL] + _toward_sun(p) + SUN.sum())


@tack.kernel
def wrapping_vector_constant(out, n):
    for i in range(n):
        h = HASH * tack.u32(3)                       # u32 arithmetic, component by component
        out[i] = h[0] + h[1]


def test_vector_and_matrix_constants_in_kernels(backend):
    """`tack.constant((0.5, 0.25, 0.0))` is `tack.Vector([0.5, 0.25, 0.0])`
    in a kernel and a matrix constant the matrix with those rows: as
    operands, by component (`SUN.y`, `SUN[2]`, `ROT[1, 0]`), as a field
    index, and inside device functions."""
    n = 3
    points = np.array([[0.1, 0.2, 0.3], [0.5, 0.25, 0.0], [1.0, -1.0, 2.0]], np.float32)
    pos = tack.Vector.field(3, dtype=tack.f32, shape=(n,))
    pos.from_numpy(points)
    grid = tack.field(dtype=tack.f32, shape=(4, 4))
    grid.fill(7.0)
    out = tack.field(dtype=tack.f32, shape=(n,))
    vector_constants(pos, grid, out, n)
    sun, rot = np.array(SUN, np.float32), np.array(ROT, np.float32)
    want = [np.linalg.norm(p - sun) + 0.25 + 0.0 + 1.0 + (rot @ p[:2]).sum() + 7.0
            + np.linalg.norm(sun - p) + 0.75 for p in points]
    np.testing.assert_allclose(out.to_numpy(), want, rtol=1e-6)


def test_typed_vector_constant_keeps_its_width(backend):
    out = tack.field(dtype=tack.u32, shape=(1,))
    wrapping_vector_constant(out, 1)
    h = (np.array([747796405, 2891336453], np.uint32) * np.uint32(3))
    assert out.to_numpy()[0] == np.uint32(h[0] + h[1])


def test_vector_constants_on_the_host():
    assert len(SUN) == 3 and SUN[1] == 0.25 and tuple(SUN) == (0.5, 0.25, 0.0)
    assert ROT[1][0] == 1.0 and ROT[0, 1] == -1.0 and len(ROT) == 2
    assert SUN.shape == (3,) and ROT.shape == (2, 2) and CELL.dtype is tack.i32
    np.testing.assert_array_equal(np.array(SUN, np.float32), [0.5, 0.25, 0.0])
    assert np.asarray(CELL).dtype == np.int32          # a typed constant converts to its type
    assert np.asarray(SUN).dtype == np.float64         # an untyped one to Python's float
    assert SUN == (0.5, 0.25, 0.0) and ROT == np.array([[0, -1], [1, 0]])
    assert repr(CELL) == "tack.constant((2, 3), tack.i32)"
    assert repr(ROT) == "tack.constant(((0.0, -1.0), (1.0, 0.0)))"
    assert isinstance(SUN[0], FloatConstant) and isinstance(CELL[1], IntConstant)
    with pytest.raises(TypeError):
        SUN + SUN                                     # not a tuple: no silent concatenation


@pytest.mark.parametrize("value, dtype, error, message", [
    ((), None, ValueError, "no components"),
    (((1, 2), (3,)), None, ValueError, "same, nonzero length"),
    ((1, (2, 3)), None, TypeError, "mix of numbers and rows"),
    ((((1,),),), None, TypeError, "two levels"),
    ((0.5, 1.5), tack.i32, TypeError, "integer type needs an integer"),
    ((2**40, 1), tack.i32, ValueError, "does not fit"),
])
def test_malformed_vector_constants_are_rejected(value, dtype, error, message):
    with pytest.raises(error, match=message):
        tack.constant(value, dtype)


BIG = tack.constant(tuple(tuple(float(i * 5 + j) for j in range(5)) for i in range(5)))


def test_a_matrix_constant_larger_than_4x4_is_rejected_where_it_is_read():
    @tack.kernel
    def bad(out, n):
        for i in range(n):
            out[i] = BIG.trace()

    with pytest.raises(Exception, match="5x5 matrix, larger than 4x4"):
        bad.get_ir()

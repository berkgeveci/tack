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


def test_a_local_shadows_a_constant(backend):
    out = tack.field(dtype=tack.f32, shape=(2,))
    shadowed(out, 2)
    np.testing.assert_array_equal(out.to_numpy(), [2.0, 2.0])


def test_constants_are_ordinary_numbers_on_the_host():
    assert isinstance(DT, float) and isinstance(DT, FloatConstant)
    assert isinstance(COUNT, int) and isinstance(COUNT, IntConstant)
    assert DT * 2 == 0.5 and COUNT + 1 == 9 and list(range(COUNT))[-1] == 7
    assert np.float32(DT) == np.float32(0.25)
    assert repr(MULTIPLIER) == "tack.constant(747796405, tack.u32)"
    assert repr(DT) == "tack.constant(0.25)"
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

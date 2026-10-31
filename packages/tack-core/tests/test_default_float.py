"""``tack.set_default_float``: the float type arrays get when nothing chooses one.
Constructors without a ``dtype``, Python floats, vector and matrix types without a
component type and implicit arrays follow it; NumPy arrays keep their type, and
kernels' precision follows their arrays."""

import numpy as np
import pytest

import tack
import tack.arrays as ta
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


@pytest.fixture
def f64_default():
    tack.set_default_device("cpu")
    tack.set_default_float(tack.f64)
    yield
    tack.set_default_float(tack.f32)


def test_f32_is_the_default():
    assert tack.get_default_float() is tack.f32
    assert tack.empty((2,)).dtype is tack.f32
    assert tack.vec3() == tack.vec(3, tack.f32)


def test_constructors_follow_it(f64_default):
    assert tack.get_default_float() is tack.f64
    assert tack.empty((2,)).dtype is tack.f64
    assert tack.zeros((2,)).dtype is tack.f64
    assert tack.ones((2,)).dtype is tack.f64
    assert tack.full((2,), 1.5).dtype is tack.f64
    assert tack.array([1.0, 2.0]).dtype is tack.f64
    assert tack.array(0.1).numpy() == 0.1                    # no rounding through f32
    assert tack.arange(0.0, 3.0).dtype is tack.f64
    assert tack.zeros((2,), tack.vec3()).dtype == tack.vec(3, tack.f64)
    assert tack.vec3() is tack.vec(3, tack.f64) and tack.mat2() is tack.mat(2, 2, tack.f64)
    assert tack.zeros((2,), layout="soa", dtype=tack.vec2()).components[0].dtype is tack.f64


def test_what_chooses_a_type_keeps_it(f64_default):
    assert tack.empty((2,), tack.f32).dtype is tack.f32
    assert tack.array(np.zeros(2, np.float32)).dtype is tack.f32
    assert tack.array([1, 2]).dtype is tack.i32
    assert tack.vec3(tack.f32) == tack.vec(3, tack.f32)
    assert tack.zeros_like(tack.zeros((2,), tack.f32)).dtype is tack.f32


def test_implicit_arrays_follow_it(f64_default):
    assert ta.ConstantArray(1.5, 4).dtype is tack.f64
    assert ta.ConstantArray(2, 4).dtype is tack.i32
    assert ta.CountingArray(4, 0.0, 0.5).dtype is tack.f64
    assert ta.UniformCoordinates((2, 2, 2)).dtype is tack.f64
    assert ta.CartesianProduct([0.0, 1.0], [0.0, 1.0]).dtype is tack.f64


def test_kernels_compute_in_their_arrays_type(f64_default):
    @tack.kernel
    def scale(x, out, s):
        for i in range(x.shape[0]):
            out[i] = x[i] * s

    x = tack.array([1.0 + 2.0**-40, 3.0])
    out = tack.empty_like(x)
    scale(x, out, 1.0 + 2.0**-40)
    np.testing.assert_array_equal(out.numpy(), x.numpy() * (1.0 + 2.0**-40))
    tack.set_default_float(tack.f32)
    narrow = tack.array([1.0, 3.0])
    scale(narrow, tack.empty_like(narrow), 2.0)           # the same kernel, in f32
    assert narrow.dtype is tack.f32


def test_it_takes_floats_only():
    assert tack.set_default_float(np.float64) is tack.f64
    assert tack.set_default_float("float32") is tack.f32
    for wrong in (tack.i32, tack.vec3(tack.f32), np.int64):
        with pytest.raises(TypeError, match="tack.f32 or tack.f64"):
            tack.set_default_float(wrong)
    assert tack.get_default_float() is tack.f32


class _NoF64CPU(CPUBackend):
    display_name = "device without f64"

    def __init__(self):
        super().__init__()
        self.supported_dtypes = frozenset(d for d in self.supported_dtypes if d is not tack.f64)

    @property
    def memory_domain(self):
        return ("cpu", 8)


def test_a_device_without_f64_says_the_default_chose_it(f64_default):
    """Metal with f64 the default: an array made without a type is refused, with
    the setting named as the cause."""
    no_f64 = dispatch.as_device(_NoF64CPU())
    with pytest.raises(TypeError, match=r"default float type: tack.set_default_float"):
        tack.zeros((2,), device=no_f64)
    assert tack.zeros((2,), tack.f32, device=no_f64).dtype is tack.f32

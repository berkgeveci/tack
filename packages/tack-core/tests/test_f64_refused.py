"""A device without f64 refuses a kernel that computes in f64, once its types are
known: Metal, a Level Zero device without fp64, and a CPU posing as such a device,
which is how the tests find what Metal's compiler would (``refuse_f64``)."""

import numpy as np
import pytest

import tack
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _NoF64CPU(CPUBackend):
    display_name = "device without f64"

    def __init__(self):
        super().__init__()
        self.supported_dtypes = frozenset(d for d in self.supported_dtypes if d is not tack.f64)

    @property
    def memory_domain(self):
        return ("cpu", 7)


@tack.kernel
def _cast(x, out):
    for i in range(out.shape[0]):
        out[i] = tack.f32(tack.f64(x[i]) * 0.5)


@tack.func
def _wide(v):
    w = tack.f64(v)
    return w * w


@tack.kernel
def _inlined(x, out):
    for i in range(out.shape[0]):
        out[i] = tack.f32(_wide(x[i]))


@tack.kernel
def _f32_only(x, out):
    for i in range(out.shape[0]):
        out[i] = x[i] * 0.5 + 1.0


@pytest.fixture
def no_f64():
    tack.set_default_device("cpu")
    yield dispatch.as_device(_NoF64CPU())
    tack.set_default_device("cpu")


def test_f64_in_a_kernel_is_refused(no_f64):
    x = tack.array(np.arange(4, dtype=np.float32), device=no_f64)
    out = tack.empty((4,), tack.f32, device=no_f64)
    with pytest.raises(TypeError, match=r"computes in f64 \(a cast to f64"):
        _cast(x, out)
    with pytest.raises(TypeError, match="computes in f64"):
        _inlined(x, out)                         # from an inlined function
    _f32_only(x, out)                            # weak literals stay f32
    np.testing.assert_array_equal(out.numpy(), np.arange(4) * 0.5 + 1.0)


def test_inspect_refuses_it_too(no_f64):
    x = tack.array(np.arange(4, dtype=np.float32), device=no_f64)
    out = tack.empty((4,), tack.f32, device=no_f64)
    with tack.device(no_f64):
        with pytest.raises(TypeError, match="computes in f64"):
            tack.inspect(_cast, x, out, mode="ir")


def test_devices_with_f64_compute_in_it():
    x = tack.array(np.arange(4, dtype=np.float32))
    out = tack.empty((4,), tack.f32)
    _cast(x, out)
    np.testing.assert_array_equal(out.numpy(), np.arange(4) * 0.5)


def test_f64_arrays_are_refused_where_they_are_made(no_f64):
    """Not at the first kernel given one: every way of making an array checks the
    device's types, so an f64 array never exists on Metal."""
    host = np.arange(4, dtype=np.float64)
    made = [lambda: tack.empty((4,), tack.f64, device=no_f64),
            lambda: tack.zeros((4,), tack.vec3(tack.f64), device=no_f64),
            lambda: tack.zeros((4,), tack.vec3(tack.f64), device=no_f64, layout="soa"),
            lambda: tack.array(host, device=no_f64),
            lambda: tack.array(host).to_device(no_f64),
            lambda: tack.zeros((4,), tack.f32, device=no_f64).astype(tack.f64),
            lambda: tack.empty_like(host, device=no_f64),
            lambda: tack.array_from_ptr(host, (4,), tack.f64, device=no_f64),
            lambda: tack.from_dlpack(host, device=no_f64)]
    for make in made:
        with pytest.raises(TypeError, match="tack.f64 arrays are not supported on"):
            make()
    with tack.device(no_f64):
        with pytest.raises(TypeError, match="Use a supported type"):
            tack.empty((4,), tack.f64)
        assert tack.array(host.astype(np.float32)).dtype is tack.f32

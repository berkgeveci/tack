"""Arrays between Tack and NVIDIA Warp (`tack.interop.warp`), without copies.

On Warp's default device, CUDA or CPU: element types map both ways, the
memory is shared and kept alive by whichever side holds it, and a program
can mix its own Warp kernels with Tack's filters.
"""

import gc

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import arrays
from tack.dataset import algorithms as alg
from tack.runtime.dispatch import env_flag

try:
    import warp as wp
except ImportError:
    if env_flag("TACK_REQUIRE_WARP"):
        raise
    pytest.skip("needs NVIDIA Warp (the warp extra)", allow_module_level=True)
from tack.interop.warp import from_warp, to_warp

pytestmark = pytest.mark.quick


@pytest.fixture(scope="module")
def device():
    return tack.device("warp")


@pytest.fixture(scope="module")
def wp_device(device):
    return device._backend.wp_device


@pytest.mark.parametrize("warp_dtype, tack_dtype, shape", [
    ("float32", tack.f32, (5,)),
    ("int64", tack.i64, (2, 3)),
    ("vec3", tack.vec(3, tack.f32), (4,)),
    ("vec2d", tack.vec(2, tack.f64), (3, 2)),
    ("mat33", tack.mat(3, 3, tack.f32), (2,)),
])
def test_round_trip_shares_memory(device, wp_device, warp_dtype, tack_dtype, shape):
    w = wp.zeros(shape, dtype=getattr(wp, warp_dtype), device=wp_device)
    t = from_warp(w)
    assert t.device == device and t.dtype == tack_dtype and t.shape == shape
    values = np.arange(np.prod(t.numpy().shape), dtype=tack_dtype.numpy_dtype
                       if isinstance(tack_dtype, tack.ScalarType)
                       else tack_dtype.scalar_dtype.numpy_dtype).reshape(t.numpy().shape)
    tack.copyto(t, values)                          # a write through Tack ...
    np.testing.assert_array_equal(w.numpy().reshape(values.shape), values)  # ... seen by Warp
    back = to_warp(t)
    assert back.dtype is w.dtype and back.shape == w.shape and back.ptr == w.ptr


def test_each_side_keeps_the_other_alive(wp_device):
    w = wp.array(np.arange(1000, dtype=np.float32), device=wp_device)
    t = from_warp(w)
    del w
    gc.collect()
    wp.empty(1000, dtype=wp.float32, device=wp_device).fill_(-1.0)  # reuse freed memory, if any
    np.testing.assert_array_equal(t.numpy(), np.arange(1000))

    t2 = tack.array(np.arange(6, dtype=np.float32).reshape(2, 3), device="warp")
    w2 = to_warp(t2)
    del t2
    gc.collect()
    np.testing.assert_array_equal(w2.numpy(), np.arange(6).reshape(2, 3))


def test_wrapped_memory_is_not_shared(wp_device):
    """Warp's own memory on Tack's Warp device needs no completed launches:
    one device, one stream. (A wrapped pointer, `array_from_ptr`, is shared.)"""
    t = from_warp(wp.zeros(8, dtype=wp.float32, device=wp_device))
    assert not t._buffer._shared


def test_implicit_arrays_are_materialized(device):
    grid = arrays.UniformCoordinates((3, 2, 1), origin=(0.0, 0.0, 0.0),
                                     spacing=(1.0, 2.0, 1.0), device=device)
    w = to_warp(grid)
    assert w.dtype is wp.vec3 and w.shape == (6,)
    np.testing.assert_array_equal(w.numpy(), arrays.materialize(grid).numpy())


def test_structure_of_arrays_is_refused(device):
    soa = tack.empty(4, tack.vec(3, tack.f32), layout="soa", device=device)
    with pytest.raises(TypeError, match="components"):
        to_warp(soa)


def test_empty_arrays(wp_device):
    t = from_warp(wp.empty(0, dtype=wp.vec3, device=wp_device))
    assert t.shape == (0,) and t.numpy().shape == (0, 3)
    assert to_warp(tack.empty(0, tack.vec(3, tack.f32), device="warp")).shape == (0,)


def test_from_dlpack_keeps_warp_memory_on_the_warp_device(device, wp_device):
    w = wp.array(np.arange(4, dtype=np.float32), device=wp_device)
    with device:
        t = tack.from_dlpack(w)
    assert t.device == device
    np.testing.assert_array_equal(t.numpy(), np.arange(4))


def test_a_tack_array_on_another_device(wp_device):
    """Through DLPack: on the CPU when Warp's device is its CPU."""
    if wp_device.is_cuda:
        pytest.skip("a CPU array cannot become a CUDA wp.array")
    t = tack.array(np.arange(6, dtype=np.float32).reshape(2, 3), tack.vec(3, tack.f32),
                   device="cpu")
    w = to_warp(t)
    assert w.dtype is wp.vec3 and w.shape == (2,)
    np.testing.assert_array_equal(w.numpy(), t.numpy())


@wp.kernel
def _sphere(field: wp.array(dtype=wp.float32), n: int):
    t = wp.tid()
    i, j, k = t % n, (t // n) % n, t // (n * n)
    p = wp.vec3(float(i), float(j), float(k)) - wp.vec3(float(n - 1) * 0.5)
    field[t] = wp.length(p)


@wp.kernel
def _lift(points: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    out[t] = points[t] + wp.vec3(0.0, 0.0, 100.0)


def test_warp_kernels_around_a_tack_filter(device, wp_device):
    """Warp makes a field, Tack contours it, Warp moves the contour's points."""
    n = 24
    f = wp.empty(n ** 3, dtype=wp.float32, device=wp_device)
    wp.launch(_sphere, n ** 3, inputs=[f, n], device=wp_device)
    with device:
        data = td.uniform_grid((n, n, n))
        data.fields["r"] = td.Field(td.H1(data), from_warp(f))
        iso = alg.contour(data, "r", 8.0)
    points = to_warp(iso.geometry.values)
    assert points.dtype is wp.vec3 and points.shape == (iso.num_points,)
    out = wp.empty_like(points)
    wp.launch(_lift, points.shape[0], inputs=[points, out], device=wp_device)
    centre = (n - 1) / 2
    radii = np.linalg.norm(out.numpy() - [centre, centre, centre + 100.0], axis=1)
    assert iso.num_points > 0 and np.abs(radii - 8.0).max() < 0.02

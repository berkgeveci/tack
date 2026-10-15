"""Devices are values: an array belongs to one, and a kernel runs where its arrays are.

``tack.device("cpu")`` is one device per type, the same value every time,
and ``tack.set_default_device`` makes it the default rather than building
another, so arrays and compiled kernels survive it. A launch runs on the
default device when that can address its arrays, else on the arrays' own;
arrays in two devices' memory are refused. Library code allocates on its
inputs' device.

A second device is simulated with a CPU backend whose memory domain says
otherwise, so the rules are tested on any machine.
"""

import threading

import numpy as np
import pytest

import tack
from tack.runtime import dispatch
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _SecondCPU(CPUBackend):
    """A CPU backend posing as another device's: its arrays are not the default's."""

    display_name = "second device"

    @property
    def memory_domain(self):
        return ("cpu", 1)


@pytest.fixture
def cpu():
    tack.set_default_device("cpu")
    yield tack.device("cpu")
    tack.set_default_device("cpu")


@pytest.fixture
def second(cpu):
    return dispatch.as_device(_SecondCPU())


@tack.kernel
def _double(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0


def _compiled_on(device, kernel):
    return len(device._backend._cache.get(kernel) or ())


def test_a_device_is_one_value(cpu):
    assert tack.device("cpu") is cpu and tack.device(cpu) is cpu
    assert tack.set_default_device("cpu") is cpu
    assert tack.get_default_device() is cpu
    assert cpu.type == "cpu" and cpu.index == 0 and str(cpu) == "cpu"
    assert repr(cpu) == "Device('cpu')"
    # A device made with options is its own.
    serial = tack.device("cpu", num_threads=1)
    assert serial != cpu and serial != tack.device("cpu", num_threads=1)
    with pytest.raises(ValueError, match="Unknown device type"):
        tack.device("vulkan")
    assert tack.device_count("cpu") == 1
    with pytest.raises(ValueError, match="Unknown device type"):
        tack.device_count("vulkan")
    with pytest.raises(ValueError, match="not a device"):
        tack.device("cpu:0")
    with pytest.raises(ValueError, match="does not accept"):
        tack.device("cpu", contxt=1)


def test_a_type_that_cannot_start_has_no_devices(monkeypatch):
    def missing(kind):
        raise ImportError(f"no {kind} bindings here")
    monkeypatch.setattr(dispatch, "_backend_class", missing)
    assert tack.device_count("cuda") == 0


def test_set_default_device_keeps_arrays_and_compiled_kernels(cpu):
    x = tack.array(np.arange(4, dtype=np.float32))
    out = tack.zeros((4,))
    _double(x, out)
    tack.set_default_device("cpu")
    _double(x, out)
    np.testing.assert_array_equal(out.numpy(), [0, 2, 4, 6])
    assert _compiled_on(cpu, _double) == 1


def test_arrays_carry_their_device(cpu, second):
    made = [tack.empty((4,), device=second), tack.zeros((4,), device=second),
            tack.ones((4,), tack.i32, device=second), tack.full((4,), 2.5, device=second),
            tack.arange(4, device=second),
            tack.array(np.zeros(4, np.float32), device=second),
            tack.empty((4,), tack.vec3(), device=second),
            tack.empty((4,), tack.mat2(), device=second)]
    assert all(a.device == second for a in made)
    x = made[3]
    assert x.copy().device == second and x.astype(tack.i32).device == second
    assert x.reshape(2, 2).device == second
    assert tack.empty_like(x).device == second and tack.zeros_like(x).device == second
    assert tack.concat([x, x]).device == second
    np.testing.assert_array_equal(tack.concat([x, x]).numpy(), np.full(8, 2.5))
    assert tack.empty((4,)).device == cpu
    assert tack.empty((4,), device="cpu").device == cpu
    assert tack.empty_like(x, device="cpu").device == cpu


def test_to_device_copies_only_between_memories(cpu, second):
    x = tack.array(np.arange(4, dtype=np.float32))
    assert x.to_device("cpu") is x
    assert x.to_device(tack.device("cpu", num_threads=1)) is x      # one memory
    moved = x.to_device(second)
    assert moved is not x and moved.device == second
    np.testing.assert_array_equal(moved.numpy(), x.numpy())
    assert tack.asarray(x, device=second).device == second
    assert tack.array(x, device=second).device == second


def test_a_kernel_runs_where_its_arrays_are(cpu, second):
    before = _compiled_on(cpu, _double)
    x = tack.array(np.arange(4, dtype=np.float32), device=second)
    out = tack.zeros((4,), device=second)
    _double(x, out)
    np.testing.assert_array_equal(out.numpy(), [0, 2, 4, 6])
    assert _compiled_on(second, _double) == 1
    assert _compiled_on(cpu, _double) == before


def test_arrays_in_the_default_devices_memory_run_on_the_default(cpu):
    serial = tack.device("cpu", num_threads=1)
    x = tack.array(np.arange(4, dtype=np.float32), device=serial)
    out = tack.zeros((4,), device=serial)
    _double(x, out)
    assert _compiled_on(cpu, _double) == 1 and _compiled_on(serial, _double) == 0
    # ...so a default made with options still governs arrays made before it.
    tack.set_default_device("cpu", num_threads=1)
    _double(tack.array(np.ones(4, np.float32)), out)
    assert _compiled_on(tack.get_default_device(), _double) == 1


def test_arrays_on_two_devices_are_refused(cpu, second):
    x = tack.zeros((4,))
    out = tack.zeros((4,), device=second)
    with pytest.raises(ValueError, match=r"Kernel '_double' was given arrays on two devices "
                                         r"\(cpu and cpu\)"):
        _double(x, out)


@tack.kernel
def _count(n) -> tack.i32:
    for i in range(n):
        last = i
    return n * 2


def test_a_kernel_without_arrays_runs_on_the_default_device(cpu, second):
    assert _count(3) == 6
    assert _compiled_on(cpu, _count) == 1
    with second:
        assert _count(4) == 8
    assert _compiled_on(second, _count) == 1


def test_the_default_device_is_scoped_per_thread(cpu, second):
    seen = {}
    with second as inner:
        assert inner is second
        assert tack.get_default_device() == second
        assert tack.zeros((2,)).device == second
        with tack.device("cpu"):
            assert tack.zeros((2,)).device == cpu
        assert tack.zeros((2,)).device == second
        thread = threading.Thread(target=lambda: seen.update(d=tack.get_default_device()))
        thread.start()
        thread.join()
    assert seen["d"] == cpu
    with pytest.raises(RuntimeError), second:
        raise RuntimeError
    assert tack.get_default_device() == cpu


def test_library_code_runs_and_allocates_on_its_inputs_device(cpu, second):
    from tack.algorithms import argsort, dot, exclusive_scan
    before = sum(len(slot) for slot in cpu._backend._cache.values())
    keys = tack.array(np.array([5, 1, 4, 1, 3], np.int32), device=second)
    order = argsort(keys)
    assert order.device == second
    np.testing.assert_array_equal(order.numpy(), [1, 3, 4, 2, 0])
    sums = tack.zeros((5,), tack.i32, device=second)
    exclusive_scan(keys, sums, 5)
    np.testing.assert_array_equal(sums.numpy(), [0, 5, 6, 10, 11])
    assert dot(keys, keys) == 52
    assert keys.sum() == 14
    assert sum(len(slot) for slot in cpu._backend._cache.values()) == before
    assert sum(len(slot) for slot in second._backend._cache.values()) > 0


@tack.device_class
class _Holder:
    def __init__(self, values):
        self.values = values

    @tack.func
    def get(self, i):
        return self.values[i]


@tack.kernel
def _scale_held(h, factor):
    for i in range(h.values.shape[0]):
        h.values[i] = h.get(i) * factor


@tack.kernel
def _copy_held(h, out):
    for i in range(out.shape[0]):
        out[i] = h.get(i)


def test_a_device_class_s_arrays_choose_the_device(cpu, second):
    held = _Holder(tack.array(np.ones(3, np.float32), device=second))
    _scale_held(held, 3.0)
    _scale_held(held, 1.0)                    # a warm call: the fast path
    np.testing.assert_array_equal(held.values.numpy(), [3, 3, 3])
    assert _compiled_on(second, _scale_held) == 1 and _compiled_on(cpu, _scale_held) == 0
    with pytest.raises(ValueError, match="arrays on two devices"):
        _copy_held(held, tack.zeros((3,)))
    with pytest.raises(ValueError, match="arrays on two devices"):
        _copy_held(held, tack.zeros((3,)))    # and warm


def test_a_texture_lives_on_its_arrays_device(cpu, second):
    volume = tack.array(np.ones(8, np.float32), device=second)
    texture = tack.texture3d(volume, (2, 2, 2))
    assert texture._device is second._backend


def test_sync_waits_for_every_device(cpu, monkeypatch):
    serial = tack.device("cpu", num_threads=1)
    waited = []
    for device in (cpu, serial):
        monkeypatch.setattr(device._backend, "synchronize",
                            lambda d=device: waited.append(d))
    tack.sync()
    assert cpu in waited and serial in waited


@pytest.fixture
def debug():
    """Debug mode on, and afterwards as it was: under TACK_DEBUG=1 it stays on."""
    was = dispatch.debug_enabled()
    tack.set_debug(True)
    yield
    tack.set_debug(was)


def test_debug_mode_is_its_own_switch(cpu):
    was = dispatch.debug_enabled()
    try:
        tack.set_debug(False)
        assert not dispatch.debug_enabled()
        tack.set_debug(True)
        assert dispatch.debug_enabled()
        assert tack.get_default_device() is cpu
    finally:
        tack.set_debug(was)


@tack.kernel
def _shift(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i + 1]


def test_debug_mode_checks_a_launch_off_the_default_device(cpu, second, debug):
    """The hidden bounds arrays are made where the launch runs, not on the
    default device (CUDA arrays with the CPU as default failed in the launch
    with an AttributeError from the CPU's buffer)."""
    from tack.runtime import debug_bounds

    x, out = tack.ones((4,), device=second), tack.empty((4,), device=second)
    with pytest.raises(IndexError, match=r"x\[i \+ 1\]"):
        _shift(x, out)
    assert debug_bounds._arrays(second._backend)["errors"].device == second
    with pytest.raises(IndexError):
        _shift(tack.ones((4,)), tack.empty((4,)))       # and back on the default


def test_library_functions_refuse_two_devices_under_their_own_name(cpu, second):
    """Not under the name of the internal kernel that meets the arrays first."""
    from tack.algorithms import exclusive_scan

    here = tack.arange(4, dtype=tack.i32)
    there = tack.empty((4,), tack.i32, device=second)
    with pytest.raises(ValueError, match=r"^exclusive_scan\(\) was given arrays on two "
                                         r"devices \(cpu and cpu\)"):
        exclusive_scan(here, there, 4)
    with pytest.raises(ValueError, match=r"^copyto\(\) was given arrays on two devices"):
        tack.copyto(there, here)
    with pytest.raises(ValueError, match=r"^concat\(\) was given arrays on two devices"):
        tack.concat([here, there])


@pytest.mark.parametrize("count", ["4", 2.0, True, None.__class__])
def test_a_thread_count_is_an_integer(count):
    with pytest.raises(TypeError, match="num_threads is a thread count"):
        tack.device("cpu", num_threads=count)


def test_an_implicit_array_is_where_its_arrays_are(cpu, second):
    from tack import arrays
    from tack.algorithms import exclusive_scan
    base = tack.array(np.arange(4, dtype=np.float32), device=second)
    perm = arrays.Permutation(tack.array(np.array([3, 2, 1, 0], np.int32), device=second),
                              base)
    assert perm.device == second
    assert arrays.CountingArray(3).device == cpu          # holds nothing: the default
    assert perm.sum() == 6
    assert arrays.materialize(perm).device == second
    out = tack.zeros((4,), device=second)
    assert exclusive_scan(perm, out, 4) == 6
    np.testing.assert_allclose(out.numpy(), [0, 3, 5, 6])
    with pytest.raises(ValueError, match="two devices"):
        exclusive_scan(perm, tack.zeros((4,)), 4)

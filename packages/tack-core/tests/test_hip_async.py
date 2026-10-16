"""HIP launches are queued on a stream and return at once; the host waits
only where it reads an array. Results must be the same as when every launch
completed before returning: order, each launch's own scalars, host writes
and fills ordered among the launches, dropped temporaries freed after the
launches that use them, and sync waiting for everything."""
import gc
import threading
import time

import numpy as np
import pytest

import tack

try:
    tack.set_default_device("hip")
except RuntimeError as e:
    pytest.skip(f"HIP not available: {e}", allow_module_level=True)

from tack.runtime import hip_backend
from tack.runtime.dispatch import get_backend


@pytest.fixture(autouse=True)
def _on_hip():
    tack.set_default_device("hip")


@tack.kernel
def _add_into(acc, k: tack.i32):
    for i in range(acc.shape[0]):
        acc[i] = acc[i] + k + i


@tack.kernel
def _copy_scaled(x, out, a: tack.f32):
    for i in range(out.shape[0]):
        out[i] = x[i] * a


@tack.kernel
def _record_value(value, out, k: tack.i32, n: tack.i32):
    for i in range(n):
        out[k * n + i] = value[0] + 0.0 * float(i)


def test_launches_run_in_order():
    """400 launches, each reading what the one before wrote."""
    acc = tack.zeros((1000,), tack.i32)
    for k in range(400):
        _add_into(acc, k)
    expected = sum(range(400)) + 400 * np.arange(1000)
    np.testing.assert_array_equal(acc.numpy(), expected)


def test_each_queued_launch_keeps_its_own_scalars():
    """A variant's scalars are packed into one buffer rewritten per call;
    the rewrite is queued after the launch that reads the old values."""
    outs = [tack.zeros((16,), tack.i32) for _ in range(40)]
    for k, out in enumerate(outs):
        _add_into(out, k)
    for k, out in enumerate(outs):
        np.testing.assert_array_equal(out.numpy(), k + np.arange(16))


def test_host_writes_wait_for_the_launches_before_them():
    acc = tack.zeros((64,), tack.i32)
    _add_into(acc, 1)
    tack.copyto(acc, np.full(64, 100, np.int32))     # after the launch, not under it
    _add_into(acc, 2)
    np.testing.assert_array_equal(acc.numpy(), 102 + np.arange(64))


def test_fills_are_ordered_among_the_launches():
    n, steps = 4096, 300
    value = tack.empty((1,))
    out = tack.empty((steps * n,))
    for k in range(steps):
        value.fill(float(k))
        _record_value(value, out, k, n)
    np.testing.assert_array_equal(out.numpy().reshape(steps, n)[:, 0], np.arange(steps))


def test_a_dropped_temporary_is_freed_after_its_launches():
    """An array Python drops while launches that use it are queued is freed
    in stream order; the launches read and write it whole."""
    x = tack.empty((1 << 20,))
    x.fill(2.0)
    out = tack.zeros((1 << 20,))
    for _ in range(50):
        tmp = tack.zeros((1 << 20,))
        _copy_scaled(x, tmp, 3.0)
        _copy_scaled(tmp, out, 0.5)
        del tmp                                   # queued launches still use it
        gc.collect()
    np.testing.assert_array_equal(out.numpy(), 3.0)


def test_sync_waits_for_queued_launches():
    acc = tack.zeros((64,), tack.i32)
    for k in range(50):
        _add_into(acc, k)
    tack.sync()
    (err,) = hip_backend.hip.hipStreamQuery(get_backend()._stream)
    assert err == hip_backend.hip.hipError_t.hipSuccess      # nothing left queued
    np.testing.assert_array_equal(acc.numpy(), sum(range(50)) + 50 * np.arange(64))


def test_an_indexed_read_waits():
    acc = tack.zeros((64,), tack.i32)
    _add_into(acc, 7)
    assert acc[3] == 10


def test_a_launch_returns_before_it_completes():
    """The host side of a launch is far shorter than the kernel: a heavy
    kernel queued 20 times takes less than one run of it to queue."""
    @tack.kernel
    def heavy(x, out):
        for i in range(out.shape[0]):
            s = 0.0
            for k in range(2000):
                s += tack.sin(x[i] + k * 0.001)
            out[i] = s

    x = tack.empty((1 << 20,))
    out = tack.zeros((1 << 20,))
    heavy(x, out)
    tack.sync()
    t0 = time.perf_counter()
    heavy(x, out)
    tack.sync()
    one = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        heavy(x, out)
    queued = time.perf_counter() - t0
    tack.sync()
    assert queued < one, f"20 launches took {queued*1e3:.1f} ms to queue, one run is {one*1e3:.1f}"


def test_a_shared_buffer_completes_its_launches():
    """Memory handed outside Tack is complete when a launch on it returns."""
    acc = tack.zeros((64,), tack.i32)
    acc._buffer.share()
    assert acc._buffer._shared
    _add_into(acc, 3)
    (err,) = hip_backend.hip.hipStreamQuery(get_backend()._stream)
    assert err == hip_backend.hip.hipError_t.hipSuccess
    np.testing.assert_array_equal(acc.numpy(), 3 + np.arange(64))


def test_reductions_see_the_queued_launches():
    acc = tack.zeros((1000,))
    for k in range(10):
        _copy_scaled(acc, acc, 1.0)
    acc.fill(2.0)
    _copy_scaled(acc, acc, 3.0)
    assert acc.sum() == 6000.0


@tack.kernel
def _busy(x, out):
    for i in range(out.shape[0]):
        s = 0.0
        for k in range(500):
            s += tack.sin(x[i] + k * 0.001)
        out[i] = s


@pytest.mark.parametrize("n", [16, 1 << 12, 1 << 22])
def test_a_host_array_may_change_once_its_write_returns(n):
    """copyto from pageable memory is queued behind busy launches; HIP
    stages a small write and waits before a large one, so either way the
    array gets what the array held when the call returned."""
    x = tack.empty((1 << 20,))
    busy = tack.zeros((1 << 20,))
    array = tack.zeros((n,))
    for _ in range(5):
        _busy(x, busy)
    host = np.arange(n, dtype=np.float32)
    tack.copyto(array, host)
    host[:] = -1.0
    del host
    gc.collect()
    np.testing.assert_array_equal(array.numpy(), np.arange(n, dtype=np.float32))


def test_a_dlpack_export_completes_its_launches():
    """A consumer reads exported memory on its own; the queued launches are
    complete when it gets the capsule, and later ones before they return."""
    acc = tack.zeros((64,), tack.i32)
    for k in range(20):
        _add_into(acc, k)
    acc.__dlpack__()
    assert acc._buffer._shared
    (err,) = hip_backend.hip.hipStreamQuery(get_backend()._stream)
    assert err == hip_backend.hip.hipError_t.hipSuccess
    _add_into(acc, 1)
    (err,) = hip_backend.hip.hipStreamQuery(get_backend()._stream)
    assert err == hip_backend.hip.hipError_t.hipSuccess


@tack.kernel
def _spin(out, iters: tack.i32):
    for i in range(out.shape[0]):
        acc = 0.0
        for k in range(iters):
            acc = acc * 0.999 + 1.0
        out[i] = acc


def _long_launch(out, seconds=0.3):
    """Iterations for a launch of about ``seconds`` on this device."""
    iters = 1000
    while iters < 1 << 28:
        _spin(out, iters)
        tack.sync()
        t0 = time.perf_counter()
        _spin(out, iters)
        tack.sync()
        if time.perf_counter() - t0 >= seconds:
            return iters
        iters *= 4
    pytest.skip("no launch long enough to measure here")


def _nothing_queued():
    (err,) = hip_backend.hip.hipStreamQuery(get_backend()._stream)
    return err == hip_backend.hip.hipError_t.hipSuccess


def test_a_marker_waits_for_the_launches_before_it():
    """`async_marker` covers every launch queued so far, and its wait
    raises nothing when they succeed."""
    acc = tack.zeros((64,), tack.i32)
    for k in range(50):
        _add_into(acc, k)
    marker = get_backend().async_marker()
    assert isinstance(marker, hip_backend._StreamMarker)
    marker.wait()
    assert _nothing_queued()
    np.testing.assert_array_equal(acc.numpy(), sum(range(50)) + 50 * np.arange(64))


def test_a_marker_does_not_wait_for_later_launches():
    out = tack.empty((1 << 18,))
    short = tack.zeros((16,), tack.i32)
    iters = _long_launch(out)
    _add_into(short, 1)
    marker = get_backend().async_marker()
    _spin(out, iters)                        # queued after the mark
    t0 = time.perf_counter()
    marker.wait()
    assert time.perf_counter() - t0 < 0.1
    assert not _nothing_queued()
    np.testing.assert_array_equal(short.numpy(), 1 + np.arange(16))


@pytest.mark.parametrize("wait", ["sync", "numpy", "index", "sum", "marker"])
def test_a_wait_releases_the_gil_and_the_stream(wait):
    """While one thread waits for a long launch, another thread keeps
    launching on the same stream. hip-python's wrappers hold the GIL while
    the runtime waits, and ROCm's hipStreamSynchronize holds the stream:
    with either, the launcher got one launch per 350 ms wait. Waiting on an
    event through ctypes, it launches every millisecond."""
    out = tack.empty((1 << 18,))
    short = tack.zeros((16,), tack.i32)
    iters = _long_launch(out)
    _add_into(short, 1)                      # compiled before the timing,
    out.sum()                                # as is the reduction kernel:
    tack.sync()                              # hipRTC holds the GIL
    waits = {
        "sync": tack.sync,
        "numpy": out.numpy,
        "index": lambda: out[3],
        "sum": out.sum,
        "marker": lambda: get_backend().async_marker().wait(),
    }
    _spin(out, iters)
    waiter = threading.Thread(target=waits[wait])
    waiter.start()
    time.sleep(0.01)
    launched = 0
    t0 = time.perf_counter()
    while waiter.is_alive():
        _add_into(short, 1)
        launched += 1
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    waiter.join()
    # At least one per 20 ms of a 1 ms cadence: a loaded host's 1 ms sleep
    # takes ~10 ms (illyad at load 120-185 reached 0.09 per ms), and a wait
    # that blocks launches allows about one per wait (0.003 per ms).
    assert launched >= elapsed * 1000 * 0.05, (launched, elapsed)
    assert short.numpy()[0] == 1 + launched


def test_a_read_from_another_thread_binds_the_buffers_device():
    """A worker thread starts on device 0; reading an array makes its device
    current for the read and puts the previous one back."""
    out = tack.empty((8,))
    out.fill(2.5)
    device = get_backend()._device
    seen = {}

    def read():
        with out._buffer.thread_binding():
            seen["bound"] = hip_backend.hip.hipGetDevice()[1]
            seen["values"] = out.numpy()
        seen["after"] = hip_backend.hip.hipGetDevice()[1]

    t = threading.Thread(target=read)
    t.start()
    t.join()
    assert seen["bound"] == device
    assert seen["after"] == 0
    np.testing.assert_array_equal(seen["values"], 2.5)
    assert out._buffer.waits_off_thread


def test_waits_go_through_hip_python_without_the_runtime_library(monkeypatch):
    """Without the runtime library through ctypes the waits still work,
    holding the GIL."""
    monkeypatch.setattr(hip_backend, "_RUNTIME", hip_backend._NO_RUNTIME)
    assert hip_backend._runtime() is None
    acc = tack.zeros((64,), tack.i32)
    _add_into(acc, 3)
    get_backend().async_marker().wait()
    tack.sync()
    assert acc[1] == 4
    np.testing.assert_array_equal(acc.numpy(), 3 + np.arange(64))

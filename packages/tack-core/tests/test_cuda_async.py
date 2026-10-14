"""CUDA launches are queued on a stream and return at once; the host waits
only where it reads a field. Results must be the same as when every launch
completed before returning: order, each launch's own scalars, host writes
and fills ordered among the launches, dropped temporaries freed after the
launches that use them, and sync waiting for everything."""
import gc
import time

import numpy as np
import pytest

import tack

try:
    tack.init(arch=tack.cuda)
except RuntimeError as e:
    pytest.skip(f"CUDA not available: {e}", allow_module_level=True)

from tack.runtime import cuda_backend
from tack.runtime.dispatch import get_backend


@pytest.fixture(autouse=True)
def _on_cuda():
    tack.init(arch=tack.cuda)


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
    acc = tack.zeros(tack.i32, (1000,))
    for k in range(400):
        _add_into(acc, k)
    expected = sum(range(400)) + 400 * np.arange(1000)
    np.testing.assert_array_equal(acc.to_numpy(), expected)


def test_each_queued_launch_keeps_its_own_scalars():
    """A variant's scalars are packed into one buffer rewritten per call;
    the rewrite is queued after the launch that reads the old values."""
    outs = [tack.zeros(tack.i32, (16,)) for _ in range(40)]
    for k, out in enumerate(outs):
        _add_into(out, k)
    for k, out in enumerate(outs):
        np.testing.assert_array_equal(out.to_numpy(), k + np.arange(16))


def test_host_writes_wait_for_the_launches_before_them():
    acc = tack.zeros(tack.i32, (64,))
    _add_into(acc, 1)
    acc.from_numpy(np.full(64, 100, np.int32))     # after the launch, not under it
    _add_into(acc, 2)
    np.testing.assert_array_equal(acc.to_numpy(), 102 + np.arange(64))


def test_fills_are_ordered_among_the_launches():
    n, steps = 4096, 300
    value = tack.field(tack.f32, shape=(1,))
    out = tack.field(tack.f32, shape=(steps * n,))
    for k in range(steps):
        value.fill(float(k))
        _record_value(value, out, k, n)
    np.testing.assert_array_equal(out.to_numpy().reshape(steps, n)[:, 0], np.arange(steps))


def test_a_dropped_temporary_is_freed_after_its_launches():
    """A field Python drops while launches that use it are queued is freed
    in stream order; the launches read and write it whole."""
    x = tack.field(tack.f32, (1 << 20,))
    x.fill(2.0)
    out = tack.zeros(tack.f32, (1 << 20,))
    for _ in range(50):
        tmp = tack.zeros(tack.f32, (1 << 20,))
        _copy_scaled(x, tmp, 3.0)
        _copy_scaled(tmp, out, 0.5)
        del tmp                                   # queued launches still use it
        gc.collect()
    np.testing.assert_array_equal(out.to_numpy(), 3.0)


def test_sync_waits_for_queued_launches():
    acc = tack.zeros(tack.i32, (64,))
    for k in range(50):
        _add_into(acc, k)
    tack.sync()
    (err,) = cuda_backend.driver.cuStreamQuery(get_backend()._stream)
    assert err == cuda_backend.driver.CUresult.CUDA_SUCCESS      # nothing left queued
    np.testing.assert_array_equal(acc.to_numpy(), sum(range(50)) + 50 * np.arange(64))


def test_an_indexed_read_waits():
    acc = tack.zeros(tack.i32, (64,))
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

    x = tack.field(tack.f32, (1 << 20,))
    out = tack.zeros(tack.f32, (1 << 20,))
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
    acc = tack.zeros(tack.i32, (64,))
    acc._buffer.share()
    assert acc._buffer._shared
    _add_into(acc, 3)
    (err,) = cuda_backend.driver.cuStreamQuery(get_backend()._stream)
    assert err == cuda_backend.driver.CUresult.CUDA_SUCCESS
    np.testing.assert_array_equal(acc.to_numpy(), 3 + np.arange(64))


def test_reductions_see_the_queued_launches():
    acc = tack.zeros(tack.f32, (1000,))
    for k in range(10):
        _copy_scaled(acc, acc, 1.0)
    acc.fill(2.0)
    _copy_scaled(acc, acc, 3.0)
    assert acc.sum() == 6000.0


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


@pytest.mark.parametrize("wait", ["sync", "to_numpy", "index", "sum", "marker", "exported"])
def test_a_wait_does_not_hold_the_context(wait):
    """While one thread waits for a long launch, another thread keeps
    launching on the same stream. A copy to pageable host memory waits
    inside the driver with the context held, so a read that copied first
    let the launcher make one launch per 910 ms wait (H100, driver 595);
    waiting for the stream first, it launches every millisecond."""
    import threading
    out = tack.field(tack.f32, (1 << 18,))
    exported = tack.field(tack.f32, (16,), exportable=True)
    short = tack.zeros(tack.i32, (16,))
    iters = _long_launch(out)
    _add_into(short, 1)                      # compiled before the timing,
    out.sum()                                # as is the reduction kernel
    tack.sync()
    waits = {
        "sync": tack.sync,
        "to_numpy": out.to_numpy,
        "index": lambda: out[3],
        "sum": out.sum,
        "marker": lambda: get_backend().async_marker().wait(),
        "exported": lambda: (exported.to_numpy(), exported[2]),
    }
    _spin(out, iters)

    def waiter_main():
        with out._buffer.thread_binding():   # the context, on this thread
            waits[wait]()

    waiter = threading.Thread(target=waiter_main)
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
    assert launched >= elapsed * 1000 * 0.1, (launched, elapsed)
    assert short.to_numpy()[0] == 1 + launched

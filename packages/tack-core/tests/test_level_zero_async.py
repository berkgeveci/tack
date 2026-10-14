"""Level Zero launches are queued on one immediate command list, ordered by
barriers, and return at once; the host waits only where it reads a field. Results must be the same as when every launch
completed before returning: order, each launch's own scalars, host writes
and fills ordered among the launches, dropped temporaries freed after the
launches that use them, and sync waiting for everything."""
import asyncio
import gc
import threading
import time

import numpy as np
import pytest

import tack

try:
    tack.init(arch=tack.level_zero)
except (ImportError, RuntimeError, OSError, AttributeError) as e:
    pytest.skip(f"Level Zero not available: {e}", allow_module_level=True)

from tack.runtime import level_zero_backend as l0
from tack.runtime.dispatch import get_backend


@pytest.fixture(autouse=True)
def _on_level_zero():
    tack.init(arch=tack.level_zero)


def _nothing_queued():
    """Whether the backend's queue has run everything appended to it."""
    backend = get_backend()
    return l0._get_ze().zeCommandListHostSynchronize(backend._queue, 0) == l0.ZE_RESULT_SUCCESS


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
    the rewrite is queued after the launch that reads the old values. Each
    launch also writes its own field: the kernel's argument state, set
    again by the next call, was captured when the launch was appended."""
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
    after the work queued so far (the driver's deferred free); the launches
    read and write it whole."""
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
    assert _nothing_queued()
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
    assert _nothing_queued()
    np.testing.assert_array_equal(acc.to_numpy(), 3 + np.arange(64))


def test_reductions_see_the_queued_launches():
    acc = tack.zeros(tack.f32, (1000,))
    for k in range(10):
        _copy_scaled(acc, acc, 1.0)
    acc.fill(2.0)
    _copy_scaled(acc, acc, 3.0)
    assert acc.sum() == 6000.0


def test_the_backend_queues_on_this_driver():
    """zeCommandListHostSynchronize is there (API 1.6); without it the
    backend runs every launch to completion."""
    assert get_backend()._queued


def test_a_host_array_may_change_once_from_numpy_returns():
    """The copy to the device runs later, from a private copy of the array."""
    acc = tack.zeros(tack.i32, (1 << 16,))
    _add_into(acc, 1)                              # the copy below queues behind it
    values = np.full(1 << 16, 5, np.int32)
    acc.from_numpy(values)
    values[:] = -1
    del values
    gc.collect()
    _add_into(acc, 0)
    np.testing.assert_array_equal(acc.to_numpy(), 5 + np.arange(1 << 16))


def test_the_host_memory_held_for_queued_copies_is_bounded():
    """A loop that writes and never reads waits now and then, rather than
    holding every array it ever wrote."""
    backend = get_backend()
    acc = tack.zeros(tack.i32, (256,))
    for k in range(3 * backend._HOLD_COUNT):
        acc.from_numpy(np.full(256, k, np.int32))
        assert len(backend._held) <= backend._HOLD_COUNT + 1
    np.testing.assert_array_equal(acc.to_numpy(), 3 * backend._HOLD_COUNT - 1)
    assert not backend._held                        # the read waited, releasing them


@tack.kernel
def _heavy(x, out):
    for i in range(out.shape[0]):
        s = 0.0
        for k in range(2000):
            s += tack.sin(x[i] + k * 0.001)
        out[i] = s


def _heavy_queue(seconds):
    """Queue about ``seconds`` of `_heavy` (timed by its fastest of three
    runs, as other processes may share the device) and return the fields."""
    x = tack.field(tack.f32, (1 << 20,))
    out = tack.zeros(tack.f32, (1 << 20,))
    _heavy(x, out)
    tack.sync()
    one = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        _heavy(x, out)
        tack.sync()
        one = min(one, time.perf_counter() - t0)
    for _ in range(max(1, int(seconds / one))):
        _heavy(x, out)
    return x, out


def test_a_read_waits_outside_the_launch_lock():
    """A read queues its copy and an event under the backend's lock and
    waits on the event without it, so another thread may launch while a
    read waits for long work."""
    backend = get_backend()
    x, out = _heavy_queue(0.5)
    reader = threading.Thread(target=out.to_numpy)
    reader.start()
    time.sleep(0.05)
    assert reader.is_alive(), "the queued work finished before the check"
    assert backend._launch_lock.acquire(timeout=0.05), "the read held the lock while it waited"
    backend._launch_lock.release()
    waited = reader.is_alive()
    reader.join()
    assert waited, "the read finished before the lock was taken"


def test_a_marker_releases_the_holds_before_it():
    backend = get_backend()
    acc = tack.zeros(tack.i32, (256,))
    tack.sync()
    acc.from_numpy(np.full(256, 1, np.int32))
    marker = backend.async_marker()
    acc.from_numpy(np.full(256, 2, np.int32))      # after the mark: still held
    marker.wait()
    marker.wait()                                  # a second wait returns at once
    assert len(backend._held) == 1 and backend._held_bytes == 256 * 4
    np.testing.assert_array_equal(acc.to_numpy(), 2)
    assert not backend._held and backend._held_bytes == 0


def test_markers_reuse_their_events():
    """Each wait hands its event back: many waits use one pool."""
    backend = get_backend()
    acc = tack.zeros(tack.i32, (64,))
    acc.to_numpy()
    pools = len(backend._event_pools)
    for k in range(3 * backend._EVENT_POOL_SIZE):
        _add_into(acc, 1)
        asyncio.run(tack.sync_async())
        assert acc[0] == k + 1
    assert len(backend._event_pools) == pools
    assert len(backend._events) == pools * backend._EVENT_POOL_SIZE


def test_small_host_copies_do_not_wait_for_the_queue():
    """Appending a copy from pageable memory waited for the queue to drain
    (to the device up to 4 MiB, from it up to 1 KiB, driver 25.05): a
    launch's scalar pack, a small from_numpy and a pattern fill waited for
    the long kernel queued before them. They are staged in USM host memory."""
    acc = tack.zeros(tack.i32, (64,))
    big = tack.zeros(tack.f32, (1 << 20,))
    _add_into(acc, 0)
    big.fill(1.0)
    tack.sync()
    _heavy_queue(0.3)
    t0 = time.perf_counter()
    _add_into(acc, 5)                              # its scalar is uploaded
    acc.from_numpy(np.arange(64, dtype=np.int32))
    big.from_numpy(np.full(1 << 20, 2.0, np.float32))
    queued = time.perf_counter() - t0
    tack.sync()
    whole = time.perf_counter() - t0
    assert queued < 0.25 * whole, (queued, whole)
    np.testing.assert_array_equal(acc.to_numpy(), np.arange(64))
    assert big[7] == 2.0


@pytest.mark.parametrize("nbytes", [4, 1 << 10, (64 << 10) + 4, (16 << 20) + 4])
def test_copies_either_side_of_the_staging_limits(nbytes):
    """Up to `_HostStaging.READ_LIMIT`/`WRITE_LIMIT` copies are staged,
    above them direct; both ways round trip."""
    n = nbytes // 4
    values = np.arange(n, dtype=np.int32) * 3 - 7
    f = tack.zeros(tack.i32, (n,))
    f.from_numpy(values)
    _add_into(f, 1)
    np.testing.assert_array_equal(f.to_numpy(), values + 1 + np.arange(n))
    assert f[n - 1] == values[-1] + n


def test_staging_blocks_go_back_to_the_pool():
    backend = get_backend()
    staging = backend._staging
    acc = tack.zeros(tack.i32, (256,))
    for k in range(200):
        acc.from_numpy(np.full(256, k, np.int32))
        acc[k % 256]
    tack.sync()
    assert not backend._held
    kept = sum(len(blocks) * size for size, blocks in staging._free.items())
    assert kept == staging._kept <= staging._KEEP_BYTES
    assert len(staging._free[1024]) <= 2               # reused, not one per copy


def test_threads_launch_and_free_at_once():
    """Four threads each allocate, launch on and drop fields. The deferred
    free ran outside the launch lock and crashed the driver."""
    import threading

    def work(k):
        for _ in range(400):
            f = tack.zeros(tack.i32, (256,))
            _add_into(f, k)
            del f

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    tack.sync()

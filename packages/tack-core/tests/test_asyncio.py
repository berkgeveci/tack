"""`tack.sync_async` and `Array.numpy_async`: waiting for the device from
``asyncio`` without blocking the event loop (`tack.runtime.aio`)."""
import asyncio
import sys
import time

import numpy as np
import pytest

import tack
from tack.runtime import aio
from tack.runtime.dispatch import get_backend

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests

# The tasks below tick on 1 ms asyncio sleeps, which last a tick of the event
# loop's clock: about 1 ms on Linux and macOS, but 15.6 ms on Windows (13 ms
# idle on aws-windows, 41 ms with four pytest workers). There a free wait is
# asked for fewer ticks per ms, and waits are made longer so that it still
# stands clear of a blocked wait's 2.
_TICKS_PER_MS = 0.01 if sys.platform == "win32" else 0.05
_MIN_WAIT = 0.4 if sys.platform == "win32" else 0.1


@tack.kernel
def _ramp(out, scale: tack.f32):
    for i in range(out.shape[0]):
        out[i] = i * scale


@tack.kernel
def _spin(out, iters: tack.i32):
    for i in range(out.shape[0]):
        acc = 0.0
        for k in range(iters):
            acc = acc * 0.999 + 1.0
        out[i] = acc


def _queues(backend_name):
    """Whether this backend waits off the loop (has a marker)."""
    return get_backend().async_marker() is not None


def test_sync_async_then_the_results_are_there(backend):
    out = tack.empty((1000,))

    async def main():
        _ramp(out, 2.0)
        await tack.sync_async()
        return out.numpy()

    np.testing.assert_array_equal(asyncio.run(main()), np.arange(1000, dtype=np.float32) * 2)


def test_numpy_async_reads_what_was_queued(backend):
    arrays = [tack.empty((500,)) for _ in range(4)]
    vec = tack.empty((7,), tack.vec(3, tack.f32))
    tack.copyto(vec, np.arange(21, dtype=np.float32).reshape(7, 3))

    async def main():
        for k, f in enumerate(arrays):
            _ramp(f, float(k + 1))
        rows, again = await asyncio.gather(vec.numpy_async(), vec.numpy_async())
        return await asyncio.gather(*(f.numpy_async() for f in arrays)), rows, again

    got, rows, again = asyncio.run(main())
    for k, values in enumerate(got):
        np.testing.assert_array_equal(values, np.arange(500, dtype=np.float32) * (k + 1))
    assert rows.shape == (7, 3)
    np.testing.assert_array_equal(rows, np.arange(21, dtype=np.float32).reshape(7, 3))
    np.testing.assert_array_equal(again, rows)


def _long_launch(out):
    """Iterations for a launch of about 150 ms on this device (or None if
    even a long one is too quick to tell from the loop's own overhead)."""
    iters = 1000
    while iters < 1 << 28:
        _spin(out, iters)
        tack.sync()
        t0 = time.perf_counter()
        _spin(out, iters)
        tack.sync()
        if time.perf_counter() - t0 >= 0.15:
            return iters
        iters *= 4
    return None


def _long_enough(run, iters, attempts=4):
    """``run(iters)``, which returns a result and the waits it measured,
    with iterations doubled until every wait lasts `_MIN_WAIT`. Other processes
    on the device (pytest-xdist workers) can slow the calibration far more
    than the run: on Level Zero with four workers a launch calibrated at
    300 ms later ran in 44. They only ever lengthen a wait, so a retry
    settles it."""
    for _ in range(attempts):
        result, waits = run(iters)
        if min(waits) >= _MIN_WAIT:
            return result
        iters *= 2
    pytest.fail(f"waits under {_MIN_WAIT * 1000:.0f} ms after {attempts} doublings: {waits}")


@pytest.mark.usefixtures("no_debug")
@pytest.mark.parametrize("wait", ["sync_async", "numpy_async"])
def test_the_loop_runs_while_the_device_works(backend, wait):
    """Another task keeps running during the wait -- and, as a control, does
    not during a plain `tack.sync` in the same coroutine."""
    if not _queues(backend):
        pytest.skip(f"{backend} launches complete before returning")
    out = tack.empty((1 << 18,))
    iters = _long_launch(out)
    if iters is None:
        pytest.skip("no launch long enough to measure here")

    async def measure(block, iters):
        ticks = 0
        stop = asyncio.Event()

        async def ticker():
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.001)

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0.005)
        before = ticks
        _spin(out, iters)
        t0 = time.perf_counter()
        if block:
            tack.sync()
        elif wait == "sync_async":
            await tack.sync_async()
        else:
            values = await out.numpy_async()
            assert values.shape == (1 << 18,) and np.isfinite(values).all()
        elapsed = time.perf_counter() - t0
        stop.set()
        await task
        return ticks - before, elapsed

    def both(iters):
        blocked = asyncio.run(measure(True, iters))
        free = asyncio.run(measure(False, iters))
        return (blocked, free), (blocked[1], free[1])

    (blocked_ticks, blocked_s), (free_ticks, free_s) = _long_enough(both, iters)
    # Ticks are 1 ms sleeps; the free wait should see many of them, the
    # blocked one almost none. On a loaded host a 1 ms sleep takes 2-4 ms
    # (Level Zero, load 115, four workers: 50 ticks in 207 ms), and up to
    # ~10 ms (HIP on illyad, load 120-185).
    assert blocked_ticks <= 2, blocked_ticks
    # 0.05 per ms keeps the free wait clear of the blocked one's 2 at 100 ms
    # (0.01 at 400 ms on Windows).
    assert free_ticks >= free_s * 1000 * _TICKS_PER_MS, (free_ticks, free_s)


@pytest.mark.usefixtures("no_debug")
def test_the_loop_launches_while_a_read_waits(backend):
    """A task that launches keeps launching while another awaits a read:
    the worker's wait holds no lock a launch needs (Level Zero's reads wait
    on an event outside the backend's lock)."""
    if not _queues(backend):
        pytest.skip(f"{backend} launches complete before returning")
    out = tack.empty((1 << 18,))
    short = tack.empty((16,))
    iters = _long_launch(out)
    if iters is None:
        pytest.skip("no launch long enough to measure here")
    _ramp(short, 1.0)                        # compiled before, not on the loop
    tack.sync()

    async def main(iters):
        launched = 0
        stop = asyncio.Event()

        async def launcher():
            nonlocal launched
            while not stop.is_set():
                _ramp(short, 1.0)
                launched += 1
                await asyncio.sleep(0.001)

        _spin(out, iters)
        task = asyncio.create_task(launcher())
        t0 = time.perf_counter()
        await out.numpy_async()
        elapsed = time.perf_counter() - t0
        stop.set()
        await task
        return launched, elapsed

    def once(iters):
        launched, elapsed = asyncio.run(main(iters))
        return (launched, elapsed), (elapsed,)

    launched, elapsed = _long_enough(once, iters)
    # At least one per 20 ms of a 1 ms cadence: a loaded host's 1 ms sleep
    # takes ~10 ms (illyad at load 120-185 reached 0.09 per ms), and a wait
    # that blocks launches allows about one per wait (0.003 per ms). Windows
    # asks for one per 100 ms (`_TICKS_PER_MS`).
    assert launched >= elapsed * 1000 * _TICKS_PER_MS, (launched, elapsed)


@pytest.mark.usefixtures("no_debug")
def test_sync_async_does_not_wait_for_later_launches(backend):
    """The mark is made when sync_async is called: a long launch another
    task queues while it waits is not waited for."""
    if not _queues(backend):
        pytest.skip(f"{backend} launches complete before returning")
    out = tack.empty((1 << 18,))
    short = tack.empty((16,))
    iters = _long_launch(out)
    if iters is None:
        pytest.skip("no launch long enough to measure here")

    async def main():
        _ramp(short, 1.0)
        waiter = asyncio.create_task(tack.sync_async())
        await asyncio.sleep(0)               # the waiter runs up to its await: marked
        t0 = time.perf_counter()
        _spin(out, iters)                    # queued after the mark
        await waiter
        early = time.perf_counter() - t0
        await tack.sync_async()
        return early, time.perf_counter() - t0

    early, whole = asyncio.run(main())
    assert early < 0.5 * whole, (early, whole)


def test_a_launch_error_is_raised_at_the_await(backend, monkeypatch):
    """Whatever the marker's wait raises reaches the awaiting coroutine."""
    class Failing:
        def wait(self):
            raise RuntimeError("device error from the marker")

    monkeypatch.setattr(type(get_backend()), "async_marker", lambda self: Failing())
    with pytest.raises(RuntimeError, match="device error from the marker"):
        asyncio.run(tack.sync_async())


def test_without_a_marker_the_wait_is_inline(backend, monkeypatch):
    """A backend with no marker -- the CPU, or one that has not added it --
    waits on the calling thread and never starts a worker."""
    monkeypatch.setattr(type(get_backend()), "async_marker", lambda self: None)
    calls = []
    real = type(get_backend()).synchronize
    monkeypatch.setattr(type(get_backend()), "synchronize",
                        lambda self: calls.append(1) or real(self))
    monkeypatch.setattr(aio, "_workers", lambda: pytest.fail("a worker was started"))
    out = tack.empty((100,))
    _ramp(out, 1.0)
    asyncio.run(aio.sync_async(get_backend()))
    assert calls == [1]
    np.testing.assert_array_equal(out.numpy(), np.arange(100, dtype=np.float32))


def test_sync_async_waits_for_every_device(monkeypatch):
    """As `tack.sync` does: every device's mark is taken at the call and
    waited for, and a failed one is raised after the others are waited for."""
    tack.set_default_device("cpu")
    serial = tack.device("cpu", num_threads=1)
    waited = []

    class Marker:
        def __init__(self, device, error=None):
            self.device, self.error = device, error

        def wait(self):
            waited.append(self.device)
            if self.error:
                raise RuntimeError(self.error)

    devices = (tack.device("cpu"), serial)
    for device in devices:
        # In the instance's dict, so the undo removes it: setattr's would
        # leave the bound class method on the registry's backend, shadowing
        # later patches of the class.
        monkeypatch.setitem(vars(device._backend), "async_marker",
                            lambda d=device: Marker(d))
    asyncio.run(tack.sync_async())
    assert set(devices) <= set(waited)

    waited.clear()
    monkeypatch.setitem(vars(serial._backend), "async_marker",
                        lambda: Marker(serial, "device error on serial"))
    with pytest.raises(RuntimeError, match="device error on serial"):
        asyncio.run(tack.sync_async())
    assert set(devices) <= set(waited)


def test_the_cpu_reads_inline():
    tack.set_default_device("cpu")
    assert get_backend().async_marker() is None
    out = tack.empty((10,))
    _ramp(out, 3.0)
    assert not out._buffer.waits_off_thread
    np.testing.assert_array_equal(asyncio.run(out.numpy_async()),
                                  np.arange(10, dtype=np.float32) * 3)


def test_waits_from_several_loops_and_threads(backend):
    """Each asyncio.run is a new loop; the workers are Tack's and outlive
    them. Two threads with their own loops wait at once."""
    import threading
    results = {}
    if backend == "cuda":
        from cuda.bindings import driver
        context = get_backend()._token.handle

    def run(name, scale):
        if backend == "cuda":
            driver.cuCtxSetCurrent(context)   # a CUDA launch needs the context current

        async def main():
            out = tack.empty((256,))
            _ramp(out, scale)
            await tack.sync_async()
            return await out.numpy_async()
        results[name] = asyncio.run(main())

    for _ in range(3):
        run("main", 1.0)
    threads = [threading.Thread(target=run, args=(f"t{k}", float(k))) for k in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    np.testing.assert_array_equal(results["main"], np.arange(256, dtype=np.float32))
    for k in range(2):
        np.testing.assert_array_equal(results[f"t{k}"], np.arange(256, dtype=np.float32) * k)

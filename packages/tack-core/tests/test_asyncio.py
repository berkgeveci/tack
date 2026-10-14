"""`tack.sync_async` and `Field.to_numpy_async`: waiting for the device from
``asyncio`` without blocking the event loop (`tack.runtime.aio`)."""
import asyncio
import time

import numpy as np
import pytest

import tack
from tack.runtime import aio
from tack.runtime.dispatch import get_backend

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


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
    out = tack.field(tack.f32, (1000,))

    async def main():
        _ramp(out, 2.0)
        await tack.sync_async()
        return out.to_numpy()

    np.testing.assert_array_equal(asyncio.run(main()), np.arange(1000, dtype=np.float32) * 2)


def test_to_numpy_async_reads_what_was_queued(backend):
    fields = [tack.field(tack.f32, (500,)) for _ in range(4)]
    vec = tack.Vector.field(3, tack.f32, shape=(7,))
    vec.from_numpy(np.arange(21, dtype=np.float32))

    async def main():
        for k, f in enumerate(fields):
            _ramp(f, float(k + 1))
        flat, rows = await asyncio.gather(vec.to_numpy_async(), vec.to_numpy_async(vectors=True))
        return await asyncio.gather(*(f.to_numpy_async() for f in fields)), flat, rows

    got, flat, rows = asyncio.run(main())
    for k, values in enumerate(got):
        np.testing.assert_array_equal(values, np.arange(500, dtype=np.float32) * (k + 1))
    assert flat.shape == (21,) and rows.shape == (7, 3)
    np.testing.assert_array_equal(rows, np.arange(21, dtype=np.float32).reshape(7, 3))


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


@pytest.mark.parametrize("wait", ["sync_async", "to_numpy_async"])
def test_the_loop_runs_while_the_device_works(backend, wait):
    """Another task keeps running during the wait -- and, as a control, does
    not during a plain `tack.sync` in the same coroutine."""
    if not _queues(backend):
        pytest.skip(f"{backend} launches complete before returning")
    out = tack.field(tack.f32, (1 << 18,))
    iters = _long_launch(out)
    if iters is None:
        pytest.skip("no launch long enough to measure here")

    async def measure(block):
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
            values = await out.to_numpy_async()
            assert values.shape == (1 << 18,) and np.isfinite(values).all()
        elapsed = time.perf_counter() - t0
        stop.set()
        await task
        return ticks - before, elapsed

    blocked_ticks, blocked_s = asyncio.run(measure(block=True))
    free_ticks, free_s = asyncio.run(measure(block=False))
    assert blocked_s >= 0.1 and free_s >= 0.1, (blocked_s, free_s)
    # Ticks are 1 ms sleeps; the free wait should see many of them, the
    # blocked one almost none. On a loaded host a 1 ms sleep takes 2-4 ms
    # (Level Zero, load 115, four workers: 50 ticks in 207 ms).
    assert blocked_ticks <= 2, blocked_ticks
    assert free_ticks >= free_s * 1000 * 0.1, (free_ticks, free_s)


def test_sync_async_does_not_wait_for_later_launches(backend):
    """The mark is made when sync_async is called: a long launch another
    task queues while it waits is not waited for."""
    if not _queues(backend):
        pytest.skip(f"{backend} launches complete before returning")
    out = tack.field(tack.f32, (1 << 18,))
    short = tack.field(tack.f32, (16,))
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
    out = tack.field(tack.f32, (100,))
    _ramp(out, 1.0)
    asyncio.run(tack.sync_async())
    assert calls == [1]
    np.testing.assert_array_equal(out.to_numpy(), np.arange(100, dtype=np.float32))


def test_the_cpu_reads_inline():
    tack.init(arch=tack.cpu)
    assert get_backend().async_marker() is None
    out = tack.field(tack.f32, (10,))
    _ramp(out, 3.0)
    assert not out._buffer.waits_off_thread
    np.testing.assert_array_equal(asyncio.run(out.to_numpy_async()),
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
            out = tack.field(tack.f32, (256,))
            _ramp(out, scale)
            await tack.sync_async()
            return await out.to_numpy_async()
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

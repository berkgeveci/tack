"""Waiting for the device from ``asyncio`` without blocking the event loop.

Kernel launches stay plain calls: on CUDA and Metal they are queued and
return at once, and every host access to a field waits for the launches
before it. Those waits block the calling thread, which in an application
with an event loop (a trame server, a Jupyter kernel, a web service) is
the loop. The two coroutines here wait in a worker thread instead and
resume the caller when the device is done:

    await tack.sync_async()               # the launches queued so far
    values = await field.to_numpy_async() # wait, then copy to the host

A backend opts in with two hooks on `Backend` (`async_marker`) and on
`DeviceBuffer` (`thread_binding`, `waits_off_thread`): CUDA records an event
on its stream and waits on it; Metal waits on the last command buffer it
committed. A backend without them (the CPU, whose launches complete before
returning, and any backend that has not added them) waits inline, which
is correct and simply does not free the loop.

The workers belong to Tack, not to the loop's default executor, so a busy
or small default executor cannot hold up a wait. Python's GIL is released
while a worker waits in the driver (measured with cuda-bindings 13.3: the
loop ran throughout a 230 ms wait).
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

_executor: ThreadPoolExecutor | None = None
_executor_lock = threading.Lock()
# Waits that block on long device work each hold a worker; more than this
# many at once queue behind them, in order.
_WORKERS = 4


def _workers() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = ThreadPoolExecutor(max_workers=_WORKERS,
                                               thread_name_prefix="tack-wait")
    return _executor


async def _off_loop(fn):
    import asyncio
    return await asyncio.get_running_loop().run_in_executor(_workers(), fn)


async def sync_async(backend=None):
    """Wait, without blocking the event loop, until every kernel launched so
    far on ``backend`` (the active one by default) has finished.

    The ``asyncio`` form of `tack.sync`: launches made after the call, by
    this task or another, are not waited for. A launch's error is raised
    here, as `tack.sync` raises it. On a backend whose launches complete
    before returning (the CPU) it returns at once.
    """
    if backend is None:
        from tack.runtime.dispatch import _current_backend as backend
        if backend is None:
            return
    marker = backend.async_marker()
    if marker is None:
        backend.synchronize()
        return
    await _off_loop(marker.wait)


async def to_numpy_async(field, vectors: bool = False):
    """`Field.to_numpy` as a coroutine: the copy, and the wait for the
    launches before it, run in a worker thread while the loop goes on.

    The result holds every write queued before the call. Work queued later
    by another task may or may not be in it, as with any read that runs
    beside writes; order the two with ``await`` if it matters.
    """
    buffer = field._buffer
    if not buffer.waits_off_thread:
        return field.to_numpy(vectors=vectors)

    def read():
        with buffer.thread_binding():
            return field.to_numpy(vectors=vectors)

    return await _off_loop(read)

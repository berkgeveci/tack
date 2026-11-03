# Wait without blocking: asyncio

A program built on an event loop, such as a trame or web server or a Jupyter
kernel, must not block the loop while a GPU works, or every other request
waits too. Tack can wait for the device and read results back while the loop
keeps running. Not every device waits off the loop, and the page says which
ones do.

Two coroutines do the waiting in a worker thread, so the loop keeps running.
`await tack.sync_async()` waits for the kernels launched so far, and
`await array.numpy_async()` waits and then copies an array to NumPy:

<!-- tack-check: asyncio-first -->
```python
import asyncio

import numpy as np
import tack

tack.set_default_device("cpu")     # "cuda", "metal", "hip" or "level_zero" on a GPU


@tack.kernel
def step(field, out, decay):
    for i in range(field.shape[0]):
        out[i] = field[i] * decay


async def frame(field, out):
    step(field, out, 0.5)                  # a plain call: queued on a GPU
    await tack.sync_async()                # the loop runs while the device works
    return await out.numpy_async()         # wait and copy, off the loop


async def heartbeat(ticks):
    for _ in range(3):
        ticks.append("tick")               # other work the loop keeps doing
        await asyncio.sleep(0)


async def main():
    field = tack.full(1000, 2.0)
    out = tack.empty_like(field)
    ticks = []
    values, _ = await asyncio.gather(frame(field, out), heartbeat(ticks))
    return values, ticks


values, ticks = asyncio.run(main())
np.testing.assert_allclose(values, 1.0)
assert ticks == ["tick"] * 3
```

## Why a read blocks the loop

On Metal, CUDA, HIP and Level Zero a kernel call returns as soon as its launch
is queued. The host waits only when it needs a result: when it reads an array
with `numpy()`, indexes an element, reduces an array, or calls `tack.sync()`.
[Choose a device](choose-a-device.md#know-when-a-call-returns) explains this
queue.

Those waits block the thread that makes them. In an `asyncio` program that
thread runs the event loop, so a 200 ms kernel followed by `out.numpy()`
freezes the loop for 200 ms. The two coroutines move the wait to a thread of
Tack's own, and resume your coroutine when the device is done.

Kernel calls stay plain calls. Queuing a launch takes microseconds, so there
is nothing to await, and you call kernels from a coroutine as from any other
code.

## Wait for the device

`await tack.sync_async()` is the `asyncio` form of `tack.sync()`. It waits
until every kernel launched before the call, on every device, has finished.

- **Only earlier launches count.** The point to wait for is fixed when you
  call it. Launches that this task or another makes while it waits are not
  waited for.
- **Errors arrive here.** An error inside a queued launch is raised at the
  `await`, as `tack.sync()` would raise it. With several devices, Tack first
  waits for all of them, then raises the first error.

## Read an array without blocking

`await a.numpy_async()` is the `asyncio` form of `a.numpy()`. The wait for
earlier launches and the copy both run in the worker thread. The result is a
new NumPy array, as from `numpy()`.

The result holds every write queued before the call. A write that another
task queues while the read is in flight may or may not be in it, as with any
read that runs beside writes. When one task needs another's result, order
them with `await`: let the writer finish its launches, then start the read.

## Know which devices wait off the loop

| device | where the wait runs |
|---|---|
| CUDA, HIP, Metal, Level Zero | in a Tack worker thread; the loop keeps running |
| CPU | inline, and at once: CPU launches finish before the call returns |
| Warp (prototype) | inline: the loop waits |

The same code is correct on every device. On the CPU there is nothing to
wait for, so the coroutines return as soon as the copy is made. On Warp they
block, as `tack.sync()` and `numpy()` would.

## Details

- **Tack's worker threads.** The waits run in a pool of four threads that
  belong to Tack, named `tack-wait`, not in the loop's default executor. A
  busy or small default executor therefore cannot hold up a wait. More than
  four long waits at once queue behind each other, in order.
- **The interpreter lock.** A worker thread releases Python's global
  interpreter lock while it waits in the driver, so the loop's other tasks
  run meanwhile. On CUDA, during a 490 ms kernel another task ran 436 times
  while one awaited `sync_async`, and not at all under `tack.sync()`. The
  [backend implementations note](../design/backend-implementations.md) has
  the measurements for each device.
- **GPU contexts.** A worker thread makes the array's CUDA context or HIP
  device current itself, so you need no setup in the worker.
- **Blocking calls still work.** `tack.sync()` and `numpy()` remain correct
  inside a coroutine. They just hold the loop while they wait.

## Where to go next

- [Choose a device](choose-a-device.md), for how launches are queued and
  when the host waits.
- [Share arrays](share-arrays.md), to hand results to another library
  without a copy.
- The [runtime API](../contracts/runtime-api.md#tacksync), for the exact
  behaviour of `tack.sync` and its `asyncio` form.

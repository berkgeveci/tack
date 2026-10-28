"""An allocation that runs out of device memory collects Python's garbage and tries
once more (``lang.array._allocate``): a dropped array held in a reference cycle
keeps its memory until the cycle collector runs, and that collector counts
objects, not device memory. A 16 GB GPU ran out with dropped meshes still on it
(a topology's cached locator holds a dataset of it)."""

import gc
import weakref

import numpy as np
import pytest

import tack
from tack.lang.array import NumpyBuffer
from tack.runtime.backend import DeviceMemoryError
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.quick


class _Budget(CPUBackend):
    """A CPU backend with a memory budget, as a device has: an allocation past it
    raises DeviceMemoryError, and a buffer gives its bytes back when freed."""

    def __init__(self, budget):
        super().__init__()
        self.budget = budget
        self.live = 0
        self.attempts = 0

    def allocate_array(self, dtype, shape, exportable=False):
        self.attempts += 1
        nbytes = int(np.prod(shape)) * dtype.numpy_dtype.itemsize
        if self.live + nbytes > self.budget:
            raise DeviceMemoryError(f"over budget by {self.live + nbytes - self.budget} bytes")
        buffer = NumpyBuffer(dtype.numpy_dtype, shape)
        self.live += nbytes
        weakref.finalize(buffer, self._give_back, nbytes)
        return buffer

    def _give_back(self, nbytes):
        self.live -= nbytes


class _Cycle:
    def __init__(self, array):
        self.array = array
        self.me = self


def test_an_allocation_collects_a_dropped_cycle_and_tries_again():
    backend = _Budget(budget=8 << 20)
    enabled = gc.isenabled()
    gc.disable()
    try:
        held = _Cycle(tack.empty((1 << 20,), tack.f32, device=backend))     # 4 MiB
        del held                                                            # in a cycle: not freed
        assert backend.live == 4 << 20
        backend.attempts = 0
        out = tack.empty((1 << 20,), tack.f64, device=backend)              # 8 MiB: over budget
        assert backend.attempts == 2                                        # refused, then collected
        assert out.shape == (1 << 20,) and backend.live == 8 << 20
    finally:
        if enabled:
            gc.enable()


def test_memory_still_short_after_collecting_raises():
    backend = _Budget(budget=1 << 20)
    with pytest.raises(MemoryError, match="over budget"):
        tack.empty((1 << 20,), tack.f32, device=backend)
    assert backend.attempts == 2


def test_the_retry_frees_what_a_cache_holds():
    """Collecting returns a cycle's buffers to a backend's cache (HIP's, Metal's),
    where they still hold device memory: the retry frees them first."""
    class Cached(_Budget):
        released = 0

        def release_memory(self):
            self.released += 1

    backend = Cached(budget=1 << 20)
    with pytest.raises(MemoryError):
        tack.empty((1 << 20,), tack.f32, device=backend)
    assert backend.released == 1


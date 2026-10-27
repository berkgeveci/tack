"""Device memory of dropped arrays, kept for the next arrays of its size.

``BufferCache`` holds addresses by capacity: an allocation rounds its size up
to a capacity (a quarter of its power of two, so a reused buffer wastes at most
a quarter), takes a held buffer of that capacity if there is one, and a dropped
array's buffer goes back instead of to the driver. Past ``limit`` bytes held,
the oldest are freed, so the sizes a program uses now stay.

HIP allocates through one (``hip_backend._Queue.cache``): ROCm 7.1.1's
stream-ordered pool slowed with every allocation it had made (``hipMallocAsync``
from 63 to 600 µs a call over the eta pieces' global ids, 2.7 ms when trimmed
after each step), and stalled the 512 eta pieces for two hours; plain
``hipMalloc`` stayed at 17 µs, but each ``hipFree`` waits for the device. A
backend uses a cache only where every use of a buffer is queued on one
in-order stream per device: the dropped array's queued work then runs before
anything queued on the buffer's next array, as a stream-ordered pool
promises. Memory handed outside Tack is never given back."""

import threading
from collections import OrderedDict

__all__ = ["BufferCache"]


class BufferCache:
    """Buffers of dropped arrays by capacity; ``free(address)`` returns one to the
    device when it is evicted or the cache is cleared."""

    #: The smallest capacity: smaller requests share it.
    MIN = 256

    def __init__(self, limit, free):
        self.limit = limit
        self.held = 0
        self._free = free
        self._by_capacity = {}            # capacity -> tokens, the latest last
        self._entries = OrderedDict()     # token -> (capacity, address), the oldest first
        self._next = 0
        self._lock = threading.Lock()

    @classmethod
    def capacity(cls, nbytes):
        """The class ``nbytes`` falls in: rounded up to a quarter of its power of
        two, at least ``MIN``."""
        nbytes = max(int(nbytes), cls.MIN)
        step = max(1 << (nbytes.bit_length() - 1) >> 2, 1)
        return -(-nbytes // step) * step

    def take(self, capacity):
        """A held buffer of ``capacity`` bytes, or ``None``."""
        with self._lock:
            tokens = self._by_capacity.get(capacity)
            if not tokens:
                return None
            _, address = self._entries.pop(tokens.pop())
            self.held -= capacity
            return address

    def give(self, address, capacity):
        """Keep the buffer at ``address``; past the limit, free the oldest held
        (one larger than the limit is freed at once)."""
        evicted = []
        with self._lock:
            if capacity > self.limit:
                evicted.append(address)
            else:
                token, self._next = self._next, self._next + 1
                self._entries[token] = (capacity, address)
                self._by_capacity.setdefault(capacity, []).append(token)
                self.held += capacity
                while self.held > self.limit:
                    oldest, (size, old) = self._entries.popitem(last=False)
                    self._by_capacity[size].remove(oldest)
                    self.held -= size
                    evicted.append(old)
        for old in evicted:                     # outside the lock: a free may wait
            self._free(old)

    def clear(self):
        """Free every held buffer (an allocation that ran out tries again after)."""
        with self._lock:
            addresses = [address for _, address in self._entries.values()]
            self._by_capacity.clear()
            self._entries.clear()
            self.held = 0
        for address in addresses:
            self._free(address)

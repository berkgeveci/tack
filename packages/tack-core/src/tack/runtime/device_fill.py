"""Filling device memory on the device.

``Field.fill`` and ``tack.zeros``/``ones``/``full`` on CUDA, HIP and Level Zero
used to build a host array the size of the field and upload it. On a 5M-cell
contour that was 24 ms of 40, more than every kernel together. A fill now
stays on the device: the backend's native fill where the pattern allows one
(a byte memset, CUDA's and HIP's 16- and 32-bit memsets, Level Zero's pattern
fill), otherwise one element written from the host and doubled by
device-to-device copies, a few dozen copies for any field size.
"""

import numpy as np


def scalar_pattern(value, numpy_dtype):
    """The bytes of one element holding ``value``, converted as ``np.full`` does."""
    return np.full(1, value, dtype=numpy_dtype).tobytes()


def element_pattern(element, numpy_dtype):
    """The bytes of one vector or matrix element, converted as ``from_numpy`` does."""
    return np.ascontiguousarray(element, dtype=numpy_dtype).tobytes()


def fill(nbytes, pattern, native, write, copy):
    """Fill ``nbytes`` of device memory with ``pattern`` repeated.

    ``native(pattern)`` fills the whole allocation and returns true, or returns
    false when the backend has no fill for this pattern. ``write(offset, data)``
    copies host bytes to the allocation at ``offset``; ``copy(dst, src, size)``
    copies within it. ``nbytes`` must be a multiple of the pattern's length.
    """
    if nbytes == 0:
        return
    size = len(pattern)
    if nbytes % size:
        raise ValueError(f"a {size}-byte pattern does not divide {nbytes} bytes")
    if native(pattern):
        return
    write(0, pattern)
    filled = size
    while filled < nbytes:
        step = min(filled, nbytes - filled)
        copy(filled, 0, step)
        filled += step


def uniform_byte(pattern):
    """The byte every byte of ``pattern`` is, or None."""
    return pattern[0] if pattern.count(pattern[:1]) == len(pattern) else None


def as_unsigned(pattern):
    """A 2- or 4-byte pattern as the unsigned integer memset16/32 writes."""
    return int(np.frombuffer(pattern, dtype=np.uint16 if len(pattern) == 2 else np.uint32)[0])


def as_signed(pattern):
    """A 4-byte pattern as the signed integer HIP's memset32 takes.

    ``hipMemsetD32`` declares its value an ``int``, and hip-python refuses
    anything past 2**31 - 1 instead of wrapping it, so -1.0 (0xBF800000) or a
    -1 fill would not convert as an unsigned value.
    """
    return int(np.frombuffer(pattern, dtype=np.int32)[0])

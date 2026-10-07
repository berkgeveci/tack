"""Level Zero buffers: what only an L0Buffer does."""

import numpy as np
import pytest

import tack


@pytest.fixture(autouse=True)
def level_zero_backend():
    """Initialize the Level Zero backend for all tests in this module."""
    try:
        tack.init(arch=tack.level_zero)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Level Zero not available: {e}")


def test_host_element_reads_copy_one_element():
    """field[i] copies just the element on Level Zero, from an owned buffer,
    a wrapped pointer in each form, a view and a vector field; a range past
    the allocation is refused rather than read."""
    f = tack.field(dtype=tack.f64, shape=(4, 5))
    f.from_numpy(np.arange(20, dtype=np.float64).reshape(4, 5))
    f._buffer.to_numpy = None          # a whole-buffer copy would fail here
    assert f[3, 4] == 19.0 and f[-4, 0] == 0.0
    assert f.reshape((20,))[7] == 7.0

    # USM device addresses here lie past 2^63, so only the unsigned forms.
    address = f._buffer.address
    for ptr in (address, np.uint64(address)):
        alias = tack.field_from_ptr(ptr, tack.f64, (20,))
        alias._buffer.to_numpy = None
        assert alias[13] == 13.0

    v = tack.Vector.field(3, dtype=tack.i32, shape=(6,))
    v.from_numpy(np.arange(18, dtype=np.int32).reshape(6, 3))
    v._buffer.to_numpy = None
    np.testing.assert_array_equal(v[5], [15, 16, 17])

    with pytest.raises(IndexError, match="outside a buffer of 20 elements"):
        f._buffer.read_range(19, 2)
    with pytest.raises(IndexError, match="outside"):
        f._buffer.read_range(-1, 1)


def test_a_host_read_sees_the_last_kernel_write():
    """Dispatch waits for its queue, and the read is a synchronous copy on
    the immediate list, so each read sees the kernel just launched."""
    f = tack.field(dtype=tack.i32, shape=(1024,))

    @tack.kernel
    def mark(out, value):
        for i in range(out.shape[0]):
            out[i] = value + i

    for value in (7, 70, 700):
        mark(f, value)
        assert f[1023] == value + 1023

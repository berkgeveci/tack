"""Tests for tack.algorithms module."""

import numpy as np

import tack
from tack import algorithms


def test_exclusive_scan_basic(backend):
    """Exclusive scan: output[i] = sum(input[0..i-1])."""
    n = 8
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    inp.from_numpy(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    result = out.to_numpy()
    expected = np.array([0, 3, 4, 8, 9, 14, 23, 25], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert total == 31  # sum of all elements


def test_exclusive_scan_ones(backend):
    """Exclusive scan of all ones = [0, 1, 2, ..., n-1]."""
    n = 100
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    inp.from_numpy(np.ones(n, dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    result = out.to_numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.int32))
    assert total == n


def test_exclusive_scan_zeros(backend):
    """Exclusive scan of all zeros stays zero."""
    n = 16
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    inp.from_numpy(np.zeros(n, dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    np.testing.assert_array_equal(out.to_numpy(), np.zeros(n, dtype=np.int32))
    assert total == 0


def test_exclusive_scan_power_of_two(backend):
    """Scan with power-of-two size."""
    n = 16
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    data = np.arange(1, n + 1, dtype=np.int32)
    inp.from_numpy(data)

    total = algorithms.exclusive_scan(inp, out, n)

    expected = np.zeros(n, dtype=np.int32)
    expected[1:] = np.cumsum(data[:-1])
    np.testing.assert_array_equal(out.to_numpy(), expected)
    assert total == int(np.sum(data))


def test_exclusive_scan_non_power_of_two(backend):
    """Scan with non-power-of-two size."""
    n = 37
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    data = np.arange(1, n + 1, dtype=np.int32)
    inp.from_numpy(data)

    total = algorithms.exclusive_scan(inp, out, n)

    expected = np.zeros(n, dtype=np.int32)
    expected[1:] = np.cumsum(data[:-1])
    np.testing.assert_array_equal(out.to_numpy(), expected)
    assert total == int(np.sum(data))


def test_inclusive_scan_basic(backend):
    """Inclusive scan: output[i] = sum(input[0..i])."""
    n = 8
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    inp.from_numpy(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))

    total = algorithms.inclusive_scan(inp, out, n)

    result = out.to_numpy()
    expected = np.array([3, 4, 8, 9, 14, 23, 25, 31], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert total == 31


def test_inclusive_scan_ones(backend):
    """Inclusive scan of all ones = [1, 2, 3, ..., n]."""
    n = 64
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    inp.from_numpy(np.ones(n, dtype=np.int32))

    total = algorithms.inclusive_scan(inp, out, n)

    np.testing.assert_array_equal(out.to_numpy(), np.arange(1, n + 1, dtype=np.int32))
    assert total == n


def test_exclusive_scan_preserves_input(backend):
    """Exclusive scan should not modify the input field."""
    n = 16
    inp = tack.field(dtype=tack.i32, shape=(n,))
    out = tack.field(dtype=tack.i32, shape=(n,))
    data = np.arange(n, dtype=np.int32)
    inp.from_numpy(data)

    algorithms.exclusive_scan(inp, out, n)

    np.testing.assert_array_equal(inp.to_numpy(), data)


def test_copy(backend):
    """Copy field contents."""
    n = 100
    src = tack.field(dtype=tack.f32, shape=(n,))
    dst = tack.field(dtype=tack.f32, shape=(n,))
    data = np.random.randn(n).astype(np.float32)
    src.from_numpy(data)

    algorithms.copy(src, dst, n)

    np.testing.assert_array_equal(dst.to_numpy(), data)


def test_fill_value(backend):
    """Fill field with constant value."""
    n = 100
    dst = tack.field(dtype=tack.f32, shape=(n,))

    algorithms.fill_value(dst, 42.0, n)

    np.testing.assert_array_equal(dst.to_numpy(), np.full(n, 42.0, dtype=np.float32))

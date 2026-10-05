"""Tests for tack.algorithms module."""

import numpy as np
import pytest

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


def test_scans_of_no_elements_return_zero_and_write_nothing(backend):
    """n=0 used to read index -1: garbage or a launch error on CUDA."""
    for scan in (algorithms.exclusive_scan, algorithms.inclusive_scan):
        inp = tack.field(dtype=tack.i32, shape=(4,))
        out = tack.field(dtype=tack.i32, shape=(4,))
        inp.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
        out.from_numpy(np.full(4, -7, dtype=np.int32))
        assert scan(inp, out, 0) == 0
        np.testing.assert_array_equal(out.to_numpy(), np.full(4, -7))
        with pytest.raises(ValueError, match="outside"):
            scan(inp, out, 5)
        with pytest.raises(ValueError, match="outside"):
            scan(inp, out, -1)


def test_scans_refuse_an_output_shorter_than_n(backend):
    inp = tack.field(dtype=tack.i32, shape=(8,))
    out = tack.field(dtype=tack.i32, shape=(4,))
    inp.fill(1)
    for scan in (algorithms.exclusive_scan, algorithms.inclusive_scan):
        with pytest.raises(ValueError, match="outside"):
            scan(inp, out, 8)


# --- scans run in the output field's dtype ---

def _np_scans(a):
    inclusive = np.cumsum(a, dtype=a.dtype)
    exclusive = np.concatenate([np.zeros(1, a.dtype), inclusive[:-1]])
    return exclusive, inclusive



@pytest.mark.parametrize("dt, values", [
    ("u32", [4_000_000_000, 1, 2, 3, 5]),
    ("i64", [2**40, 1, -2, 3, 2**41]),
    ("u8", [200, 100, 7, 1, 0]),
    ("f32", [0.5, 1.25, 2.0, 3.5, -0.75]),
], ids=["u32", "i64", "u8", "f32"])
def test_scans_keep_the_output_dtype(backend, dt, values):
    """exclusive_scan used to scan in i32 (floats truncated, i64 wrapped), and
    both returned the total through i32."""
    a = np.array(values, dtype=getattr(np, {"u32": "uint32", "i64": "int64", "u8": "uint8",
                                             "f32": "float32"}[dt]))
    exclusive, inclusive = _np_scans(a)
    for scan, expected in ((algorithms.exclusive_scan, exclusive),
                           (algorithms.inclusive_scan, inclusive)):
        inp = tack.field(dtype=getattr(tack, dt), shape=a.shape)
        out = tack.field(dtype=getattr(tack, dt), shape=a.shape)
        inp.from_numpy(a)
        total = scan(inp, out, a.size)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert total == inclusive[-1].item()
        assert type(total) is type(inclusive[-1].item())


def test_scans_keep_f64(f64_backend):
    a = np.array([0.1, 0.2, 0.3, 0.4, 1e-17, 2.5], dtype=np.float64)
    exclusive, inclusive = _np_scans(a)
    for scan, expected in ((algorithms.exclusive_scan, exclusive),
                           (algorithms.inclusive_scan, inclusive)):
        inp = tack.field(dtype=tack.f64, shape=a.shape)
        out = tack.field(dtype=tack.f64, shape=a.shape)
        inp.from_numpy(a)
        total = scan(inp, out, a.size)
        # The tree adds in a different order from cumsum: equal to rounding.
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-15, atol=1e-17)
        assert total == pytest.approx(a.sum(), rel=1e-15)
        assert isinstance(total, float)


def test_scans_convert_the_input_to_the_output_dtype(backend):
    inp = tack.field(dtype=tack.i32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    inp.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))
    assert algorithms.exclusive_scan(inp, out, 4) == 10.0
    np.testing.assert_array_equal(out.to_numpy(), [0.0, 1.0, 3.0, 6.0])

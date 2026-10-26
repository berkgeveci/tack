"""Tests for tack.algorithms module."""

import numpy as np
import pytest

import tack
from tack import algorithms

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def test_exclusive_scan_basic(backend):
    """Exclusive scan: output[i] = sum(input[0..i-1])."""
    n = 8
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(inp, np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    result = out.numpy()
    expected = np.array([0, 3, 4, 8, 9, 14, 23, 25], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert total == 31  # sum of all elements


def test_exclusive_scan_ones(backend):
    """Exclusive scan of all ones = [0, 1, 2, ..., n-1]."""
    n = 100
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(inp, np.ones(n, dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.int32))
    assert total == n


def test_exclusive_scan_zeros(backend):
    """Exclusive scan of all zeros stays zero."""
    n = 16
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(inp, np.zeros(n, dtype=np.int32))

    total = algorithms.exclusive_scan(inp, out, n)

    np.testing.assert_array_equal(out.numpy(), np.zeros(n, dtype=np.int32))
    assert total == 0


def test_exclusive_scan_power_of_two(backend):
    """Scan with power-of-two size."""
    n = 16
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    data = np.arange(1, n + 1, dtype=np.int32)
    tack.copyto(inp, data)

    total = algorithms.exclusive_scan(inp, out, n)

    expected = np.zeros(n, dtype=np.int32)
    expected[1:] = np.cumsum(data[:-1])
    np.testing.assert_array_equal(out.numpy(), expected)
    assert total == int(np.sum(data))


def test_exclusive_scan_non_power_of_two(backend):
    """Scan with non-power-of-two size."""
    n = 37
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    data = np.arange(1, n + 1, dtype=np.int32)
    tack.copyto(inp, data)

    total = algorithms.exclusive_scan(inp, out, n)

    expected = np.zeros(n, dtype=np.int32)
    expected[1:] = np.cumsum(data[:-1])
    np.testing.assert_array_equal(out.numpy(), expected)
    assert total == int(np.sum(data))


def test_inclusive_scan_basic(backend):
    """Inclusive scan: output[i] = sum(input[0..i])."""
    n = 8
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(inp, np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int32))

    total = algorithms.inclusive_scan(inp, out, n)

    result = out.numpy()
    expected = np.array([3, 4, 8, 9, 14, 23, 25, 31], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)
    assert total == 31


def test_inclusive_scan_ones(backend):
    """Inclusive scan of all ones = [1, 2, 3, ..., n]."""
    n = 64
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(inp, np.ones(n, dtype=np.int32))

    total = algorithms.inclusive_scan(inp, out, n)

    np.testing.assert_array_equal(out.numpy(), np.arange(1, n + 1, dtype=np.int32))
    assert total == n


def test_exclusive_scan_preserves_input(backend):
    """Exclusive scan should not modify the input field."""
    n = 16
    inp = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    data = np.arange(n, dtype=np.int32)
    tack.copyto(inp, data)

    algorithms.exclusive_scan(inp, out, n)

    np.testing.assert_array_equal(inp.numpy(), data)


def test_copy(backend):
    """Copy field contents."""
    n = 100
    src = tack.empty((n,))
    dst = tack.empty((n,))
    data = np.random.randn(n).astype(np.float32)
    tack.copyto(src, data)

    algorithms.copy(src, dst, n)

    np.testing.assert_array_equal(dst.numpy(), data)


def test_fill_value(backend):
    """Fill field with constant value."""
    n = 100
    dst = tack.empty((n,))

    algorithms.fill_value(dst, 42.0, n)

    np.testing.assert_array_equal(dst.numpy(), np.full(n, 42.0, dtype=np.float32))


@pytest.mark.parametrize("n", [0, 1, 7, 4096, 4097, 50_000])
@pytest.mark.parametrize("total", [True, False])
def test_exclusive_scan_closes_the_offsets(backend, n, total):
    """With ``close`` the total lands at ``output[n]``, so the output is CSR
    offsets: in the scan's own launch for one launch's worth (a small mesh's
    filters saved a launch each), by one more for the chunked levels, and as
    0 for no elements."""
    values = np.random.default_rng(n).integers(0, 9, n).astype(np.int32)
    out = tack.empty((n + 1,), tack.i64)
    out.fill(-7)
    got = algorithms.exclusive_scan(tack.array(values), out, n, total=total, close=True)
    want = np.concatenate([[0], np.cumsum(values)])
    np.testing.assert_array_equal(out.numpy(), want)
    assert got == (int(want[-1]) if total else None)
    with pytest.raises(ValueError, match="close writes output"):
        algorithms.exclusive_scan(tack.array(values), tack.empty((n,), tack.i64), n, close=True)


def test_scans_of_no_elements_return_zero_and_write_nothing(backend):
    """n=0 used to read index -1: garbage or a launch error on CUDA."""
    for scan in (algorithms.exclusive_scan, algorithms.inclusive_scan):
        inp = tack.empty((4,), tack.i32)
        out = tack.empty((4,), tack.i32)
        tack.copyto(inp, np.array([1, 2, 3, 4], dtype=np.int32))
        tack.copyto(out, np.full(4, -7, dtype=np.int32))
        assert scan(inp, out, 0) == 0
        np.testing.assert_array_equal(out.numpy(), np.full(4, -7))
        with pytest.raises(ValueError, match="outside"):
            scan(inp, out, 5)
        with pytest.raises(ValueError, match="outside"):
            scan(inp, out, -1)


def test_scans_refuse_an_output_shorter_than_n(backend):
    inp = tack.empty((8,), tack.i32)
    out = tack.empty((4,), tack.i32)
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
        inp = tack.empty(a.shape, getattr(tack, dt))
        out = tack.empty(a.shape, getattr(tack, dt))
        tack.copyto(inp, a)
        total = scan(inp, out, a.size)
        np.testing.assert_array_equal(out.numpy(), expected)
        assert total == inclusive[-1].item()
        assert type(total) is type(inclusive[-1].item())


def test_scans_keep_f64(f64_backend):
    a = np.array([0.1, 0.2, 0.3, 0.4, 1e-17, 2.5], dtype=np.float64)
    exclusive, inclusive = _np_scans(a)
    for scan, expected in ((algorithms.exclusive_scan, exclusive),
                           (algorithms.inclusive_scan, inclusive)):
        inp = tack.empty(a.shape, tack.f64)
        out = tack.empty(a.shape, tack.f64)
        tack.copyto(inp, a)
        total = scan(inp, out, a.size)
        # The tree adds in a different order from cumsum: equal to rounding.
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-15, atol=1e-17)
        assert total == pytest.approx(a.sum(), rel=1e-15)
        assert isinstance(total, float)


def test_scans_convert_the_input_to_the_output_dtype(backend):
    inp = tack.empty((4,), tack.i32)
    out = tack.empty((4,))
    tack.copyto(inp, np.array([1, 2, 3, 4], dtype=np.int32))
    assert algorithms.exclusive_scan(inp, out, 4) == 10.0
    np.testing.assert_array_equal(out.numpy(), [0.0, 1.0, 3.0, 6.0])


def test_i32_counts_scan_into_i64_offsets_past_2_31(backend):
    """Stream compaction with i32 counts and i64 offsets: the sums overflow
    i32, so they must be formed in the output's dtype, not the input's."""
    a = np.array([2_000_000_000, 2_000_000_000, -5, 2_000_000_000, 7], dtype=np.int32)
    inclusive = np.cumsum(a, dtype=np.int64)
    exclusive = np.concatenate([np.zeros(1, np.int64), inclusive[:-1]])
    for scan, expected in ((algorithms.exclusive_scan, exclusive),
                           (algorithms.inclusive_scan, inclusive)):
        inp = tack.empty(a.shape, tack.i32)
        out = tack.empty(a.shape, tack.i64)
        tack.copyto(inp, a)
        assert scan(inp, out, a.size) == 6_000_000_002
        np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.parametrize("n", [255, 256, 257, 65_536, 131_089], ids=lambda n: f"n{n}")
def test_scans_across_chunks_and_levels(backend, n):
    """The scan totals chunks of 256, scans those totals the same way, and
    so on: these sizes end exactly on a chunk, just past one, and need two
    and three levels. In place, as compaction scans its flags, and over a
    prefix of a longer field."""
    rng = np.random.default_rng(n)
    values = rng.integers(-1000, 1000, size=n + 9).astype(np.int32)
    for scan, expected in ((algorithms.exclusive_scan,
                            np.concatenate([[0], np.cumsum(values[:n])[:-1]])),
                           (algorithms.inclusive_scan, np.cumsum(values[:n]))):
        data = tack.empty(values.shape, tack.i32)
        tack.copyto(data, values)
        assert scan(data, data, n) == values[:n].sum()
        np.testing.assert_array_equal(data.numpy()[:n], expected)
        np.testing.assert_array_equal(data.numpy()[n:], values[n:])


def test_chunk_totals_form_in_the_output_dtype(backend):
    """Every chunk of these i32 counts totals past 2^31: the chunk sums must be
    formed in the i64 output's dtype, at every level."""
    counts = np.full(1000, 20_000_000, dtype=np.int32)
    inp = tack.empty(counts.shape, tack.i32)
    out = tack.empty(counts.shape, tack.i64)
    tack.copyto(inp, counts)
    assert algorithms.exclusive_scan(inp, out, counts.size) == 20_000_000_000
    np.testing.assert_array_equal(out.numpy(), np.arange(1000, dtype=np.int64) * 20_000_000)


@pytest.mark.parametrize("n", [0, 1, 256, 3000, 131_089], ids=lambda n: f"n{n}")
def test_scans_without_a_total_never_read_the_host(backend, monkeypatch, n):
    """total=False gives the same prefixes and no host read: nothing waits for the
    device, so a caller needing only offsets keeps its launches queued."""
    from tack.algorithms import scan

    def refuse(field, count):
        raise AssertionError("total=False read the total")

    values = np.random.default_rng(n).integers(0, 50, max(n, 1)).astype(np.int32)[:n]
    for which, inclusive in ((algorithms.exclusive_scan, False),
                             (algorithms.inclusive_scan, True)):
        a = tack.empty((max(n, 1),), tack.i32)
        if n:
            tack.copyto(a, np.concatenate([values, np.zeros(max(n, 1) - n, np.int32)]))
        out = tack.empty((max(n, 1),), tack.i64)
        with monkeypatch.context() as m:
            m.setattr(scan, "_total", refuse)
            assert which(a, out, n, total=False) is None
        want = np.cumsum(values) if inclusive else np.concatenate([[0], np.cumsum(values)[:-1]])
        np.testing.assert_array_equal(out.numpy()[:n], want[:n])

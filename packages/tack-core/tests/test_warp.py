"""The Warp backend (a prototype): Tack kernels compiled and launched by NVIDIA Warp.

Each kernel runs on the CPU backend and on ``tack.device("warp")`` (Warp's
CUDA device when it has one, else its CPU), and the two must agree. The
cases cover what the generator spells differently from Warp's own
semantics: Python's ``//`` and ``%``, typed locals, loops with ``break``,
atomics, ``ndindex``, local arrays, constant tables and device classes.
"""

import numpy as np
import pytest

import tack

wp = pytest.importorskip("warp")


@pytest.fixture(scope="module")
def warp():
    return tack.device("warp")


def _both(warp, run):
    """``run(device)`` on the CPU and on Warp; their results."""
    return run(tack.device("cpu")), run(warp)


@tack.kernel
def _saxpy(a, x, y, out):
    for i in range(x.shape[0]):
        out[i] = a * x[i] + y[i]


def test_saxpy(warp):
    def run(d):
        x = tack.array(np.arange(100, dtype=np.float32), device=d)
        y = tack.array(np.ones(100, np.float32), device=d)
        out = tack.empty(100, tack.f32, device=d)
        _saxpy(2.5, x, y, out)
        return out.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp, cpu)


@tack.kernel
def _division(a, b, q, r, fq, fr):
    for i in range(a.shape[0]):
        q[i] = a[i] // b[i]
        r[i] = a[i] % b[i]
        fq[i] = float(a[i]) // float(b[i])
        fr[i] = float(a[i]) % float(b[i])


def test_floor_division_and_remainder_are_pythons(warp):
    """Warp's integer ``%`` truncates like C while its ``//`` rounds down."""
    a = np.array([-7, 7, -7, 7, 0, -1, 9], np.int32)
    b = np.array([2, -2, -2, 2, 3, 4, -4], np.int32)

    def run(d):
        out = [tack.empty(len(a), t, device=d) for t in (tack.i32, tack.i32, tack.f32, tack.f32)]
        _division(tack.array(a, device=d), tack.array(b, device=d), *out)
        return [o.numpy() for o in out]
    cpu, on_warp = _both(warp, run)
    for c, w in zip(cpu, on_warp):
        np.testing.assert_array_equal(w, c)
    np.testing.assert_array_equal(on_warp[0], a // b)
    np.testing.assert_array_equal(on_warp[1], a % b)


@tack.kernel
def _loops(x, out):
    for i in range(x.shape[0]):
        s = 0.0
        for j in range(i % 5):
            if j == 3:
                break
            s += x[j]
        k = i
        while k > 3:
            k = k // 2
        out[i] = s + k + (i % 3 == 0) + (2 ** (i % 4))


def test_loops_branches_and_typed_locals(warp):
    def run(d):
        x = tack.array(np.linspace(0, 1, 40, dtype=np.float32), device=d)
        out = tack.empty(40, tack.f32, device=d)
        _loops(x, out)
        return out.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp, cpu)


@tack.kernel
def _histogram(keys, counts, lowest):
    for i in range(keys.shape[0]):
        tack.atomic_add(counts, keys[i] % 7, 1)
        tack.atomic_min(lowest, keys[i] % 7, keys[i])


def test_atomics(warp):
    keys = np.random.default_rng(0).integers(0, 1000, 5000).astype(np.int32)

    def run(d):
        counts = tack.zeros(7, tack.i32, device=d)
        lowest = tack.full(7, 2**30, tack.i32, device=d)
        _histogram(tack.array(keys, device=d), counts, lowest)
        return counts.numpy(), lowest.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp[0], cpu[0])
    np.testing.assert_array_equal(on_warp[1], cpu[1])


@tack.kernel
def _grid(img):
    for i, j, k in tack.ndindex(img.shape[0], img.shape[1], img.shape[2]):
        img[i, j, k] = i * 100 + j * 10 + k


def test_ndindex(warp):
    img = tack.empty((3, 4, 5), tack.i32, device=warp)
    _grid(img)
    i, j, k = np.indices((3, 4, 5))
    np.testing.assert_array_equal(img.numpy(), i * 100 + j * 10 + k)


@tack.kernel
def _reverse_rows(x, out, rows):
    for i in range(rows):
        row = tack.local_array(tack.f32, 4)
        for c in range(4):
            row[c] = x[i * 4 + c]
        for c in range(4):
            out[i * 4 + c] = row[3 - c]


def test_local_arrays(warp):
    x = np.arange(32, dtype=np.float32)
    out = tack.empty(32, tack.f32, device=warp)
    _reverse_rows(tack.array(x, device=warp), out, 8)
    np.testing.assert_array_equal(out.numpy(), x.reshape(8, 4)[:, ::-1].reshape(-1))


_TABLE = tack.constant((3, 1, 4, 1, 5, 9, 2, 6), dtype=tack.i32)


@tack.kernel
def _lookup(out):
    for i in range(out.shape[0]):
        out[i] = _TABLE[i]


def test_constant_tables(warp):
    out = tack.empty(10, tack.i32, device=warp)
    _lookup(out)
    np.testing.assert_array_equal(out.numpy(), [3, 1, 4, 1, 5, 9, 2, 6, 6, 6])


@tack.device_class
class _Scaled:
    """A device class with a method calling a sibling, through a subclass."""
    def __init__(self, values, scale):
        self.values = values
        self.scale = scale

    @tack.func
    def raw(self, i):
        return self.values[i]

    @tack.func
    def get(self, i):
        return self.raw(i) * self.scale


@tack.device_class
class _Shifted(_Scaled):
    OFFSET = 10.0

    @tack.func
    def get(self, i):
        return self.raw(i) * self.scale + self.OFFSET


@tack.kernel
def _read(view, out):
    for i in range(out.shape[0]):
        out[i] = view.get(i)


def test_device_classes_methods_and_subclasses(warp):
    """What Warp's structs lack (methods, inheritance) Tack expands before codegen."""
    x = tack.array(np.arange(6, dtype=np.float32), device=warp)
    out = tack.empty(6, tack.f32, device=warp)
    _read(_Shifted(x, 2.0), out)
    np.testing.assert_array_equal(out.numpy(), np.arange(6) * 2.0 + 10.0)


def test_a_dataset_algorithm_runs_on_warp(warp):
    """tack-vis, unchanged, on the Warp device."""
    td = pytest.importorskip("tack.dataset")
    from tack.dataset import algorithms as alg

    def run(d):
        with d:
            data = td.uniform_grid((6, 5, 4))
            elevation = alg.point_elevation(data)
            return alg.gradients(data, elevation).values.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_allclose(on_warp, cpu, rtol=1e-6, atol=1e-6)


@tack.kernel
def _keywords(short, int, out):
    for i in range(out.shape[0]):
        out[i] = short[i] + int


def test_names_that_are_cpp_keywords(warp):
    """Warp copies names into C++ unchanged; tack-vis has a parameter called ``short``."""
    out = tack.empty(4, tack.i32, device=warp)
    _keywords(tack.array(np.arange(4, dtype=np.int32), device=warp), 10, out)
    np.testing.assert_array_equal(out.numpy(), np.arange(4) + 10)


@tack.kernel
def _wrapping(a, b, out):
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        out[i, 0] = (x * y) // y if y else x
        out[i, 1] = -x
        out[i, 2] = abs(x)
        out[i, 3] = x << 1


def test_signed_overflow_wraps(warp):
    """Warp's C++ makes signed overflow undefined: clang folded ``(x * y) // y`` to x."""
    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
    a = np.array([lo, lo, hi, -3, 7], np.int32)
    b = np.array([lo, -2, 2, 0, 3], np.int32)

    def run(d):
        out = tack.empty((5, 4), tack.i32, device=d)
        _wrapping(tack.array(a, device=d), tack.array(b, device=d), out)
        return out.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp, cpu)
    assert on_warp[0, 0] == 0 and on_warp[0, 1] == lo and on_warp[0, 2] == lo


@tack.kernel
def _negated(a, out):
    for i in range(a.shape[0]):
        out[i] = -a[i]


def test_negating_zero_gives_negative_zero(warp):
    """Warp's own float negation gives +0.0 for 0.0."""
    out = tack.empty(2, tack.f32, device=warp)
    _negated(tack.array(np.array([0.0, -0.0], np.float32), device=warp), out)
    assert list(np.signbit(out.numpy())) == [True, False]


@tack.kernel
def _wide_ranges(starts, ends, out):
    for i in range(starts.shape[0]):
        total = 0
        for j in range(starts[i], ends[i], 3):
            if j % 2 == 0:
                continue
            total = total + 1
        out[i] = total


def test_sequential_loops_take_64_bit_bounds(warp):
    """Warp's ``range`` is 32-bit; the loop counts its trips in 64 bits, and
    ``continue`` still steps."""
    starts = np.array([2**40, -5, 10, 2**62], np.int64)
    ends = np.array([2**40 + 30, 5, 10, 2**62 - 9], np.int64)

    def run(d):
        out = tack.empty(4, tack.i64, device=d)
        _wide_ranges(tack.array(starts, device=d), tack.array(ends, device=d), out)
        return out.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp, cpu)
    expected = [sum(1 for j in range(s, e, 3) if j % 2) for s, e in zip(starts.tolist(), ends.tolist())]
    np.testing.assert_array_equal(on_warp, expected)


@tack.kernel
def _powers(a, e, out):
    for i in range(a.shape[0]):
        out[i, 0] = a[i] ** e[i]
        out[i, 1] = pow(a[i], e[i])


def test_integer_power_keeps_the_exponents_type(warp):
    """A u8 base with a u16 exponent of 300: narrowed to the base's type it was 44."""
    def run(d):
        out = tack.empty((3, 2), tack.u8, device=d)
        _powers(tack.array(np.array([3, 2, 255], np.uint8), device=d),
                tack.array(np.array([300, 9, 2], np.uint16), device=d), out)
        return out.numpy()
    cpu, on_warp = _both(warp, run)
    np.testing.assert_array_equal(on_warp, cpu)
    np.testing.assert_array_equal(on_warp[:, 0], [pow(3, 300, 256), pow(2, 9, 256), pow(255, 2, 256)])


@tack.kernel
def _extrema(a, b, out):
    for i in range(a.shape[0]):
        out[i, 0] = min(a[i], b[i])
        out[i, 1] = max(a[i], b[i])


def test_float_min_and_max_prefer_a_number_to_nan(warp):
    out = tack.empty((2, 2), tack.f32, device=warp)
    _extrema(tack.array(np.array([np.nan, 1.0], np.float32), device=warp),
             tack.array(np.array([2.0, np.nan], np.float32), device=warp), out)
    np.testing.assert_array_equal(out.numpy(), [[2.0, 2.0], [1.0, 1.0]])


def test_wrap_ptr_shares_memory_both_ways(warp):
    """`array_from_ptr` on Warp wraps memory it does not own: a Warp array's, and
    on Warp's CPU a NumPy array's; writes through either side are seen by both."""
    owner = wp.array(np.arange(6, dtype=np.float32), device=warp._backend.wp_device)
    a = tack.array_from_ptr(owner.ptr, (6,), tack.f32, device=warp, writable=True)
    out = tack.empty(6, tack.f32, device=warp)
    _saxpy(2.0, a, a, out)
    np.testing.assert_array_equal(out.numpy(), np.arange(6) * 3.0)
    _saxpy(1.0, out, a, a)                  # store through the wrapped memory
    np.testing.assert_array_equal(owner.numpy(), np.arange(6) * 4.0)
    if not warp._backend.wp_device.is_cuda:
        host = np.ones(4, np.float32)
        h = tack.array_from_ptr(host, host.shape, tack.f32, device=warp, writable=True)
        _saxpy(5.0, h, h, h)
        np.testing.assert_array_equal(host, np.full(4, 6.0))


@tack.kernel
def _past_the_end(a, out):
    for i in range(out.shape[0]):
        out[i] = a[i + 2]


def test_debug_mode_checks_bounds(warp, monkeypatch):
    """Debug mode on Warp: the access past the end is refused, recorded and
    raised as IndexError naming it, and nothing is read or written out of
    bounds (Warp's CPU arrays have no guard pages)."""
    from tack.runtime import dispatch
    monkeypatch.setattr(dispatch, "_debug", True)
    a = tack.array(np.arange(4, dtype=np.float32), device=warp)
    out = tack.zeros(4, tack.f32, device=warp)
    with pytest.raises(IndexError, match=r"a\[i \+ 2\]"):
        _past_the_end(a, out)
    np.testing.assert_array_equal(out.numpy(), [2.0, 3.0, 0.0, 0.0])


def test_inspect_is_not_offered(warp):
    """`supports_inspect` is false: tack.inspect refuses the Warp device in every
    mode, and says where to look instead."""
    assert not warp._backend.supports_inspect and tack.device("cpu")._backend.supports_inspect
    x = tack.array(np.ones(4, np.float32), device=warp)
    for mode in ("ir", "source", "optimized"):
        with pytest.raises(NotImplementedError, match="another device"):
            tack.inspect(_saxpy, 1.0, x, x, x, mode=mode)


@tack.kernel
def _rotate_lanes(x, out):
    lanes = tack.shared_like(x, 256)
    for i in range(x.shape[0]):
        t = tack.thread_id()
        lanes[t] = x[i]
        tack.barrier()
        out[i] = lanes[(t + 1) % 256]


@tack.kernel
def _block_reductions(x, out):
    for i in range(x.shape[0]):
        out[i, 0] = tack.block_sum(x[i])
        out[i, 1] = tack.block_min(x[i])
        out[i, 2] = tack.block_max(x[i])


def test_workgroups_on_cuda(warp):
    """Shared arrays, barriers, lanes and block reductions are CUDA's (native
    helpers); the scans and sorts take their tile paths with them."""
    if not warp._backend.wp_device.is_cuda:
        assert not warp._backend.supports_workgroups
        pytest.skip("workgroups are CUDA's; Warp's device here is its CPU")
    x = np.random.default_rng(3).standard_normal(1024).astype(np.float32)
    x[300], x[600:700], x[650] = np.nan, 0.0, -0.0
    a = tack.array(x, device=warp)
    rotated = tack.empty(1024, tack.f32, device=warp)
    _rotate_lanes(a, rotated)
    np.testing.assert_array_equal(rotated.numpy(), np.roll(x.reshape(4, 256), -1, axis=1).ravel())
    reduced = tack.empty((1024, 3), tack.f32, device=warp)
    _block_reductions(a, reduced)
    blocks = x.reshape(4, 256)
    got = reduced.numpy().reshape(4, 256, 3)
    np.testing.assert_allclose(got[:, :, 0], np.broadcast_to(blocks.sum(axis=1, dtype=np.float64)[:, None],
                                                              (4, 256)), rtol=1e-5)
    np.testing.assert_array_equal(got[1:, :, 1], np.broadcast_to(blocks[1:].min(axis=1)[:, None], (3, 256)))
    assert np.isnan(got[1, :, 0]).all() and np.isnan(got[1, :, 2]).all()   # NaN propagates
    np.testing.assert_array_equal(np.signbit(got[2, :, 1]), np.signbit(blocks[2].min()))

    from tack.algorithms import argsort, exclusive_scan
    ints = np.random.default_rng(4).integers(0, 100, 100_000).astype(np.int32)
    scanned = tack.empty(len(ints), tack.i64, device=warp)
    exclusive_scan(tack.array(ints, device=warp), scanned, len(ints))
    np.testing.assert_array_equal(scanned.numpy(), np.concatenate([[0], np.cumsum(ints)[:-1]]))
    keys = np.random.default_rng(5).integers(-1 << 40, 1 << 40, 50_000)
    np.testing.assert_array_equal(argsort(tack.array(keys, device=warp)).numpy(),
                                  np.argsort(keys, kind="stable"))


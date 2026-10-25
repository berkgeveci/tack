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
def _reverse_rows(x, out):
    for i in range(x.shape[0]):
        row = tack.local_array(tack.f32, 4)
        for c in range(4):
            row[c] = x[i * 4 + c]
        for c in range(4):
            out[i * 4 + c] = row[3 - c]


def test_local_arrays(warp):
    x = np.arange(32, dtype=np.float32)
    out = tack.empty(32, tack.f32, device=warp)
    _reverse_rows(tack.array(x, device=warp), out)
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

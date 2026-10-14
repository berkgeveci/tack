"""The CPU backend's allocation and fill: a new field is not filled, and a
large fill runs as a kernel through the dispatch (`CPUBackend.fill_field`),
storing what `ndarray.fill` would store and refusing what it would refuse."""
import platform

import numpy as np
import pytest

import tack
from tack.runtime import cpu as cpu_mod
from tack.runtime.cpu import CPUBackend

pytestmark = pytest.mark.skipif(platform.system() == "Windows", reason="posix threads")

_BIG = cpu_mod._PARALLEL_FILL_MIN_BYTES


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    tack.init(arch="cpu", num_threads=4)
    yield tack.runtime.dispatch.get_backend()
    tack.init(arch="cpu")


def _fills(monkeypatch, backend):
    """Record each fill the backend routed through its kernel."""
    seen = []
    real = CPUBackend.fill_field

    def spy(self, field, value):
        took = real(self, field, value)
        seen.append((field._buffer.nbytes, took))
        return took
    monkeypatch.setattr(CPUBackend, "fill_field", spy)
    return seen


def test_a_new_field_is_allocated_without_a_fill(monkeypatch, backend):
    calls = []
    real_empty, real_zeros = np.empty, np.zeros
    monkeypatch.setattr(np, "empty", lambda *a, **k: calls.append("empty") or real_empty(*a, **k))
    monkeypatch.setattr(np, "zeros", lambda *a, **k: calls.append("zeros") or real_zeros(*a, **k))
    fills = _fills(monkeypatch, backend)
    f = tack.field(tack.f32, (_BIG,))
    assert calls == ["empty"] and fills == []
    assert f._buffer._data.shape == (_BIG,)


def test_zeros_fills_through_the_kernel_above_the_threshold(monkeypatch, backend):
    fills = _fills(monkeypatch, backend)
    big = tack.zeros(tack.i32, (_BIG // 4 + 3,))
    small = tack.zeros(tack.i32, (_BIG // 4 - 1,))
    assert fills == [(big._buffer.nbytes, True), (small._buffer.nbytes, False)]
    assert not big.to_numpy().any() and not small.to_numpy().any()
    assert big.to_numpy().shape == (_BIG // 4 + 3,)


@pytest.mark.parametrize("dtype", [tack.u8, tack.i16, tack.i32, tack.u32, tack.i64,
                                   tack.u64, tack.f32, tack.f64], ids=str)
@pytest.mark.parametrize("value", [0, 7, 1.5, 2 ** 40, 2 ** 63 + 5, -3, np.float32(0.1)],
                         ids=repr)
def test_a_large_fill_stores_what_numpy_would(monkeypatch, backend, dtype, value):
    n = _BIG // np.dtype(dtype.numpy_dtype).itemsize + 11
    expected = np.empty(n, dtype.numpy_dtype)
    try:
        expected.fill(value)
    except (OverflowError, ValueError) as refused:
        with pytest.raises(type(refused)):
            tack.field(dtype, (n,)).fill(value)
        return
    fills = _fills(monkeypatch, backend)
    f = tack.field(dtype, (n,))
    f.fill(value)
    assert fills == [(f._buffer.nbytes, True)]
    np.testing.assert_array_equal(f.to_numpy(), expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0], ids=repr)
def test_a_large_float_fill_keeps_the_special_values(backend, value):
    for dtype in (tack.f32, tack.f64):
        f = tack.field(dtype, (_BIG // 4 + 1,))
        f.fill(value)
        got = f.to_numpy()
        np.testing.assert_array_equal(got, np.full(got.shape, value, dtype.numpy_dtype))
        assert np.signbit(got).all() == np.signbit(value)


def test_a_large_fill_refuses_what_numpy_refuses(backend):
    with pytest.raises(ValueError):
        tack.field(tack.i32, (_BIG,)).fill(float("nan"))
    with pytest.raises(OverflowError):
        tack.field(tack.u8, (_BIG,)).fill(-1)


def test_a_large_fill_covers_a_two_dimensional_field_and_a_view(monkeypatch, backend):
    fills = _fills(monkeypatch, backend)
    grid = tack.field(tack.f32, (300, _BIG // 4 // 100))
    grid.fill(2.5)
    assert fills == [(grid._buffer.nbytes, True)]
    np.testing.assert_array_equal(grid.to_numpy(), np.full(grid.shape, 2.5, np.float32))
    flat = grid.reshape((grid.shape[0] * grid.shape[1],))
    flat.fill(-1.0)
    np.testing.assert_array_equal(grid.to_numpy(), np.full(grid.shape, -1.0, np.float32))


def test_a_large_fill_of_a_vector_field_takes_a_scalar(backend):
    n = _BIG // 12 + 5
    v = tack.Vector.field(3, tack.f32, shape=(n,))
    v.fill(4.0)
    np.testing.assert_array_equal(v.to_numpy(vectors=True), np.full((n, 3), 4.0, np.float32))
    v.fill([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v.to_numpy(vectors=True), np.tile([1.0, 2.0, 3.0], (n, 1)))


def test_a_read_only_field_refuses_a_fill_before_any_kernel(backend):
    host = np.zeros(_BIG // 4, np.float32)
    f = tack.field_from_ptr(host, tack.f32, host.shape)
    with pytest.raises(RuntimeError, match="read-only"):
        f.fill(1.0)
    assert not host.any()


def test_one_thread_fills_in_place(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    tack.init(arch="cpu", num_threads=1)
    try:
        fills = _fills(monkeypatch, tack.runtime.dispatch.get_backend())
        f = tack.zeros(tack.f64, (_BIG,))
        assert fills == [(f._buffer.nbytes, False)]
        assert not f.to_numpy().any()
    finally:
        tack.init(arch="cpu")


@pytest.mark.skipif(platform.system() != "Linux", reason="set_mempolicy")
def test_the_executor_threads_carry_the_interleave_policy(monkeypatch):
    """The executor's threads are made under the interleave policy, as the
    JIT'd pool's are, so pages they first-touch are spread across the
    nodes the way a buffer filled on the calling thread was."""
    if not cpu_mod._numa_available:
        pytest.skip("one memory node, or no libnuma")
    policy = cpu_mod._read_mempolicy()
    if policy is None or policy[0] != cpu_mod._MPOL_DEFAULT:
        pytest.skip("this task is bound; the binding is kept instead")
    monkeypatch.setenv("TACK_CPU_POOL", "python")
    backend = CPUBackend(num_threads=3)
    pool = backend._get_pool()
    seen = [pool.submit(cpu_mod._read_mempolicy).result() for _ in range(6)]
    assert {mode for mode, _ in seen} == {cpu_mod._MPOL_INTERLEAVE}
    assert cpu_mod._read_mempolicy() == policy
    pool.shutdown()


@pytest.mark.skipif(platform.system() != "Linux", reason="glibc mallopt")
def test_the_cpu_backend_keeps_freed_pages(monkeypatch):
    """A block freed and reallocated keeps its pages: the second fill of a
    64 MiB array takes almost no minor faults once the backend has set
    glibc's thresholds (it is process-wide, so the backend made by any
    earlier test counts), and `TACK_CPU_MALLOC=0` is honoured."""
    import resource
    backend = CPUBackend(num_threads=2)
    if not backend.malloc_tuned:
        pytest.skip("mallopt not available (not glibc?) or TACK_CPU_MALLOC=0")
    n = (64 << 20) // 4
    a = np.empty(n, np.float32)
    a.fill(1.0)
    del a
    before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    b = np.empty(n, np.float32)
    b.fill(2.0)
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before
    assert faults < (64 << 20) // 4096 // 8, f"{faults} faults refilling a freed 64 MiB block"
    del b
    monkeypatch.setenv("TACK_CPU_MALLOC", "0")
    monkeypatch.setattr(cpu_mod, "_malloc_tuned", None)
    assert not cpu_mod._tune_malloc()
    monkeypatch.setattr(cpu_mod, "_malloc_tuned", None)
    monkeypatch.delenv("TACK_CPU_MALLOC")
    assert cpu_mod._tune_malloc()

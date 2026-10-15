"""The CPU backend's allocation and fill: a new array is not filled, and a
large fill runs as a kernel through the dispatch (`CPUBackend.fill_array`),
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
    real = CPUBackend.fill_array

    def spy(self, array, value):
        took = real(self, array, value)
        seen.append((array._buffer.nbytes, took))
        return took
    monkeypatch.setattr(CPUBackend, "fill_array", spy)
    return seen


def test_a_new_array_is_allocated_without_a_fill(monkeypatch, backend):
    calls = []
    real_empty, real_zeros = np.empty, np.zeros
    monkeypatch.setattr(np, "empty", lambda *a, **k: calls.append("empty") or real_empty(*a, **k))
    monkeypatch.setattr(np, "zeros", lambda *a, **k: calls.append("zeros") or real_zeros(*a, **k))
    fills = _fills(monkeypatch, backend)
    f = tack.empty((_BIG,))
    assert calls == ["empty"] and fills == []
    assert f._buffer._data.shape == (_BIG,)


def test_zeros_fills_through_the_kernel_above_the_threshold(monkeypatch, backend):
    fills = _fills(monkeypatch, backend)
    big = tack.zeros((_BIG // 4 + 3,), tack.i32)
    small = tack.zeros((_BIG // 4 - 1,), tack.i32)
    assert fills == [(big._buffer.nbytes, True), (small._buffer.nbytes, False)]
    assert not big.numpy().any() and not small.numpy().any()
    assert big.numpy().shape == (_BIG // 4 + 3,)


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
            tack.empty((n,), dtype).fill(value)
        return
    fills = _fills(monkeypatch, backend)
    f = tack.empty((n,), dtype)
    f.fill(value)
    assert fills == [(f._buffer.nbytes, True)]
    np.testing.assert_array_equal(f.numpy(), expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0], ids=repr)
def test_a_large_float_fill_keeps_the_special_values(backend, value):
    for dtype in (tack.f32, tack.f64):
        f = tack.empty((_BIG // 4 + 1,), dtype)
        f.fill(value)
        got = f.numpy()
        np.testing.assert_array_equal(got, np.full(got.shape, value, dtype.numpy_dtype))
        assert np.signbit(got).all() == np.signbit(value)


def test_a_large_fill_refuses_what_numpy_refuses(backend):
    with pytest.raises(ValueError):
        tack.empty((_BIG,), tack.i32).fill(float("nan"))
    with pytest.raises(OverflowError):
        tack.empty((_BIG,), tack.u8).fill(-1)


def test_a_large_fill_covers_a_two_dimensional_array_and_a_view(monkeypatch, backend):
    fills = _fills(monkeypatch, backend)
    grid = tack.empty((300, _BIG // 4 // 100))
    grid.fill(2.5)
    assert fills == [(grid._buffer.nbytes, True)]
    np.testing.assert_array_equal(grid.numpy(), np.full(grid.shape, 2.5, np.float32))
    flat = grid.reshape((grid.shape[0] * grid.shape[1],))
    flat.fill(-1.0)
    np.testing.assert_array_equal(grid.numpy(), np.full(grid.shape, -1.0, np.float32))


def test_a_large_fill_of_a_vector_array_takes_a_scalar(backend):
    n = _BIG // 12 + 5
    v = tack.empty((n,), tack.vec(3, tack.f32))
    v.fill(4.0)
    np.testing.assert_array_equal(v.numpy(), np.full((n, 3), 4.0, np.float32))
    v.fill([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v.numpy(), np.tile([1.0, 2.0, 3.0], (n, 1)))


def test_a_read_only_array_refuses_a_fill_before_any_kernel(backend):
    host = np.zeros(_BIG // 4, np.float32)
    f = tack.array_from_ptr(host, host.shape, tack.f32)
    with pytest.raises(RuntimeError, match="read-only"):
        f.fill(1.0)
    assert not host.any()


def test_one_thread_fills_in_place(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    tack.init(arch="cpu", num_threads=1)
    try:
        fills = _fills(monkeypatch, tack.runtime.dispatch.get_backend())
        f = tack.zeros((_BIG,), tack.f64)
        assert fills == [(f._buffer.nbytes, False)]
        assert not f.numpy().any()
    finally:
        tack.init(arch="cpu")


def test_a_array_of_another_backend_fills_in_place(monkeypatch, backend):
    """An array made before `tack.init` switched to this backend has a buffer
    the CPU kernel must not touch: `Array.fill` asks the current backend,
    and the CPU backend answers only for its own buffers."""
    from tack.lang.array import NumpyBuffer

    class ForeignBuffer(NumpyBuffer):
        backend_name = "elsewhere"

        def fill(self, value):
            self.filled = value
            self._data.fill(value)

    f = tack.empty((_BIG,))
    foreign = ForeignBuffer(np.float32, (_BIG,))
    f._buffer = foreign
    fills = _fills(monkeypatch, backend)
    f.fill(3.0)
    assert fills == [(foreign.nbytes, False)] and foreign.filled == 3.0
    assert (foreign._data == 3.0).all()


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


def test_freed_pages_are_left_to_glibc_by_default(monkeypatch):
    """Keeping freed pages is opt-in: without `TACK_CPU_MALLOC=1` the
    backend leaves glibc's thresholds alone."""
    monkeypatch.delenv("TACK_CPU_MALLOC", raising=False)
    monkeypatch.setattr(cpu_mod, "_malloc_tuned", None)
    assert not cpu_mod._tune_malloc()
    assert not CPUBackend(num_threads=2).malloc_tuned
    monkeypatch.setenv("TACK_CPU_MALLOC", "0")
    monkeypatch.setattr(cpu_mod, "_malloc_tuned", None)
    assert not cpu_mod._tune_malloc()


_KEEPS_PAGES = """
import resource
import numpy as np
from tack.runtime.cpu import CPUBackend
if not CPUBackend(num_threads=2).malloc_tuned:
    print("untuned")
    raise SystemExit
n = (64 << 20) // 4
a = np.empty(n, np.float32)
a.fill(1.0)
del a
before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
b = np.empty(n, np.float32)
b.fill(2.0)
print(resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before)
"""


@pytest.mark.skipif(platform.system() != "Linux", reason="glibc mallopt")
def test_with_tack_cpu_malloc_a_freed_block_keeps_its_pages():
    """With `TACK_CPU_MALLOC=1` a block freed and reallocated keeps its
    pages: the second fill of a 64 MiB array takes almost no minor faults.
    In a fresh process, since the setting is process-wide and cannot be
    undone."""
    import os
    import subprocess
    import sys
    env = dict(os.environ, TACK_CPU_MALLOC="1")
    run = subprocess.run([sys.executable, "-c", _KEEPS_PAGES], env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-800:]
    out = run.stdout.strip().splitlines()[-1]
    if out == "untuned":
        pytest.skip("mallopt not available (not glibc?)")
    faults = int(out)
    assert faults < (64 << 20) // 4096 // 8, f"{faults} faults refilling a freed 64 MiB block"

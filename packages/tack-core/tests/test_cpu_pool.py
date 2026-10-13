"""The JIT'd thread pool (`tack.runtime.cpu_pool`): a fan-out as one ctypes
call. The policy that decides *whether* to fan out is tested through the
Python path in test_cpu_threading.py; here the pool itself is: that it is
chosen, that it runs every chunk once with the kernel's own arguments, that
backends with different thread counts share it, that it zeroes new fields,
and that it can be switched off."""
import ctypes
import os
import platform

import numpy as np
import pytest

import tack
from tack.runtime import cpu as cpu_mod
from tack.runtime import cpu_pool
from tack.runtime.cpu import CPUBackend
from tack.runtime.kernel_utils import resolve_variant

pytestmark = pytest.mark.skipif(platform.system() == "Windows",
                                reason="the pool is pthreads")


@tack.kernel
def _mixed(x, y, out, a: tack.f32, b: tack.f64, k: tack.i32, m: tack.i64):
    for i in range(out.shape[0]):
        out[i] = x[i] * a + y[i] * b + k + m


@tack.kernel
def _number(out):
    for i in range(out.shape[0]):
        out[i] = out[i] + i


@tack.kernel
def _grid(out, scale: tack.i32):
    for i, j in tack.ndrange(300, 257):
        out[i * 257 + j] = i * scale + j


def _compiled(backend, kernel, args):
    """The compiled variant and its bound prefix, the way `execute` gets them."""
    variant, effective = resolve_variant(backend, kernel, tuple(args), {},
                                         build=backend._build_variant,
                                         specialize_disjoint=cpu_mod._SPECIALIZE_DISJOINT)
    loop_end, extents = cpu_mod._get_launch(variant.ir, list(effective))
    return variant.payload, variant.payload.bind(list(effective), extents), loop_end


def _fresh_backend(threads, monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    return CPUBackend(num_threads=threads)


def test_the_pool_is_the_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    assert cpu_pool.enabled()
    assert CPUBackend(num_threads=2).pool_kind == "jit"
    for off in ("python", "0", "off"):
        monkeypatch.setenv("TACK_CPU_POOL", off)
        assert not cpu_pool.enabled()
        assert CPUBackend(num_threads=2).pool_kind == "python"


def test_one_thread_never_makes_a_pool(monkeypatch):
    backend = _fresh_backend(1, monkeypatch)
    assert backend._jit_pool() is None and backend._cpool is None


def test_a_fan_out_runs_every_element_once_with_every_argument_kind(monkeypatch):
    """Pointers of three widths, f32, f64, i32 and i64 scalars, through the
    pack and the wrapper, on a range the thread count does not divide."""
    backend = _fresh_backend(4, monkeypatch)
    n = 100_003
    x = tack.field(tack.f32, (n,)); x.from_numpy(np.arange(n, dtype=np.float32))
    y = tack.field(tack.f64, (n,)); y.from_numpy(np.full(n, 0.5))
    out = tack.zeros(tack.f64, (n,))
    compiled, prefix, loop_end = _compiled(backend, _mixed, [x, y, out, 2.0, 4.0, 7, 1 << 40])
    assert loop_end == n
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    assert backend._cpool is not None, "the fan-out did not go through the pool"
    expected = np.arange(n, dtype=np.float32).astype(np.float64) * 2.0 + 2.0 + 7 + (1 << 40)
    np.testing.assert_array_equal(out.to_numpy(), expected)
    # Every chunk ran, timed, and none twice: the spans say four chunks ran
    # and the result says each element was touched exactly once.
    spans = backend._cpool.spans()
    assert sum(1 for s in spans[:4] if s > 0) == 4


def test_each_element_is_visited_exactly_once(monkeypatch):
    backend = _fresh_backend(8, monkeypatch)
    n = 1_000_001
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    np.testing.assert_array_equal(out.to_numpy(), 2 * np.arange(n, dtype=np.int64))


def test_a_multi_dimensional_range_carries_its_extents(monkeypatch):
    backend = _fresh_backend(7, monkeypatch)
    out = tack.zeros(tack.i32, (300 * 257,))
    compiled, prefix, loop_end = _compiled(backend, _grid, [out, 1000])
    assert loop_end == 300 * 257
    backend._parallel_execute(compiled, prefix, 0, loop_end, whole_range=True)
    i, j = np.indices((300, 257))
    np.testing.assert_array_equal(out.to_numpy().reshape(300, 257), i * 1000 + j)


def test_backends_with_different_thread_counts_share_the_pool(monkeypatch):
    two, three = _fresh_backend(2, monkeypatch), _fresh_backend(3, monkeypatch)
    n = 50_000
    out = tack.zeros(tack.i64, (n,))
    for backend, chunks in ((two, 2), (three, 3)):
        compiled, prefix, _ = _compiled(backend, _number, [out])
        backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
        spans = backend._cpool.spans()
        assert sum(1 for s in spans[:chunks] if s > 0) == chunks
        assert all(s == 0 for s in spans[chunks:])
    assert two._cpool is three._cpool
    np.testing.assert_array_equal(out.to_numpy(), 2 * np.arange(n, dtype=np.int64))


def test_an_empty_range_is_a_fan_out_of_nothing(monkeypatch):
    """The calibration probes fan out empty ranges: no element may run."""
    backend = _fresh_backend(4, monkeypatch)
    n = 4096
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    assert backend._empty_fan_out_ns(compiled, prefix) > 0
    backend._parallel_execute(compiled, prefix, 0, 0, whole_range=True)
    assert not out.to_numpy().any()


def test_the_policy_measures_the_pool(monkeypatch):
    """The fan-out the policy calibrates is the pool's, not the executor's."""
    backend = _fresh_backend(4, monkeypatch)
    n = 4096
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._calibrate_fan_out(compiled, prefix)
    assert backend._pool is None, "the ThreadPoolExecutor was made"
    assert 0 < backend._fan_out_ns < 5_000_000


def test_a_large_field_is_zeroed_through_the_pool(monkeypatch):
    backend = _fresh_backend(4, monkeypatch)
    n = 4096
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)   # makes the pool
    big = backend.allocate_field(tack.f32, (cpu_mod._POOL_ZERO_MIN_BYTES // 4 * 3,))
    assert not big._data.any()
    small = backend.allocate_field(tack.u8, (16,))
    assert not small._data.any()


def test_zero_stops_at_the_end_of_the_buffer(monkeypatch):
    backend = _fresh_backend(5, monkeypatch)
    n = 4096
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    buf = np.full(1_000_003, 0xAB, np.uint8)
    backend._cpool.zero(buf.ctypes.data, 1_000_000, 5)
    assert not buf[:1_000_000].any() and (buf[1_000_000:] == 0xAB).all()


def test_the_python_path_still_works(monkeypatch):
    monkeypatch.setenv("TACK_CPU_POOL", "python")
    backend = CPUBackend(num_threads=4)
    n = 100_003
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    assert backend._cpool is None and backend._pool is not None
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.int64))


def test_a_pool_that_cannot_be_made_falls_back(monkeypatch):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    backend = CPUBackend(num_threads=4)
    monkeypatch.setattr(cpu_pool.CPool, "get", classmethod(lambda cls, threads: None))
    assert backend._jit_pool() is None
    assert backend.pool_kind == "python"
    n = 10_000
    out = tack.zeros(tack.i64, (n,))
    compiled, prefix, _ = _compiled(backend, _number, [out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.int64))


def test_a_kernel_through_init_and_call(monkeypatch):
    """The ordinary path: tack.init, a kernel call big enough to thread."""
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    tack.init(arch=tack.cpu, num_threads=4)
    n = 2_000_000
    out = tack.zeros(tack.i64, (n,))
    for _ in range(3):
        _number(out)
    np.testing.assert_array_equal(out.to_numpy(), 3 * np.arange(n, dtype=np.int64))

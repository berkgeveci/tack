"""CPU threading policy — when to fan out, and that it stays correct.

The backend decides between a serial run and a thread fan-out by comparing
a measured per-kernel cost against a measured fan-out cost, rather than
against a fixed element count.  These tests pin the decision at both ends
and, more importantly, check that every path through `_dispatch` produces
the same answer — the probe path splits a range in two, so a kernel must
survive being run in pieces.
"""

import numpy as np
import pytest

import tack
from tack.runtime.cpu import CPUBackend
from tack.runtime.dispatch import get_backend


@pytest.fixture(autouse=True)
def cpu():
    tack.init(arch=tack.cpu)
    return get_backend()


@tack.kernel
def _scale(x, out, n):
    for i in range(n):
        out[i] = x[i] * 2.0 + 1.0


@tack.kernel
def _expensive(x, out, n):
    for i in range(n):
        v = x[i]
        acc = 0.0
        for j in range(24):
            acc = acc + sin(v + float(j)) * cos(v - float(j))
        out[i] = acc


@tack.kernel
def _sum_into(x, total, n):
    for i in range(n):
        tack.atomic_add(total, 0, x[i])


def _run(kernel, n, dtype=tack.f32):
    x = tack.field(dtype=dtype, shape=(n,))
    out = tack.field(dtype=dtype, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32) * 0.001)
    kernel(x, out, n)
    return x.to_numpy(), out.to_numpy()


# ── Correctness across every dispatch path ───────────────────────────

@pytest.mark.parametrize("n", [1, 7, 64, 1023, 1024, 4096, 16384, 65536, 300000])
def test_results_match_a_single_serial_run(cpu, n):
    """Whatever the policy decides, the answer is the same."""
    x, out = _run(_scale, n)
    np.testing.assert_allclose(out, x * 2.0 + 1.0, rtol=1e-6)


def test_repeated_dispatches_stay_correct(cpu):
    """The estimate updates as it goes; the results must not drift."""
    n = 200000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32) * 0.001)
    expected = x.to_numpy() * 2.0 + 1.0
    for _ in range(20):
        _scale(x, out, n)
        np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-6)


def test_atomics_are_not_double_counted(cpu):
    """The calibration probe and the range split must not re-run work.

    An empty range does no iterations, and the probe split covers each
    element exactly once — so an atomic accumulation totals correctly.
    """
    n = 100000
    x = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))
    for _ in range(3):
        total = tack.field(dtype=tack.f32, shape=(1,))
        total.fill(0)
        _sum_into(x, total, n)
        assert total.to_numpy()[0] == pytest.approx(float(n), rel=1e-4)


def test_expensive_kernel_matches_numpy(cpu):
    """A kernel that certainly threads still agrees with a reference."""
    n = 50000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    src = np.linspace(0.0, 3.0, n, dtype=np.float32)
    x.from_numpy(src)
    _expensive(x, out, n)

    j = np.arange(24, dtype=np.float64)
    expected = (np.sin(src[:, None] + j) * np.cos(src[:, None] - j)).sum(axis=1)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=2e-4)


# ── The policy itself ────────────────────────────────────────────────

def test_small_range_never_spins_up_threads(cpu):
    """A tiny dispatch must not pay for a pool, or even measure one."""
    backend = CPUBackend()
    x = tack.field(dtype=tack.f32, shape=(256,))
    out = tack.field(dtype=tack.f32, shape=(256,))
    compiled = _compile_for(backend, _scale, [x, out, 256])

    for _ in range(5):
        backend._dispatch(compiled, [x, out, 256], 256)

    assert backend._pool is None
    assert backend._fan_out_ns is None


def test_cost_estimate_is_recorded(cpu):
    """A serial run leaves behind a per-element cost and a threshold."""
    backend = CPUBackend()
    n = 4096
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    compiled = _compile_for(backend, _scale, [x, out, n])

    assert compiled.ns_per_elem == 0.0
    backend._dispatch(compiled, [x, out, n], n)
    assert compiled.ns_per_elem > 0.0
    assert compiled.parallel_min_elems > 0


def test_a_cheaper_kernel_gets_a_larger_threshold(cpu):
    """The policy itself: threshold is inversely proportional to cost.

    This is the whole reason the decision is measured rather than a fixed
    element count, and it is a pure function — so it is tested as one,
    with costs supplied rather than timed. Two earlier versions of this
    test timed two real kernels and compared them, and both flaked on CI:
    first on a 10x ratio between the measured costs, then on the ordering
    of the thresholds *derived* from those same measurements, which
    inherits exactly the same noise (it failed 2279 < 2266 — a 0.6% gap).

    A cheap kernel's per-element cost is dominated by fixed dispatch
    overhead, so on a shared runner it is largely noise. Nothing derived
    from it should be asserted at all.
    """
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    backend._fan_out_ns = 200_000.0   # pin it; this is about the arithmetic

    cheap = backend._min_elems(0.1)     # ~memory-bound multiply-add
    medium = backend._min_elems(3.0)    # ~a sqrt/sin expression
    costly = backend._min_elems(128.0)  # ~a long inner loop

    assert costly < medium < cheap
    # And the relationship is the reciprocal one, not merely monotone.
    assert cheap == pytest.approx(medium * 30, rel=0.01)


def test_threshold_falls_back_when_the_cost_is_unknown(cpu):
    """An unmeasured kernel must not be threaded on a guess."""
    backend = CPUBackend()
    assert backend._min_elems(0.0) > 10 ** 12


def test_threshold_never_drops_below_the_thread_count(cpu):
    """Fewer elements than threads cannot be worth a fan-out."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    backend._fan_out_ns = 1.0    # absurdly cheap threads
    assert backend._min_elems(10 ** 9) >= backend.num_threads


def test_measured_cost_ranks_two_real_kernels(cpu):
    """The measurement does work — checked where noise cannot reach it.

    Timed over a range big enough that the expensive kernel takes
    milliseconds and the cheap one microseconds, the gap is ~1000x. That
    survives any runner. `_run_serial` is called directly so the parallel
    decision cannot stop the estimate from updating.
    """
    backend = CPUBackend()
    n = 100_000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 1, n, dtype=np.float32))

    cheap = _compile_for(backend, _scale, [x, out, n])
    costly = _compile_for(backend, _expensive, [x, out, n])
    args = [x, out, n]

    backend._run_serial(cheap, cheap.bind(args), 0, n)
    backend._run_serial(costly, costly.bind(args), 0, n)

    assert costly.ns_per_elem > cheap.ns_per_elem * 20


def test_single_thread_backend_never_threads(cpu):
    """num_threads=1 keeps everything on the calling thread."""
    backend = CPUBackend(num_threads=1)
    n = 500000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    compiled = _compile_for(backend, _scale, [x, out, n])

    for _ in range(3):
        backend._dispatch(compiled, [x, out, n], n)

    assert backend._pool is None
    assert compiled.parallel_min_elems > n
    np.testing.assert_allclose(out.to_numpy(), x.to_numpy() * 2.0 + 1.0, rtol=1e-6)


def test_thread_count_from_environment(cpu, monkeypatch):
    monkeypatch.setenv("TACK_CPU_THREADS", "3")
    assert CPUBackend().num_threads == 3


def test_explicit_thread_count_wins_over_environment(cpu, monkeypatch):
    monkeypatch.setenv("TACK_CPU_THREADS", "3")
    assert CPUBackend(num_threads=2).num_threads == 2


def test_expensive_kernel_does_fan_out(cpu):
    """An expensive kernel over a large range must actually use threads."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")

    n = 200000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 1, n, dtype=np.float32))
    compiled = _compile_for(backend, _expensive, [x, out, n])

    for _ in range(2):
        backend._dispatch(compiled, [x, out, n], n)

    assert backend._pool is not None
    assert backend._fan_out_ns > 0
    assert compiled.parallel_min_elems <= n


def _compile_for(backend, kernel, args):
    """Compile `kernel` for `args` into `backend`'s cache and return it."""
    from tack.lang.ir_resolve import resolve_ir
    from tack.lang.ir_optimize import optimize_ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types
    from tack.runtime.cpu import _compile_kernel

    ir_func = kernel.get_ir().functions[0]
    names = {p.name: a for p, a in zip(ir_func.params, args)
             if hasattr(a, "_buffer")}
    resolve_ir(ir_func, names)
    infer_param_types(ir_func, tuple(args))
    optimize_ir(ir_func)
    annotate_types(ir_func)
    return _compile_kernel(ir_func)

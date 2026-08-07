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
from tack.runtime import cpu as cpu_mod
from tack.runtime.cpu import _MAX_SAMPLE_RATIO, CPUBackend
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


# ── Constants that are not constants ─────────────────────────────────
#
# Two thresholds used to be element counts, and an element count silently
# encodes the thread cost and timer of whichever machine it was picked on.
# Both are now derived from measurements taken here.

def test_probe_threshold_scales_with_the_fan_out_cost(cpu):
    """Where threads are cheap, a shorter range can repay one."""
    backend = CPUBackend()

    backend._fan_out_ns = 200_000.0
    expensive = backend._probe_min_range()
    backend._fan_out_ns = 20_000.0
    cheap_threads = backend._probe_min_range()

    assert cheap_threads < expensive, (
        f"threshold stayed at {cheap_threads} elements when fan-out got "
        f"ten times cheaper")
    assert cheap_threads == pytest.approx(expensive / 10, rel=0.05)


def test_probe_threshold_has_a_value_before_calibration(cpu):
    backend = CPUBackend()
    assert backend._fan_out_ns is None
    assert backend._probe_min_range() > backend.num_threads


def test_sample_size_scales_with_the_kernel(cpu):
    """A costly kernel needs fewer elements to time than a cheap one."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    compiled.call_overhead_ns = 1000.0
    n = 1 << 20

    compiled.ns_per_elem = 100.0          # costly
    few = backend._sample_elems(compiled, n)
    compiled.ns_per_elem = 0.1            # cheap
    many = backend._sample_elems(compiled, n)

    assert few < many


def test_sample_size_survives_a_corrupt_estimate(cpu):
    """The re-measurement must not be sized by the thing it re-measures.

    A wildly high estimate asks for a handful of elements, whose timing is
    then almost all fixed call cost — so the sample confirms the corruption
    instead of correcting it. Measured directly, that took a corrupted
    estimate three elements at a time and left it 640x high.
    """
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    honest = compiled.ns_per_elem
    n = 1 << 20

    compiled.ns_per_elem = honest * 10_000
    corrupt = backend._sample_elems(compiled, n)

    assert corrupt >= n // 64, (
        f"a corrupt estimate shrank the sample to {corrupt} of {n} elements")


def test_a_range_too_small_to_thread_is_timed_whole(cpu):
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    n = backend._probe_min_range() // 2
    assert backend._sample_elems(compiled, n) == n


# ── Counting cores ───────────────────────────────────────────────────

def test_core_count_is_sane():
    """Physical cores, not logical: a hyperthread shares an execution unit,
    so a second compute thread on one costs a fan-out slot to buy little."""
    import os

    from tack.runtime.cpu import _physical_core_count

    count = _physical_core_count()
    assert count >= 1
    assert count <= (os.cpu_count() or 1), (
        f"reported {count} cores, more than the {os.cpu_count()} logical "
        f"processors that exist")


def test_every_platform_probe_answers_or_declines():
    """A probe off its own platform must return None, not guess."""
    from tack.runtime.cpu import _linux_core_count, _macos_core_count, _windows_core_count
    answered = 0
    for probe in (_linux_core_count, _macos_core_count, _windows_core_count):
        result = probe()
        assert result is None or result >= 1, f"{probe.__name__} -> {result}"
        answered += result is not None
    assert answered >= 1, "no platform probe recognised this machine"


# ── What the estimate is an estimate of ──────────────────────────────
#
# A dispatch costs a fixed amount before it touches a single element --
# ctypes marshalling, mostly, about a microsecond. Dividing that into a
# per-element rate makes a cheap kernel read as more expensive than it is,
# and the threshold derived from it then fans out ranges that serial would
# have finished sooner. Measured across a fine sweep of three kernels, that
# was eight wrong fan-out decisions in eighteen; charging only the
# per-element part leaves two, both near-ties in the other direction.
#
# It matters exactly where the decision is tightest: for an expensive
# kernel the fixed cost is lost in the work, for a cheap one it *is* the
# measurement.

def test_the_fixed_call_cost_is_measured(cpu):
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    assert compiled.call_overhead_ns > 0.0


def test_the_fixed_cost_is_not_charged_per_element(cpu, monkeypatch):
    """Two runs differing only in fixed cost must give one rate."""
    backend = CPUBackend()
    n = 8192
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)

    work = 4096.0                      # ns of actual per-element work
    overhead = compiled.call_overhead_ns

    compiled.ns_per_elem = 0.0         # start clean, so the sample is taken as-is
    seq = iter([0, int(overhead + work)])
    monkeypatch.setattr(cpu_mod.time, "perf_counter_ns", lambda: next(seq))
    backend._run_serial(compiled, prefix, 0, n)

    assert compiled.ns_per_elem == pytest.approx(work / n, rel=0.02), (
        f"{compiled.ns_per_elem:.4f} ns/elem for {work} ns of work over {n} "
        f"elements; the {overhead:.0f} ns call cost is being charged to the "
        f"elements")


def test_a_kernel_that_does_nothing_still_reads_as_measured(cpu, monkeypatch):
    """Subtracting the fixed cost must not leave a zero.

    ns_per_elem of 0.0 means "never measured", which would send the backend
    round the probe path on every dispatch forever.
    """
    backend = CPUBackend()
    n = 8192
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)

    compiled.ns_per_elem = 0.0
    seq = iter([0, int(compiled.call_overhead_ns)])   # all fixed cost, no work
    monkeypatch.setattr(cpu_mod.time, "perf_counter_ns", lambda: next(seq))
    backend._run_serial(compiled, prefix, 0, n)

    assert compiled.ns_per_elem > 0.0


# ── Recovering from a bad estimate ───────────────────────────────────
#
# The estimate is only refreshed by serial runs, so switching to threads
# also switches off the thing that would notice the switch was wrong. Left
# alone that is a one-way door: one mistimed sample turns threading on for
# a range that does not want it, and every dispatch after pays ~200 us to
# do work worth ~20 us, for the life of the process.
#
# It cannot be caught by comparing against the estimate either -- a wildly
# high one predicts an even worse serial run, so fanning out looks like a
# win however slow it really is. Only a fresh measurement settles it.
#
# Timings are supplied rather than measured, so none of this depends on
# how busy the machine is.

def _fixed_timing(monkeypatch, compiled, work_ns):
    """Make the next _run_serial believe its *elements* took this long.

    The backend subtracts the fixed per-call cost before working out a
    per-element rate, so a supplied timing has to carry that cost the way a
    real one does — otherwise the test is measuring a run that never happened.
    """
    elapsed = int(compiled.call_overhead_ns + work_ns)
    seq = iter([0, elapsed])
    monkeypatch.setattr(cpu_mod.time, "perf_counter_ns", lambda: next(seq))


def _measured(backend, kernel, n):
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    compiled = _compile_for(backend, kernel, [x, out, n])
    backend._dispatch(compiled, [x, out, n], n)
    assert compiled.ns_per_elem > 0.0
    return compiled, [x, out, n]


def test_one_outlier_sample_cannot_flip_the_decision(cpu, monkeypatch):
    """A descheduled dispatch times far above the kernel's real cost."""
    backend = CPUBackend()
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    settled = compiled.ns_per_elem

    _fixed_timing(monkeypatch, compiled, settled * n * 1000)
    backend._run_serial(compiled, compiled.bind(args), 0, n)

    assert compiled.ns_per_elem <= settled * _MAX_SAMPLE_RATIO, (
        f"one sample moved the estimate {settled:.1f} -> "
        f"{compiled.ns_per_elem:.1f} ns/elem")


def test_a_believable_rise_is_still_tracked(cpu, monkeypatch):
    """Clamping outliers must not deafen it to a real change."""
    backend = CPUBackend()
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    settled = compiled.ns_per_elem

    for _ in range(6):
        _fixed_timing(monkeypatch, compiled, settled * n * 3)
        backend._run_serial(compiled, compiled.bind(args), 0, n)

    assert compiled.ns_per_elem > settled * 2, (
        f"a sustained 3x rise left the estimate at {compiled.ns_per_elem:.1f}, "
        f"from {settled:.1f}")


def test_a_wrong_decision_to_thread_gets_corrected(cpu):
    """However the estimate got wrong, dispatching has to recover."""
    backend = CPUBackend()
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    honest = compiled.ns_per_elem

    # The state a bad sample leaves: threading on for a range far too small.
    compiled.ns_per_elem = honest * 10_000
    compiled.parallel_min_elems = backend._min_elems(compiled.ns_per_elem)
    assert n >= compiled.parallel_min_elems, "test did not set up the flip"

    for _ in range(4):
        backend._dispatch(compiled, args, n)

    assert n < compiled.parallel_min_elems, (
        "still threading a range this size; the estimate is never "
        "re-measured once threading is on")
    assert compiled.ns_per_elem < honest * 10, (
        f"estimate stuck at {compiled.ns_per_elem:.1f}, honest is {honest:.1f}")


def test_recheck_backs_off(cpu):
    """Re-measuring every dispatch would tax kernels that want threads."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 16)

    fired = sum(1 for _ in range(4096) if compiled.recheck_due())

    # 1, 2, 4 ... 1024, then every 1024.
    assert 10 <= fired <= 20, f"{fired} re-measurements in 4096 dispatches"


def test_recheck_keeps_the_parallelism(cpu):
    """A re-measurement times a slice, not the whole range.

    Running the entire range serially to check on it would cost the
    dispatch everything threading was bought for.
    """
    backend = CPUBackend()
    n = 1 << 17
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32) * 0.001)
    compiled = _compile_for(backend, _scale, [x, out, n])

    backend._dispatch(compiled, [x, out, n], n)

    ranges = []
    real_serial = backend._run_serial
    backend._run_serial = lambda c, p, a, b: (ranges.append((a, b)),
                                              real_serial(c, p, a, b))[1]

    # Pretend the fan-out cost is already known. Otherwise the first
    # parallel dispatch calibrates it and re-derives the threshold, which
    # sends this range back to a plain serial run before the recheck is
    # ever reached.
    backend._fan_out_ns = 1000.0
    compiled.parallel_min_elems = 1          # force the parallel branch
    compiled.recheck_after = 1
    compiled.parallel_since_measure = 0
    backend._dispatch(compiled, [x, out, n], n)

    assert ranges, "no re-measurement happened"
    start, end = ranges[0]
    assert end - start < n // 2, (
        f"re-measured {end - start} of {n} elements serially")
    np.testing.assert_allclose(
        out.to_numpy(), x.to_numpy() * 2.0 + 1.0, rtol=1e-5)


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
    from tack.lang.ir_optimize import optimize_ir
    from tack.lang.ir_resolve import resolve_ir
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

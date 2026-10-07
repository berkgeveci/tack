"""CPU threading policy — when to fan out, and that it stays correct.

The backend decides between a serial run and a thread fan-out by comparing
a measured per-kernel cost against a measured fan-out cost, rather than
against a fixed element count.  These tests pin the decision at both ends
and, more importantly, check that every path through `_dispatch` produces
the same answer — the probe path splits a range in two, so a kernel must
survive being run in pieces.
"""

import os
import time

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


def test_a_range_cheap_to_run_whole_is_timed_whole(cpu, monkeypatch):
    """Splitting a probe off a range that costs less than a fan-out buys
    nothing: the serial run is the cheap option either way."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: 2_500_000.0)
    compiled.ns_per_elem = 10.0
    n = 100_000                                   # 1.0 ms of work
    assert backend._sample_elems(compiled, n) == n


def test_a_costly_kernel_is_sampled_even_on_a_short_range(cpu, monkeypatch):
    """The guard is in the kernel's own rate, not an element count.

    A 384² volume render is ~150k pixels at ~3500 ns each: 0.5 s of work,
    which no fan-out cost justifies re-running on one thread. The old
    guard compared the count against what a 24 ns/element reference
    kernel needs to repay a fan-out, and re-timed the whole frame on every
    recheck.
    """
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: 2_500_000.0)
    compiled.ns_per_elem = 3500.0
    compiled.serial_floor_ns = 400.0        # what one fan-out measured
    n = backend._probe_min_range() // 2
    assert n * 400.0 > 2_500_000.0, "test range is not costly enough"
    assert backend._sample_elems(compiled, n) < n // 2


def test_a_corrupt_estimate_still_times_a_small_range_whole(cpu, monkeypatch):
    """The guard must not take the estimate's word for "expensive": a
    corrupt-high one would turn a measurable whole run into a 64-element
    slice that measures nothing but the clock."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: 2_500_000.0)
    honest = compiled.ns_per_elem
    n = 4096

    compiled.ns_per_elem = honest * 10_000
    assert backend._sample_elems(compiled, n) == n


def test_a_low_estimate_cannot_shrink_the_sample_guard(cpu, monkeypatch):
    """A prefix-biased (low) estimate must ask for a larger slice, never
    for the whole range -- the whole range is the failure mode."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 4096)
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: 2_500_000.0)
    n = 1 << 18
    compiled.ns_per_elem = 3500.0
    honest = backend._sample_elems(compiled, n)
    compiled.ns_per_elem = 50.0                   # 70x under
    biased = backend._sample_elems(compiled, n)
    assert honest <= biased < n


# ── The serial estimate is floored at the parallel rate ──────────────

def test_serial_estimate_never_sits_below_the_parallel_rate(cpu, monkeypatch):
    """A fan-out cannot make an element cheaper than it is serially, so a
    serial sample under the measured parallel rate is a biased sample."""
    backend = CPUBackend()
    if not backend._v2:
        pytest.skip("policy v1 does not measure the parallel rate")
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    honest = compiled.ns_per_elem

    compiled.serial_floor_ns = honest * 50
    _fixed_timing(monkeypatch, compiled, honest * n)
    backend._run_serial(compiled, compiled.bind(args), 0, n)

    assert compiled.ns_per_elem >= honest * 50


def test_a_parallel_measurement_lifts_a_low_serial_estimate(cpu):
    """The floor applies as soon as a fan-out has measured the range, not
    only on the next serial sample, which may be 1024 dispatches away."""
    backend = CPUBackend()
    if not backend._v2:
        pytest.skip("policy v1 does not measure the parallel rate")
    compiled, _ = _measured(backend, _scale, 4096)
    honest = compiled.ns_per_elem
    before = compiled.parallel_min_elems

    workers = 4
    backend._record_parallel_cost(compiled, [honest * 400.0] * workers, workers,
                                  bounds_serial=True)

    assert compiled.ns_per_elem_parallel == pytest.approx(honest * 100.0)
    assert compiled.ns_per_elem >= compiled.ns_per_elem_parallel
    assert compiled.parallel_min_elems < before


def test_a_short_fan_out_does_not_floor_the_serial_estimate(cpu):
    """Worker spans of a few microseconds are mostly GIL hand-back, not
    the kernel: 100x the serial rate, measured. Such a fan-out may still
    inform `r_p`, which the threshold tolerates, but never the floor."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 4096
    compiled, args = _measured(backend, _scale, n)

    backend._parallel_execute(compiled, compiled.bind(args), 0, n)

    assert compiled.serial_floor_ns == 0.0


def test_a_long_fan_out_floors_the_serial_estimate(cpu):
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    if not backend._v2:
        pytest.skip("policy v1 does not measure the parallel rate")
    n = 200000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 1, n, dtype=np.float32))
    compiled = _compile_for(backend, _expensive, [x, out, n])
    backend._dispatch(compiled, [x, out, n], n)      # measures the call cost

    backend._parallel_execute(compiled, compiled.bind([x, out, n]), 0, n)

    assert compiled.serial_floor_ns > 0.0
    assert compiled.ns_per_elem >= compiled.serial_floor_ns


@tack.kernel
def _variable_work(x, out, n, repeats):
    for i in range(n):
        acc = 0.0
        for j in range(repeats):
            acc = acc + sin(x[i] + float(j)) * cos(x[i] - float(j))
        out[i] = acc


def test_a_workload_that_becomes_cheap_returns_to_serial(cpu, monkeypatch):
    """A cost floor measured on old inputs must not trap new inputs in
    threading, even after the recheck schedule has reached its cap.

    Execute real kernel ranges, but supply the cost observations so CI
    load and the clock cannot decide whether this transition passes.
    """
    backend = CPUBackend(num_threads=4)
    if not backend._v2:
        pytest.skip("policy v1 does not measure the parallel rate")
    n = 1 << 17
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 3, n, dtype=np.float32))
    args = [x, out, n, 24]
    compiled = _compile_for(backend, _variable_work, args)
    compiled.call_range(compiled.bind(args), 0, n)

    # A costly previous workload established a floor and backed off its
    # rechecks. The same compiled code now receives zero repetitions.
    backend._fan_out_ns = 2_500_000.0
    monkeypatch.setattr(backend, '_fan_out_estimate', lambda: 2_500_000.0)
    compiled.call_overhead_ns = 1000.0
    compiled.ns_per_elem = compiled.serial_floor_ns = 1000.0
    compiled.ns_per_elem_parallel = 1000.0
    compiled.parallel_min_elems = backend._min_elems(1000.0, 1000.0)
    compiled.recheck_after = cpu_mod._RECHECK_CAP
    args[-1] = 0
    prefix = compiled.bind(args)
    serial_ranges = []
    real_serial = backend._run_serial

    def serial(c, p, start, end):
        serial_ranges.append((start, end))
        _fixed_timing(monkeypatch, c, end - start)  # one ns per element
        real_serial(c, p, start, end)

    def parallel(c, p, start, end, **kwargs):
        c.call_range(p, start, end)
        # All worker spans are now short, including on the whole range.
        if kwargs.get('whole_range', False):
            backend._record_parallel_cost(c, [1.0] * 4, 4,
                                          retire_serial_floor=True)
        else:
            backend._record_parallel_cost(c, [1.0] * 4, 4)

    monkeypatch.setattr(backend, '_run_serial', serial)
    monkeypatch.setattr(backend, '_parallel_execute', parallel)
    for _ in range(3):
        backend._run(compiled, prefix, n)

    assert serial_ranges and serial_ranges[-1] == (0, n), (
        'the cheap workload never returned to whole-range serial execution')
    assert compiled.parallel_min_elems > n
    np.testing.assert_array_equal(out.to_numpy(), np.zeros(n, dtype=np.float32))


@pytest.mark.parametrize('whole_range, worker_spans, retire', [
    (True, [1000] * 4, True),
    (True, [50] * 4, True),  # below the rate measurement's clock threshold
    (False, [1000] * 4, False),
    (True, [1000, 1000, 1000, 160000], False),  # one costly image region
], ids=['whole-short', 'whole-below-clock', 'partial-short', 'whole-mixed'])
def test_floor_retirement_requires_the_whole_range_to_be_fast(
        cpu, monkeypatch, whole_range, worker_spans, retire):
    """Short slices and a short median must not erase the renderer's floor.
    Supply worker clocks, but execute each real kernel range exactly once.
    """
    from concurrent.futures import Future

    backend = CPUBackend(num_threads=4)
    if not backend._v2:
        pytest.skip("policy v1 does not measure the parallel rate")
    n = 4096
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    out.fill(0)
    args = [x, out, n]
    compiled = _compile_for(backend, _scale, args)
    compiled.call_overhead_ns = 1000.0
    compiled.ns_per_elem = compiled.serial_floor_ns = 1000.0
    compiled.recheck_after = cpu_mod._RECHECK_CAP
    clock = [0]
    spans = iter(worker_spans)
    real_call = compiled.call_range

    def call(p, start, end):
        real_call(p, start, end)
        clock[0] += next(spans)

    class InlinePool:
        def submit(self, fn, *args):
            result = Future()
            result.set_result(fn(*args))
            return result

    monkeypatch.setattr(backend, '_get_pool', InlinePool)
    monkeypatch.setattr(compiled, 'call_range', call)
    monkeypatch.setattr(cpu_mod.time, 'perf_counter_ns', lambda: clock[0])
    end = n if whole_range else n // 2
    backend._parallel_execute(compiled, compiled.bind(args), 0, end,
                              whole_range=whole_range)

    if retire:
        assert compiled.recheck_due(), 'stale evidence delayed a fresh serial sample'
        assert compiled.serial_floor_ns == 0.0
    else:
        assert backend._sample_elems(compiled, n) < n
        assert compiled.serial_floor_ns == 1000.0
        assert not compiled.recheck_due()
    expected = np.zeros(n, dtype=np.float32)
    expected[:end] = x.to_numpy()[:end] * 2.0 + 1.0
    np.testing.assert_array_equal(out.to_numpy(), expected)


@tack.kernel
def _front_loaded(x, out, n, cut):
    for i in range(n):
        v = x[i]
        reps = 2
        if i >= cut:
            reps = 24
        acc = 0.0
        for j in range(reps):
            acc = acc + sin(v + float(j)) * cos(v - float(j))
        out[i] = acc


def _front_loaded_rate(cut, cheap=100.0, dense=1000.0):
    """A scripted cost for `_front_loaded`: `cheap` ns/element below
    `cut`, `dense` above -- the shape of an image whose top rows are
    background."""
    def rate(start, end):
        lo = max(0, min(end, cut) - start)
        return (lo * cheap + (end - start - lo) * dense) / (end - start)
    return rate


def _cheap_prefix_dispatches(backend, monkeypatch, compiled, args, n, rounds=12):
    serial, parallel = [], []
    real_serial, real_parallel = backend._run_serial, backend._parallel_execute
    backend._run_serial = lambda c, p, a, b: (serial.append((a, b)),
                                              real_serial(c, p, a, b))[1]
    backend._parallel_execute = lambda c, p, a, b, **kw: (parallel.append((a, b)),
                                                        real_parallel(c, p, a, b, **kw))[1]
    # First sight probes a slice and the first fan-out re-decides against
    # the calibrated cost, which may legitimately run one dispatch whole;
    # everything after that has more than one sample to go on.
    for _ in range(2):
        backend._dispatch(compiled, args, n)
    serial.clear()
    for _ in range(rounds):                       # rechecks on 1, 2, 4, 8
        backend._dispatch(compiled, args, n)
    return serial, parallel


def test_a_cheap_prefix_does_not_stall_the_rechecks(cpu, monkeypatch):
    """An image kernel's first rows are background: cheap, and not what
    the frame costs. Measured from them, the estimate undershoots, and
    with the old guard every recheck then re-ran the whole frame serially
    (1.1 s against 150 ms threaded on a 512² volume render). Once a
    fan-out has measured the range, a recheck must take a slice, and the
    slices must not all come from the same place.

    Controlled: the serial clock is scripted to the front-loaded shape
    (100 ns/element in the first quarter, 1000 beyond) and the fan-out
    cost is pinned, so the sample positions and the whole-range guard
    are exercised without this host's clock in the loop. The fan-outs
    are real. The real-clock version is in the timing group below.
    """
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 1 << 17
    cut = n // 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0.0, 3.0, n, dtype=np.float32))
    args = [x, out, n, cut]
    compiled = _compile_for(backend, _front_loaded, args)
    backend._measure_call_overhead(compiled, compiled.bind(args))
    _pin_fan_out(monkeypatch, backend, fan_out_ns=2_500_000.0)
    _controlled_serial_clock(backend, _front_loaded_rate(cut))
    # The fan-outs run for real but their worker rates are not recorded:
    # a single high r_p draw can lift the serial floor and with it the
    # estimate, which would put this host's scheduler back in the loop.
    # The floor's own behaviour has its own tests above.
    monkeypatch.setattr(CPUBackend, "_record_parallel_cost",
                        lambda self, *a, **k: None)

    serial, parallel = _cheap_prefix_dispatches(backend, monkeypatch, compiled, args, n)
    assert parallel, "the controlled rates must make this kernel thread"

    whole = [(a, b) for a, b in serial if b - a >= n // 2]
    assert not whole, (
        f"{len(whole)} recheck(s) re-ran the whole range serially: {whole}; "
        f"estimate {compiled.ns_per_elem:.0f} ns/elem, parallel "
        f"{compiled.ns_per_elem_parallel:.0f}")
    starts = {a for a, b in serial}
    assert len(starts) > 1, f"every sample came from the same place: {starts}"
    assert compiled.ns_per_elem >= compiled.ns_per_elem_parallel
    # The golden-ratio positions all land past the cheap quarter, so the
    # estimate reads the dense rows, not the background.
    assert compiled.ns_per_elem > 500.0, (
        f"rotating samples left the estimate at {compiled.ns_per_elem:.0f}")
    _check_front_loaded(x, out, n, cut)


def test_negative_control_prefix_only_sampling_is_detected(cpu, monkeypatch):
    """With the sample position pinned to the prefix, the controlled
    scenario above fails its position assertion: every sample comes from
    the cheap rows. That is the defect the rotation exists to prevent."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 1 << 17
    cut = n // 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0.0, 3.0, n, dtype=np.float32))
    args = [x, out, n, cut]
    compiled = _compile_for(backend, _front_loaded, args)
    backend._measure_call_overhead(compiled, compiled.bind(args))
    _pin_fan_out(monkeypatch, backend, fan_out_ns=2_500_000.0)
    _controlled_serial_clock(backend, _front_loaded_rate(cut))
    # The fan-outs run for real but their worker rates are not recorded:
    # a single high r_p draw can lift the serial floor and with it the
    # estimate, which would put this host's scheduler back in the loop.
    # The floor's own behaviour has its own tests above.
    monkeypatch.setattr(CPUBackend, "_record_parallel_cost",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(type(compiled), "next_sample_start", lambda self, room: 0)

    serial, parallel = _cheap_prefix_dispatches(backend, monkeypatch, compiled, args, n)

    starts = {a for a, b in serial}
    assert starts == {0}, f"prefix pinning did not take: {starts}"
    # Sampling the cheap rows alone leaves the estimate at their rate --
    # or at the trusted parallel floor, the only thing allowed to lift it.
    assert compiled.ns_per_elem == pytest.approx(
        max(100.0, compiled.serial_floor_ns), rel=0.05), (
        f"estimate {compiled.ns_per_elem:.0f} with prefix-only samples "
        f"(floor {compiled.serial_floor_ns:.0f}): a dense-row sample leaked in")


def _check_front_loaded(x, out, n, cut):
    j2 = np.arange(2, dtype=np.float64)
    j24 = np.arange(24, dtype=np.float64)
    src = x.to_numpy()
    expected = np.where(
        np.arange(n) < cut,
        (np.sin(src[:, None] + j2) * np.cos(src[:, None] - j2)).sum(axis=1),
        (np.sin(src[:, None] + j24) * np.cos(src[:, None] - j24)).sum(axis=1))
    np.testing.assert_allclose(out.to_numpy(), expected, atol=2e-4)


@pytest.mark.timing
def test_timing_a_cheap_prefix_does_not_stall_the_rechecks(cpu):
    """Real clock, real scheduler: the controlled test above with nothing
    pinned. Needs an idle host (see `_require_idle`); a failure here with
    the controlled test passing points at the scheduler or the sampling
    heuristics' margins on this machine, not at the policy's logic.
    """
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    _require_idle(backend)
    n = 1 << 17
    cut = n // 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0.0, 3.0, n, dtype=np.float32))
    args = [x, out, n, cut]
    compiled = _compile_for(backend, _front_loaded, args)

    serial, parallel = _cheap_prefix_dispatches(backend, None, compiled, args, n)
    if not parallel:
        pytest.skip("this machine does not thread this kernel at all")

    whole = [(a, b) for a, b in serial if b - a >= n // 2]
    assert not whole, (
        f"{len(whole)} recheck(s) re-ran the whole range serially: {whole}; "
        f"estimate {compiled.ns_per_elem:.0f} ns/elem, parallel "
        f"{compiled.ns_per_elem_parallel:.0f}, threads {backend.num_threads}, "
        f"load {os.getloadavg()[0]:.1f}")
    starts = {a for a, b in serial}
    assert len(starts) > 1, f"every sample came from the same place: {starts}"
    assert compiled.ns_per_elem >= compiled.ns_per_elem_parallel
    _check_front_loaded(x, out, n, cut)


@tack.kernel
def _count_visits(visits, n):
    for i in range(n):
        visits[i] = visits[i] + 1


def test_first_sight_samples_inside_the_range_not_its_prefix(cpu):
    """The first estimate a kernel gets decides its next dispatch, so it
    must not come from the cheap front of an image either. On a 256²
    volume render the prefix sample sent the following dispatch -- the
    whole frame -- to one thread, once per process.
    """
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 1 << 20
    visits = tack.field(dtype=tack.i32, shape=(n,))
    args = [visits, n]
    compiled = _compile_for(backend, _count_visits, args)
    assert n >= backend._probe_min_range()

    serial = []
    real_serial = backend._run_serial
    backend._run_serial = lambda c, p, a, b: (serial.append((a, b)),
                                              real_serial(c, p, a, b))[1]
    backend._dispatch(compiled, args, n)

    first_start, first_end = serial[0]
    assert first_start > n // 2, f"first sample was {serial[0]}"
    assert first_end - first_start < n // 8
    # Head, sample and tail together cover every element exactly once.
    np.testing.assert_array_equal(visits.to_numpy(), np.ones(n, dtype=np.int32))


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


def _controlled_serial_clock(backend, rate_fn):
    """Give every serial sample the rate `rate_fn(start, end)` ns/element.

    The kernel still runs, so results stay checkable; only the clock the
    policy reads around that run is scripted. `_run_serial` reads it
    exactly twice, and a sample is `(elapsed - call_overhead) / elems`,
    so the two ticks are 0 and `overhead + rate * elems`. Everything
    downstream -- smoothing, the 8x cap, confirmation, the floor, the
    threshold -- then sees a rate chosen by the test, not by this host.
    Pin `_fan_out_estimate` in the same test: with a fan-out curve it
    reads the clock too, and would consume a tick.
    """
    real = backend._run_serial

    def run_serial(compiled, prefix, start, end):
        if end <= start:
            return
        assert compiled.call_overhead_ns > 0.0, "measure the call cost first"
        ticks = iter([0, int(compiled.call_overhead_ns
                             + rate_fn(start, end) * (end - start))])
        saved = cpu_mod.time.perf_counter_ns
        cpu_mod.time.perf_counter_ns = lambda: next(ticks)
        try:
            real(compiled, prefix, start, end)
        finally:
            cpu_mod.time.perf_counter_ns = saved

    backend._run_serial = run_serial


def _pin_fan_out(monkeypatch, backend, fan_out_ns=400_000.0, margin=1.5):
    """Fix the fan-out cost and margin so thresholds are arithmetic."""
    backend._fan_out_ns = fan_out_ns
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: fan_out_ns)
    monkeypatch.setattr(backend, "_margin", lambda: margin)


def _require_idle(backend):
    """Timing-group precondition: these tests assert what the scheduler
    actually did, so they need the cores to themselves. Skipped, with the
    numbers, when the load average says otherwise."""
    try:
        load = os.getloadavg()[0]
    except (AttributeError, OSError):
        return
    if load > backend.num_threads / 2:
        pytest.skip(f"timing test needs an idle host: load average {load:.1f} "
                    f"against {backend.num_threads} threads")


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


def test_a_wrong_decision_to_thread_gets_corrected(cpu, monkeypatch):
    """However the estimate got wrong, dispatching has to recover.

    Controlled: the fan-out cost is pinned and every serial sample reads
    the honest rate, so the only thing under test is the policy's
    response -- a recheck on the first parallel dispatch, a sample 8x
    below the estimate believed outright, and the threshold recomputed
    above the range. The real-clock version of this scenario is in the
    timing group below.
    """
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    honest = compiled.ns_per_elem
    _pin_fan_out(monkeypatch, backend)
    _controlled_serial_clock(backend, lambda a, b: honest)

    # The state a bad sample leaves: threading on for a range far too small.
    compiled.ns_per_elem = honest * 10_000
    compiled.parallel_min_elems = backend._min_elems(compiled.ns_per_elem)
    assert n >= compiled.parallel_min_elems, "test did not set up the flip"

    for _ in range(4):
        backend._dispatch(compiled, args, n)

    assert n < compiled.parallel_min_elems, (
        "still threading a range this size; the estimate is never "
        "re-measured once threading is on")
    assert compiled.ns_per_elem == pytest.approx(honest, rel=1e-6), (
        f"estimate {compiled.ns_per_elem:.3f} after a clean sample of "
        f"{honest:.3f}: the 8x-below rule did not believe it outright")


def test_negative_control_no_rechecks_means_no_correction(cpu, monkeypatch):
    """The scenario above depends on the recheck: with rechecks switched
    off, the same controlled dispatches leave the corrupt estimate and the
    wrong decision exactly where they were."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    honest = compiled.ns_per_elem
    _pin_fan_out(monkeypatch, backend)
    _controlled_serial_clock(backend, lambda a, b: honest)
    monkeypatch.setattr(type(compiled), "recheck_due", lambda self: False)
    compiled.ns_per_elem = honest * 10_000
    compiled.parallel_min_elems = backend._min_elems(compiled.ns_per_elem)
    assert n >= compiled.parallel_min_elems

    for _ in range(4):
        backend._dispatch(compiled, args, n)

    assert n >= compiled.parallel_min_elems
    assert compiled.ns_per_elem == honest * 10_000


def test_a_sample_after_a_fan_out_cannot_raise_the_estimate(cpu, monkeypatch):
    """P7: a fan-out leaves the range in other cores' caches.

    A serial run straight afterwards pays to pull every line back, so it
    times a bandwidth-bound kernel 3-4x above what serial costs once it is
    running serially -- measured on a one-socket Xeon, 0.72 against 0.20
    ns/element. Believed, that keeps the estimate high, which keeps the
    backend fanning out, which keeps the next sample high. The sample is an
    upper bound, so it may lower the estimate and must not raise it.
    """
    backend = CPUBackend()
    if not backend._v2:
        pytest.skip("policy v2 only")
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)

    backend._parallel_execute(compiled, prefix, 0, n)
    # The estimate as the fan-out left it. The fan-out is a real one and
    # may move the estimate itself, through the floor its workers' spans
    # set; this test is about the serial sample that follows. Comparing
    # against the value from before the fan-out blamed the sample for a
    # worker the scheduler had stalled.
    settled = compiled.ns_per_elem
    _fixed_timing(monkeypatch, compiled, settled * n * 3)
    backend._run_serial(compiled, prefix, 0, n)

    assert compiled.ns_per_elem <= settled, (
        f"a sample taken after a fan-out raised the estimate "
        f"{settled:.3f} -> {compiled.ns_per_elem:.3f} ns/elem")


def _stall_one_worker(compiled, seconds):
    """Make the first chunk to start sleep, as a descheduled worker would."""
    real = compiled.call_range
    stalled = []

    def call_range(*args, **kwargs):
        if not stalled:
            stalled.append(True)
            time.sleep(seconds)
        return real(*args, **kwargs)

    compiled.call_range = call_range


def test_the_worker_median_is_the_lower_one_for_an_even_count():
    """Of two workers, the faster: the upper median of two is the max."""
    assert cpu_mod._worker_median([3.0]) == 3.0
    assert cpu_mod._worker_median([1.0, 900.0]) == 1.0
    assert cpu_mod._worker_median([1.0, 2.0, 900.0]) == 2.0
    assert cpu_mod._worker_median([1.0, 2.0, 3.0, 900.0]) == 2.0
    assert cpu_mod._worker_median([1.0, 2.0, 800.0, 900.0]) == 2.0


@pytest.mark.parametrize("threads", [2, 3, 4, 8])
def test_one_stalled_worker_does_not_set_the_serial_floor(cpu, threads):
    """One worker taken off its core must not stand for the kernel.

    The spans' median guards the floor, and for two workers the upper
    median is the slower one. With two threads, a 2 ms stall of one worker
    raised the serial estimate of this 0.3 ns/element kernel 1900 times,
    and its threshold with it, on a fan-out alone. That is a busy
    two-core machine's ordinary condition, and it is how a test that only
    ran a fan-out came to fail on CI runners.

    The span that qualifies as a bound is a few microseconds for this
    kernel, which a second worker reaches on its own once in some hundreds
    of fan-outs; then two are slow and the floor is rightly set. So the
    per-call cost is raised until that span is half a millisecond, and
    the stalled worker sleeps ten times as long: only it can cross."""
    backend = CPUBackend(num_threads=threads)
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)
    settled = compiled.ns_per_elem
    compiled.call_overhead_ns = 500_000.0 / (cpu_mod._RP_BOUND_SPAN_RATIO * threads)
    _stall_one_worker(compiled, 0.005)

    backend._parallel_execute(compiled, prefix, 0, n)

    assert compiled.serial_floor_ns == 0.0, (
        f"one stalled worker of {threads} set a floor of "
        f"{compiled.serial_floor_ns:.1f} ns/elem")
    assert compiled.ns_per_elem == pytest.approx(settled), (
        f"one stalled worker of {threads} moved the estimate "
        f"{settled:.3f} -> {compiled.ns_per_elem:.3f} ns/elem")


def test_two_slow_workers_still_set_the_serial_floor(cpu):
    """The lower median still believes a kernel that is slow on every
    worker: with both of two workers taking long, the floor is set."""
    backend = CPUBackend(num_threads=2)
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)
    real = compiled.call_range

    def slow(*call_args, **kwargs):
        time.sleep(0.002)
        return real(*call_args, **kwargs)

    compiled.call_range = slow
    backend._parallel_execute(compiled, prefix, 0, n)

    assert compiled.serial_floor_ns > 0.0
    assert compiled.ns_per_elem >= compiled.serial_floor_ns


def test_the_first_clean_sample_is_believed(cpu, monkeypatch):
    """P7's relapse: every sample before the first clean one is an upper bound.

    Smoothed in at a quarter's weight, the first clean sample decided by
    chance whether the threshold cleared the range. When it did not, the
    kernel fanned out again, every later sample was scattered again, and
    the estimate stuck near 3x -- two harness runs in three.
    """
    backend = CPUBackend()
    if not backend._v2:
        pytest.skip("policy v2 only")
    n = 4096
    compiled, args = _measured(backend, _scale, n)
    prefix = compiled.bind(args)
    backend._set_serial_cost(compiled, 1.0)         # built from upper bounds
    compiled.rate_confirmed = False
    compiled.scattered = False                      # a serial run laid it out

    _fixed_timing(monkeypatch, compiled, 0.3 * n)
    backend._run_serial(compiled, prefix, 0, n)

    assert compiled.ns_per_elem == pytest.approx(0.3, rel=1e-3), (
        f"the first clean sample (0.3) only moved the estimate to "
        f"{compiled.ns_per_elem:.3f}")
    assert compiled.rate_confirmed


def _stale_parallel_rate_setup(backend, monkeypatch):
    """P9's trap, reproduced by construction on any host.

    Only fan-outs measure r_p, so a high draw that holds a kernel serial
    is never corrected by the serial runs that follow. To put a kernel in
    that state without depending on this host's speed: measure its honest
    serial rate, pin every later serial sample to exactly that rate (so
    the serial side can neither free nor further trap it), and set r_p a
    hair below it, which makes the threshold formula credit fanning out
    with no gain and hold a range serial that is worth 2.5 fan-outs. The
    two setup assertions check both halves of that state.

    The kernel is compute-bound (`_expensive`, 24 sin/cos per element).
    With `_scale`, which is bandwidth-bound, two threads on a two-core or
    shared-bandwidth machine can be no faster than one, so the honest
    re-measured r_p was legitimately not under the stale value, and the
    test failed on a two-thread run (3 in 30 here, once on a CI runner)
    while testing nothing about the recovery path.
    """
    n = 1 << 18
    compiled, args = _measured(backend, _expensive, n)
    rate = compiled.ns_per_elem
    # Pin the fan-out cost *relative* to the measured rate: the range is
    # then worth 3.3 fan-outs serially on every host, and with r_p at
    # 0.95 of the serial rate the threshold formula's gap is 0.1 of it,
    # which puts the threshold at 4.5 ranges -- held serial, by r_p only.
    _pin_fan_out(monkeypatch, backend, fan_out_ns=n * rate / 3.3)
    _controlled_serial_clock(backend, lambda a, b: rate)
    compiled.rate_confirmed = True
    compiled.ns_per_elem_parallel = rate * 0.95       # the one bad draw
    compiled.serial_floor_ns = 0.0
    backend._set_serial_cost(compiled, rate)
    assert n < compiled.parallel_min_elems, "test did not set up the stale state"
    assert n * rate >= backend._fan_out_estimate() * backend._margin(), (
        "range too small for r_p to be what keeps it serial")
    return compiled, args, n, rate


def test_a_stale_parallel_rate_gets_re_measured(cpu, monkeypatch):
    """P9: a parallel rate that holds a kernel serial gets re-measured.

    Worker rates for identical fan-outs spread 2-8x run to run, and one
    high draw -- 5.25 ns/element against ~0.4 on a one-socket Xeon -- set
    r_p to the serial rate and held 6-34M-element dispatches of a cheap
    kernel serial, up to 6x slower, for good: the mirror of P3's one-way
    door. The fix fans out on the serial rechecks' back-off schedule when
    r_p alone is what keeps the range serial.

    The re-measured r_p is real, so a single one can itself be a high
    draw; what the fix guarantees is that the kernel keeps fanning out to
    find out, so the best of them must get under the stale value. On any
    machine with more than one core that holds by a wide margin: the
    honest r_p is the serial rate over the effective parallelism.
    """
    backend = CPUBackend()
    if not backend._v2 or backend.num_threads < 2:
        pytest.skip("policy v2 with threads only")
    compiled, args, n, rate = _stale_parallel_rate_setup(backend, monkeypatch)

    fanned = []
    real = backend._parallel_execute
    backend._parallel_execute = lambda c, p, a, b, **kw: (fanned.append((a, b)),
                                                        real(c, p, a, b, **kw))[1]
    seen = []
    for _ in range(8):
        backend._dispatch(compiled, args, n)
        seen.append(compiled.ns_per_elem_parallel)

    assert fanned, "a parallel rate that holds the range serial was never re-measured"
    assert min(seen) < rate * 0.95, (
        f"re-measured r_p never got under the stale {rate * 0.95:.3f}: {seen}")
    x, out = args[0].to_numpy().astype(np.float64), args[1].to_numpy()
    want = sum(np.sin(x + j) * np.cos(x - j) for j in range(24))
    np.testing.assert_allclose(out, want, rtol=1e-4, atol=1e-4)


def test_negative_control_stale_rate_without_recovery_stays_serial(
        cpu, monkeypatch):
    """The scenario above is held serial *only* by r_p: with P9's parallel
    recheck switched off, eight dispatches never fan out and r_p never
    moves. This is what makes the test above a test of the recovery path
    rather than of this host's threshold arithmetic."""
    backend = CPUBackend()
    if not backend._v2 or backend.num_threads < 2:
        pytest.skip("policy v2 with threads only")
    monkeypatch.setattr(CPUBackend, "_parallel_recheck_due",
                        lambda self, compiled, loop_end: False)
    compiled, args, n, rate = _stale_parallel_rate_setup(backend, monkeypatch)

    fanned = []
    real = backend._parallel_execute
    backend._parallel_execute = lambda c, p, a, b, **kw: (fanned.append((a, b)),
                                                        real(c, p, a, b, **kw))[1]
    for _ in range(8):
        backend._dispatch(compiled, args, n)

    assert not fanned, f"fanned out without the recovery path: {fanned}"
    assert compiled.ns_per_elem_parallel == rate * 0.95
    assert n < compiled.parallel_min_elems


def test_one_high_parallel_draw_cannot_set_the_serial_floor(cpu):
    """P9: the floor came from a single r_p sample, taken raw.

    One draw at 5.25 ns/element per worker -- against ~0.4 -- floored the
    serial estimate at r_p itself, and the threshold formula then saw no
    gain from fanning out at all.
    """
    backend = CPUBackend()
    if not backend._v2:
        pytest.skip("policy v2 only")
    compiled, _ = _measured(backend, _scale, 4096)
    compiled.ns_per_elem_parallel = 0.02
    backend._set_serial_cost(compiled, 0.2)
    workers = 20

    backend._record_parallel_cost(compiled, [0.3 * workers] * workers,
                                  workers, bounds_serial=True)

    assert compiled.serial_floor_ns < 0.15, (
        f"one draw set the floor to {compiled.serial_floor_ns:.3f}")
    assert compiled.ns_per_elem < 0.3


def test_a_dear_first_sample_does_not_buy_a_fan_out_calibration(cpu):
    """P7: the first sample of a kernel can read it ~15x too dear.

    It is taken on a range nothing has touched yet, so it pays first-touch
    page faults on top of a short slice's start-up: 15 ns/element for a
    kernel that runs at 0.2 once its fields are resident, on a one-socket
    Xeon. Believed, that puts the threshold under the range, and the next
    dispatch calibrates the fan-out -- 210 ms of deliberate sleeps -- to
    learn the range never wanted threads. A range a serial run finishes
    in a few fan-outs' time should be timed serially first.
    """
    backend = CPUBackend()
    if not backend._v2 or backend.num_threads < 2:
        pytest.skip("policy v2 with threads only")
    n = 100_000
    assert n >= backend._probe_min_range(), "test needs the first-sight path"
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    args = [x, out, n]
    compiled = _compile_for(backend, _scale, args)
    backend._dispatch(compiled, args, n)

    # The state a first-touch sample leaves, made certain rather than
    # left to whether this machine's page faults happen to be slow.
    backend._set_serial_cost(compiled, 4.0)
    assert n >= compiled.parallel_min_elems, "test did not set up the flip"

    for _ in range(3):
        backend._dispatch(compiled, args, n)

    assert backend._fan_out_ns is None, (
        "calibrated the fan-out for a range serial finishes in microseconds")
    assert n < compiled.parallel_min_elems
    np.testing.assert_allclose(out.to_numpy(), x.to_numpy() * 2.0 + 1.0,
                               rtol=1e-6)


def test_a_dispatch_never_sleeps(cpu, monkeypatch):
    """P8: the first fan-out used to calibrate the idle curve by sleeping.

    Three reps at 0, 10 and 50 ms gaps is 180 ms of `time.sleep` on the
    caller's thread, inside whichever dispatch first looked worth
    threading -- measured at 210 ms in all for a 100k-element add whose
    own run is 0.2 ms. The idle knots are learned from the pauses the
    program actually takes instead.
    """
    backend = CPUBackend()
    if not backend._v2 or backend.num_threads < 2:
        pytest.skip("policy v2 with threads only")
    slept = []
    monkeypatch.setattr(cpu_mod.time, "sleep", lambda s: slept.append(s))
    n = 1 << 16
    compiled, args = _measured(backend, _scale, n)
    backend._set_serial_cost(compiled, 1e4)      # dear: fans out at once
    compiled.rate_confirmed = True
    backend._dispatch(compiled, args, n)

    assert backend._fan_out_ns is not None, "test did not reach a fan-out"
    assert backend._fan_out_curve, "v2 built no fan-out curve"
    assert not slept, f"a dispatch slept {sum(slept) * 1e3:.0f} ms"


def test_an_idle_knot_is_learned_where_it_was_measured(cpu):
    """Until a pause has been seen, an idle knot is a pessimistic prior.

    The first fan-out measured at a natural pause replaces it outright and
    moves the knot to that pause, so the curve describes the gaps this
    program has rather than ones it was made to wait through.
    """
    backend = CPUBackend()
    if not backend._v2 or backend.num_threads < 2:
        pytest.skip("policy v2 with threads only")
    n = 1 << 16
    compiled, args = _measured(backend, _scale, n)
    backend._calibrate_fan_out(compiled, compiled.bind(args))
    hot = backend._fan_out_curve[0][1]
    idle = [cost for _, cost in backend._fan_out_curve[1:]]
    assert idle and all(cost >= hot * 2 for cost in idle), (
        f"unmeasured idle knots {idle} are not pessimistic against hot {hot}")

    backend._update_fan_out_knot(hot * 1.5, 30e6)       # a 30 ms pause

    gaps = [gap for gap, _ in backend._fan_out_curve]
    assert 30e6 in gaps, f"the knot did not move to its measured gap: {gaps}"
    assert backend._fan_out_curve[gaps.index(30e6)][1] == hot * 1.5
    assert gaps == sorted(gaps)


def test_recheck_backs_off(cpu):
    """Re-measuring every dispatch would tax kernels that want threads."""
    backend = CPUBackend()
    compiled, _ = _measured(backend, _scale, 16)

    fired = sum(1 for _ in range(4096) if compiled.recheck_due())

    # 1, 2, 4 ... 1024, then every 1024.
    assert 10 <= fired <= 20, f"{fired} re-measurements in 4096 dispatches"


def test_recheck_keeps_the_parallelism(cpu, monkeypatch):
    """A re-measurement times a slice, not the whole range.

    Running the entire range serially to check on it would cost the
    dispatch everything threading was bought for.

    The slice size is derived from the kernel's measured rate, the fixed
    call cost and the fan-out estimate, so those are pinned here: left to
    the clock, a 0.05 ns/element reading against a 400 ns call cost asks
    for 80,000 elements of 131,072, which is a correct answer to the
    wrong question and failed this test about once in twenty runs.
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
    monkeypatch.setattr(backend, "_fan_out_estimate", lambda: 1000.0)
    compiled.ns_per_elem = 1.0               # 1 ns/elem, 1 us call cost:
    compiled.call_overhead_ns = 1000.0       # a 10,000-element slice
    compiled.serial_floor_ns = 0.0
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


def test_a_dear_kernel_over_a_large_range_is_fanned_out(cpu, monkeypatch):
    """Policy check with nothing measured: a kernel whose pinned serial
    rate makes the range worth hundreds of fan-outs must take the parallel
    branch, and a kernel a thousand times cheaper over the same range must
    not. The real-clock version is in the timing group below."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    n = 200000
    compiled, args = _measured(backend, _expensive, n)
    _pin_fan_out(monkeypatch, backend)
    fanned = []
    real = backend._parallel_execute
    backend._parallel_execute = lambda c, p, a, b, **kw: (fanned.append((a, b)),
                                                        real(c, p, a, b, **kw))[1]

    for rate, expect in ((1000.0, True), (0.001, False)):
        fanned.clear()
        compiled.rate_confirmed = True
        compiled.ns_per_elem_parallel = 0.0
        compiled.serial_floor_ns = 0.0
        _controlled_serial_clock(backend, lambda a, b, r=rate: r)
        backend._set_serial_cost(compiled, rate)
        assert (n >= compiled.parallel_min_elems) == expect
        backend._dispatch(compiled, args, n)
        assert bool(fanned) == expect, (
            f"{rate} ns/elem over {n}: fanned {fanned}, expected {expect}")


@pytest.mark.timing
def test_timing_expensive_kernel_does_fan_out(cpu):
    """Real clock, real scheduler: an expensive kernel over a large range
    must actually spin up the pool and settle on threading. Needs an idle
    host (see `_require_idle`)."""
    backend = CPUBackend()
    if backend.num_threads < 2:
        pytest.skip("machine has one core")
    _require_idle(backend)

    n = 200000
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.linspace(0, 1, n, dtype=np.float32))
    compiled = _compile_for(backend, _expensive, [x, out, n])

    for _ in range(2):
        backend._dispatch(compiled, [x, out, n], n)

    assert backend._pool is not None
    assert backend._fan_out_ns > 0
    assert compiled.parallel_min_elems <= n, (
        f"threshold {compiled.parallel_min_elems} above {n}: estimate "
        f"{compiled.ns_per_elem:.1f} ns/elem, parallel "
        f"{compiled.ns_per_elem_parallel:.1f}, fan-out {backend._fan_out_ns:.0f} ns, "
        f"threads {backend.num_threads}, load {os.getloadavg()[0]:.1f}")


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

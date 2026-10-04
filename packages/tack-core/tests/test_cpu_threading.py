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


def test_a_cheap_prefix_does_not_stall_the_rechecks(cpu):
    """An image kernel's first rows are background: cheap, and not what
    the frame costs. Measured from them, the estimate undershoots, and
    with the old guard every recheck then re-ran the whole frame serially
    (1.1 s against 150 ms threaded on a 512² volume render). Once a
    fan-out has measured the range, a recheck must take a slice.
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

    serial, parallel = [], []
    real_serial, real_parallel = backend._run_serial, backend._parallel_execute
    backend._run_serial = lambda c, p, a, b: (serial.append((a, b)),
                                              real_serial(c, p, a, b))[1]
    backend._parallel_execute = lambda c, p, a, b, **kw: (parallel.append((a, b)),
                                                        real_parallel(c, p, a, b, **kw))[1]

    # First sight probes a slice, and the first fan-out calibrates the
    # real fan-out cost and re-decides against it -- which may
    # legitimately run that one dispatch whole. Everything after that
    # has more than one sample to go on.
    for _ in range(2):
        backend._dispatch(compiled, args, n)
    if not parallel:
        pytest.skip("this machine does not thread this kernel at all")
    serial.clear()
    for _ in range(12):                           # rechecks on 1, 2, 4, 8
        backend._dispatch(compiled, args, n)

    whole = [(a, b) for a, b in serial if b - a >= n // 2]
    assert not whole, (
        f"{len(whole)} recheck(s) re-ran the whole range serially: {whole}; "
        f"estimate {compiled.ns_per_elem:.0f} ns/elem, parallel "
        f"{compiled.ns_per_elem_parallel:.0f}")
    starts = {a for a, b in serial}
    assert len(starts) > 1, f"every sample came from the same place: {starts}"
    assert compiled.ns_per_elem >= compiled.ns_per_elem_parallel

    j2 = np.arange(2, dtype=np.float64)
    j24 = np.arange(24, dtype=np.float64)
    src = x.to_numpy()
    expected = np.where(
        np.arange(n) < cut,
        (np.sin(src[:, None] + j2) * np.cos(src[:, None] - j2)).sum(axis=1),
        (np.sin(src[:, None] + j24) * np.cos(src[:, None] - j24)).sum(axis=1))
    np.testing.assert_allclose(out.to_numpy(), expected, atol=2e-4)


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
    settled = compiled.ns_per_elem

    backend._parallel_execute(compiled, prefix, 0, n)
    _fixed_timing(monkeypatch, compiled, settled * n * 3)
    backend._run_serial(compiled, prefix, 0, n)

    assert compiled.ns_per_elem <= settled, (
        f"a sample taken after a fan-out raised the estimate "
        f"{settled:.3f} -> {compiled.ns_per_elem:.3f} ns/elem")


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

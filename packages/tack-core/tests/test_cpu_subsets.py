"""A thread count per launch (`CPUBackend._threads_for`, `TACK_CPU_SUBSETS`):
the size and saturation rules, the full-run schedule that keeps `r_p`
current, and the fallback when a subset runs slower than the full rate."""
import math
import platform

import numpy as np
import pytest

import tack
from tack.runtime import cpu as cpu_mod
from tack.runtime.cpu import CPUBackend
from tack.runtime.kernel_utils import resolve_variant

pytestmark = pytest.mark.skipif(platform.system() == "Windows", reason="posix threads")


@pytest.fixture(autouse=True)
def _on_the_cpu():
    """The fields below come from the active backend; make it the CPU."""
    tack.init(arch="cpu")


@tack.kernel
def _scale(x, out):
    for i in range(x.shape[0]):
        out[i] = x[i] * 2.0 + 1.0


def _compiled(backend, kernel, args):
    variant, effective = resolve_variant(backend, kernel, tuple(args), {},
                                         build=backend._build_variant,
                                         specialize_disjoint=cpu_mod._SPECIALIZE_DISJOINT)
    _, extents = cpu_mod._get_launch(variant.ir, list(effective))
    return variant.payload, variant.payload.bind(list(effective), extents)


def _backend(monkeypatch, threads=8, subsets=True, fan_out_us=10.0):
    monkeypatch.delenv("TACK_CPU_POOL", raising=False)
    if subsets:
        monkeypatch.setenv("TACK_CPU_SUBSETS", "1")
    else:
        monkeypatch.delenv("TACK_CPU_SUBSETS", raising=False)
    backend = CPUBackend(num_threads=threads)
    backend._fan_out_ns = fan_out_us * 1e3
    return backend


def _model(compiled, r_s, r_p=0.0, recheck_after=1024):
    compiled.ns_per_elem = r_s
    compiled.ns_per_elem_parallel = r_p
    compiled.rate_confirmed = True
    compiled.subset_runs, compiled.subset_recheck_after = 0, recheck_after


def test_subsets_are_off_by_default(monkeypatch):
    backend = _backend(monkeypatch, subsets=False)
    compiled, _ = _compiled(backend, _scale, [tack.zeros(tack.f32, (8,)), tack.zeros(tack.f32, (8,))])
    _model(compiled, r_s=1.0, r_p=0.5)        # a speedup of 2 on 8 threads
    assert backend._threads_for(compiled, 1_000) == 8
    assert backend._threads_for(compiled, 1_000_000) == 8


def test_the_size_rule_wakes_only_the_workers_a_short_launch_can_feed(monkeypatch):
    """Each worker's chunk must cost `_SUBSET_WORK_RATIO` shares of the
    fan-out: at 10 us over 8 threads a share is 1.25 us, so a 50 us
    launch gets 4 threads, a 10 us launch the minimum of 2, and a 1 ms
    launch every thread. Nothing is known about r_p yet."""
    backend = _backend(monkeypatch, fan_out_us=10.0)
    compiled, _ = _compiled(backend, _scale, [tack.zeros(tack.f32, (8,)), tack.zeros(tack.f32, (8,))])
    _model(compiled, r_s=1.0)
    assert backend._threads_for(compiled, 50_000) == 4
    assert backend._threads_for(compiled, 10_000) == 2
    assert backend._threads_for(compiled, 1_000_000) == 8


def test_the_saturation_rule_sizes_a_bandwidth_bound_kernel(monkeypatch):
    """A kernel that fans out to a speedup of 2 on 8 threads is bandwidth
    bound: ceil(2 * 2) = 4 threads reach the same throughput. One that
    speeds up 3x keeps every thread, since ceil(2 * 3) = 6 is more than
    half of them: only a clear saturation trims, as a compute-bound
    kernel on a hyperthreaded host (S about N / 1.6) must not be."""
    backend = _backend(monkeypatch)
    compiled, _ = _compiled(backend, _scale, [tack.zeros(tack.f32, (8,)), tack.zeros(tack.f32, (8,))])
    _model(compiled, r_s=1.0, r_p=0.5)
    assert backend._threads_for(compiled, 10_000_000) == 4
    _model(compiled, r_s=1.0, r_p=1.0 / 3)
    assert backend._threads_for(compiled, 10_000_000) == 8
    _model(compiled, r_s=1.0, r_p=1.0 / 1.2)
    assert backend._threads_for(compiled, 10_000_000) == math.ceil(1.2 * 2)


def test_a_full_run_is_taken_on_the_back_off_schedule(monkeypatch):
    """Only full runs measure r_p, so after 1, 2, 4, ... subset runs one
    launch uses every thread again, and it also retries a retired subset."""
    backend = _backend(monkeypatch)
    compiled, _ = _compiled(backend, _scale, [tack.zeros(tack.f32, (8,)), tack.zeros(tack.f32, (8,))])
    _model(compiled, r_s=1.0, r_p=0.5, recheck_after=1)
    seen = [backend._threads_for(compiled, 10_000_000) for _ in range(16)]
    full = [i for i, k in enumerate(seen) if k == 8]
    assert set(seen) == {4, 8}
    assert full == [1, 4, 9], seen            # after 1, then 2, then 4 subset runs
    compiled.subset_failed = True
    assert backend._threads_for(compiled, 10_000_000) == 8   # held at every thread
    while backend._threads_for(compiled, 10_000_000) == 8:
        pass                                   # the full run clears the retirement
    assert not compiled.subset_failed


def test_a_subset_slower_than_the_full_rate_is_retired(monkeypatch):
    """A subset run's aggregate rate is checked against r_p at N and never
    folded into it: a bad one retires the subset, a good one changes
    nothing, and r_p is left to the full runs."""
    backend = _backend(monkeypatch)
    compiled, _ = _compiled(backend, _scale, [tack.zeros(tack.f32, (8,)), tack.zeros(tack.f32, (8,))])
    _model(compiled, r_s=1.0, r_p=0.5)
    backend._record_parallel_cost(compiled, [1.4, 1.5, 1.6], 3, subset=True)   # 0.5 per element
    assert not compiled.subset_failed and compiled.ns_per_elem_parallel == 0.5
    backend._record_parallel_cost(compiled, [2.0, 2.1, 2.2], 3, subset=True)   # 0.7: too slow
    assert compiled.subset_failed and compiled.ns_per_elem_parallel == 0.5
    backend._record_parallel_cost(compiled, [4.0] * 8, 8)                      # a full run
    assert compiled.ns_per_elem_parallel == 0.5                                # smoothed toward 0.5


def test_a_subset_launch_runs_on_k_threads_and_computes_the_range(monkeypatch):
    backend = _backend(monkeypatch, threads=4)
    n = 300_000
    x, out = tack.zeros(tack.f32, (n,)), tack.zeros(tack.f32, (n,))
    x.from_numpy(np.arange(n, dtype=np.float32))
    compiled, prefix = _compiled(backend, _scale, [x, out])
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)   # makes the pool, measures
    _model(compiled, r_s=1.0, r_p=1.0)                                    # no speedup of 4: 2 threads
    out.fill(0)                      # before the spy: a large fill is itself a pool run
    used = []
    real = backend._cpool.run

    def spy(wrapper, pack, start, end, threads, measure=False):
        used.append(threads)
        return real(wrapper, pack, start, end, threads, measure=measure)
    monkeypatch.setattr(backend._cpool, "run", spy)
    backend._parallel_execute(compiled, prefix, 0, n, whole_range=True)
    assert used == [2]
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.float32) * 2 + 1)
    assert compiled.ns_per_elem_parallel == 1.0          # the subset taught the model nothing

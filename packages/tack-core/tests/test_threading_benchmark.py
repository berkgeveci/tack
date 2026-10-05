"""Measured crossover coverage and cached-threshold diagnostic provenance."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / 'benchmarks' / 'threading_decisions.py'


@pytest.fixture
def benchmark():
    spec = importlib.util.spec_from_file_location('threading_benchmark', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _points(samples):
    return [{'n': n, 'serial_ns': serial, 'parallel_ns': parallel}
            for n, serial, parallel in samples]


def _report(benchmark, samples, hot=200, parallel_slope=1):
    rows = [{**point, 'kernel': 'heavy', 'ns_per_elem': 2,
             'parallel_min_elems': 200} for point in _points(samples)]
    model_points = _points([(n, 2 * n, parallel_slope * n)
                            for n in (1000, 2000, 3000)])
    floor = {0: {'min': hot, 'median': hot, 'max': hot},
             10: {'min': 300, 'median': 300, 'max': 300}}
    backend = SimpleNamespace(_fan_out_ns=200, num_threads=8)
    return benchmark.model_report(rows, {'heavy': model_points}, floor, backend, 2)['heavy']


def test_fitted_crossing_inside_grid_does_not_prove_a_measured_bracket(benchmark, capsys):
    # The fitted crossing is 200, within the 100..300 grid, while every
    # directly measured call is faster serially: the ROCm report's failure.
    result = _report(benchmark, [(100, 200, 300), (200, 400, 550), (300, 600, 800)])
    assert 100 < result['crossover_hot'] < 300
    assert result['brackets'] != 'brackets it'
    assert 'XX heavy' in capsys.readouterr().out


@pytest.mark.parametrize('hot,parallel_slope', [(200, 1), (500, 1), (200, 3)],
                         ids=['fit-inside', 'fit-above', 'no-fitted-crossing'])
def test_measured_bracket_is_independent_of_fitted_crossing(benchmark, hot, parallel_slope):
    result = _report(benchmark, [(100, 200, 300), (200, 400, 350), (300, 600, 400)],
                     hot, parallel_slope)
    assert result['brackets'] == 'brackets it'
    assert result['serial_win_sizes'] == [100]
    assert result['parallel_win_sizes'] == [200, 300]
    assert result['bracket_lo'] == 100
    assert result['bracket_hi'] == 200
    assert result['bracket_basis'] == 'measured timings'


@pytest.mark.parametrize('samples,message', [
    ([], 'no measured points'),
    ([(1, 10, 20), (2, 20, 30)], 'no measured parallel wins'),
    ([(1, 20, 10), (2, 30, 20)], 'no measured serial wins'),
    ([(1, 10, 10), (2, 20, 20)], 'all measured points tie'),
    ([(1, 10, 20), (2, 20, 20)], 'no measured parallel wins'),
    ([(1, 20, 10), (2, 20, 20)], 'no measured serial wins'),
    ([(1, 20, 10), (2, 20, 30)], 'non-monotonic measured winners'),
    ([(1, 10, 20), (2, 30, 20), (3, 20, 30)], 'non-monotonic measured winners'),
    ([(1, 10, 20), (1, 20, 10)], 'non-monotonic measured winners'),
])
def test_unbracketed_or_ambiguous_measurements_are_reported(benchmark, samples, message):
    result = benchmark._measured_bracket(_points(samples))
    assert result['brackets'] == message
    assert result['bracket_lo'] is None
    assert result['bracket_hi'] is None


def test_ties_are_neutral_and_samples_need_not_be_sorted(benchmark):
    points = _points([(400, 40, 30), (100, 10, 20), (200, 20, 20), (300, 30, 25)])
    result = benchmark._measured_bracket(points)
    assert result['brackets'] == 'brackets it'
    assert result['serial_win_sizes'] == [100]
    assert result['parallel_win_sizes'] == [300, 400]
    assert result['tie_sizes'] == [200]
    assert (result['bracket_lo'], result['bracket_hi']) == (100, 300)


def test_report_exposes_measured_counts_and_fit_separately(benchmark, capsys):
    result = _report(benchmark, [(100, 200, 300), (200, 400, 400), (300, 600, 500)])
    assert result['fitted_crossover_location'] == 'inside grid'
    assert result['tie_sizes'] == [200]
    output = capsys.readouterr().out
    assert 'serial/parallel/ties: 1/1/1' in output
    assert 'fitted hot crossing:' in output
    assert 'brackets it' in output


def test_non_monotonic_report_requests_remeasurement(benchmark, capsys):
    result = _report(benchmark, [(100, 200, 150), (200, 400, 450), (300, 600, 500)])
    assert result['brackets'] == 'non-monotonic measured winners'
    output = capsys.readouterr().out
    assert 'Repeat ambiguous timings' in output


@pytest.mark.parametrize('change', ['idle-gap', 'curve-update'])
def test_later_fan_out_read_cannot_replace_the_cached_threshold(benchmark, monkeypatch, change):
    from tack.runtime import cpu

    clock = [0]
    monkeypatch.setattr(cpu, 'time', SimpleNamespace(perf_counter_ns=lambda: clock[0]))
    backend = cpu.CPUBackend(num_threads=8)
    backend.policy = 'v2'
    backend._v2 = True
    backend.margin_override = 1.0
    backend._last_dispatch_ns = 0
    backend._fan_out_curve = [(0, 100.0), (100, 1000.0)]
    backend._fan_out_knot_measured = [True, True]
    stored = backend._min_elems(2.0)
    assert stored == 50
    if change == 'idle-gap':
        clock[0] = 100
    else:
        backend._fan_out_curve[0] = (0, 1000.0)
    row = {'n': 100, 'parallel_min_elems': stored,
           'ns_per_elem': 2.0, 'ns_per_elem_parallel': 0.0,
           'margin': backend._margin(), 'fan_out_estimate_ns': backend._fan_out_estimate(),
           'chose': 'parallel', 'regret_us': 0.0,
           'threshold_inputs_basis': 'post-settlement live snapshot'}
    before = copy.deepcopy(row)
    check = benchmark._threshold_snapshot(row, backend.policy, backend.num_threads)
    assert check['threshold_snapshot_status'] == 'different'
    assert check['snapshot_parallel_min_elems'] == 500
    assert check['snapshot_choice_differs'] is True
    assert check['threshold_inputs_basis'] == 'post-settlement live snapshot'
    assert row == before                    # preserves the scored choice/threshold


@pytest.mark.parametrize('policy,rp,stored', [('v1', 20, 20), ('v2', 5, 40), ('v2', 20, 200)])
def test_snapshot_check_uses_the_recorded_policy_and_capped_parallel_rate(
    benchmark, policy, rp, stored,
):
    row = {'n': 256, 'ns_per_elem': 10, 'ns_per_elem_parallel': rp,
           'margin': 2, 'fan_out_estimate_ns': 100, 'parallel_min_elems': stored}
    check = benchmark._threshold_snapshot(row, policy, 8)
    assert check['threshold_snapshot_status'] == 'consistent'
    assert check['snapshot_parallel_min_elems'] == stored
    assert check['threshold_inputs_basis'] == 'unrecorded'  # consistency is not provenance


@pytest.mark.parametrize('threads,rs,stored', [(1, 10, 1 << 62), (8, 0, 1 << 62), (8, 100, 8)])
def test_snapshot_check_handles_never_parallel_and_thread_floor(benchmark, threads, rs, stored):
    row = {'n': 256, 'ns_per_elem': rs, 'ns_per_elem_parallel': 0,
           'margin': 1, 'fan_out_estimate_ns': 100, 'parallel_min_elems': stored}
    assert benchmark._threshold_snapshot(row, 'v2', threads)['snapshot_parallel_min_elems'] == stored


@pytest.mark.parametrize('missing', ['margin', 'fan_out_estimate_ns', 'ns_per_elem_parallel'])
def test_legacy_rows_without_inputs_do_not_invent_threshold_provenance(benchmark, missing):
    row = {'n': 100, 'ns_per_elem': 2, 'ns_per_elem_parallel': 0,
           'margin': 1, 'fan_out_estimate_ns': 100, 'parallel_min_elems': 50}
    del row[missing]
    check = benchmark._threshold_snapshot(row, 'v2', 8)
    assert check['threshold_snapshot_status'] == 'unavailable'
    assert check['snapshot_parallel_min_elems'] is None
    assert check['snapshot_choice_differs'] is None


@pytest.mark.parametrize('policy,threads,fan_out', [
    (None, 8, 100), ('unknown', 8, 100), ('v2', None, 100),
    ('v2', 8, float('nan')), ('v2', 8, float('inf')),
])
def test_snapshot_check_requires_known_policy_and_finite_inputs(benchmark, policy, threads, fan_out):
    row = {'n': 100, 'ns_per_elem': 2, 'ns_per_elem_parallel': 0,
           'margin': 1, 'fan_out_estimate_ns': fan_out, 'parallel_min_elems': 50}
    assert benchmark._threshold_snapshot(row, policy, threads)['threshold_snapshot_status'] == (
        'unavailable')


def test_legacy_report_qualifies_mismatches_without_rescoring_or_mutating_rows(
    benchmark, tmp_path, capsys,
):
    rows = [{**point, 'kernel': 'heavy', 'ns_per_elem': 2, 'ns_per_elem_parallel': 0,
             'parallel_min_elems': 200, 'fan_out_estimate_ns': 600, 'margin': 1,
             'chose': 'serial' if point['n'] < 200 else 'parallel', 'regret_us': 0.0}
            for point in _points([(100, 200, 300), (200, 400, 350), (300, 600, 400)])]
    original = copy.deepcopy(rows)
    score = {'wrong': 0, 'over_eager': 0, 'regret_us': 0, 'total': 3}
    result = {'rows': rows, 'score': score,
              'model_pts': {'heavy': _points([(n, 2*n, n) for n in (1000, 2000, 3000)])},
              'floor': {0: {'min': 200, 'median': 200, 'max': 200},
                        10: {'min': 300, 'median': 300, 'max': 300}},
              'conditions_before': {'median_ns': 200, 'tail': 1},
              'conditions_after': {'median_ns': 200, 'tail': 1}}
    backend = SimpleNamespace(policy='v2', num_threads=8, _fan_out_ns=200, margin=1)
    args = SimpleNamespace(scale=1, load=0, json=str(tmp_path/'legacy-report.json'))
    benchmark._report(args, backend, {'host': 'scripted'}, result)
    saved = json.loads((tmp_path/'legacy-report.json').read_text())
    assert rows == original
    assert saved['rows'] == original
    assert saved['score'] == score
    assert saved['threshold_diagnostics']['different'] == 3
    assert saved['threshold_diagnostics']['choice_differences'] == 1
    assert saved['fits']['heavy']['anchor_threshold_snapshot']['threshold_snapshot_status'] == (
        'different')
    assert 'not threshold construction' in saved['fits']['heavy']['backend_fan_out_basis']
    output = capsys.readouterr().out
    assert 'different: 3' in output
    assert 'Construction inputs were not recorded' in output
    assert 'never its reconstruction' in output
    assert 'not component attribution' in output
    assert 'where the threshold error comes from' not in output


def test_score_keeps_cached_choice_and_restores_estimators_before_annotating(
    benchmark, monkeypatch,
):
    from tack.runtime import cpu

    clock = [0]
    monkeypatch.setattr(cpu, 'time', SimpleNamespace(perf_counter_ns=lambda: clock[0]))
    backend = cpu.CPUBackend(num_threads=8)
    backend.policy = 'v2'
    backend._v2 = True
    backend.margin_override = 1.0
    backend._fan_out_ns = 100.0
    backend._last_dispatch_ns = 0
    backend._fan_out_curve = [(0, 100.0), (100, 1000.0)]
    backend._fan_out_knot_measured = [True, True]
    compiled = SimpleNamespace(ns_per_elem=2.0, ns_per_elem_parallel=0.0,
                               parallel_min_elems=backend._min_elems(2.0), scattered=False,
                               bind=lambda args: (), call_range=lambda *args: None)
    clock[0] = 100                       # inputs have changed since construction
    calls = []
    margin, fan_out = backend._margin, backend._fan_out_estimate
    monkeypatch.setattr(backend, '_margin', lambda: (calls.append('margin'), margin())[1])
    monkeypatch.setattr(backend, '_fan_out_estimate',
                        lambda: (calls.append('fan-out'), fan_out())[1])

    def measured_parallel(*args):
        compiled.ns_per_elem = 100.0
        compiled.ns_per_elem_parallel = 50.0
        compiled.parallel_min_elems = 999
        backend._fan_out_curve = [(0, 9000.0)]
        backend._fan_out_cv = 12.0

    monkeypatch.setattr(backend, '_parallel_execute', measured_parallel)
    timings = iter([80, 40])

    def timed(fn, reps):
        fn()
        return next(timings)

    monkeypatch.setattr(benchmark, 'best_of', timed)
    monkeypatch.setattr(benchmark, 'GRIDS', {'cheap': (100,)})
    monkeypatch.setattr(benchmark, 'KERNELS', {'cheap': lambda *args: None})
    monkeypatch.setattr(benchmark, 'variant_for', lambda *args: compiled)
    monkeypatch.setattr(benchmark, 'fan_out_floor', lambda *args: {})
    monkeypatch.setattr(benchmark, 'measure_model_grid', lambda *args: {})
    result = benchmark._score(SimpleNamespace(scale=1), backend, None, None, 100)
    row, = result['rows']
    assert row['parallel_min_elems'] == 50
    assert row['chose'] == 'parallel'
    assert row['regret_us'] == 0
    assert row['threshold_inputs_basis'] == 'post-settlement live snapshot'
    assert row['threshold_snapshot_status'] == 'different'
    assert row['snapshot_parallel_min_elems'] == 500
    assert row['snapshot_choice_differs'] is True
    assert compiled.parallel_min_elems == 50
    assert compiled.ns_per_elem == 2
    assert compiled.ns_per_elem_parallel == 0
    assert compiled.scattered is True    # actual data placement follows the fan-out
    assert backend._fan_out_curve == [(0, 100.0), (100, 1000.0)]
    assert backend._fan_out_cv == 0
    assert calls == ['margin', 'fan-out']  # no diagnostic calibration/read calls

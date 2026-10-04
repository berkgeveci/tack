"""Measured crossover coverage must not be inferred from a fitted line."""

import importlib.util
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

"""The example harness must never discard a wrong result or mislabel a backend."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import tack

SCRIPT = Path(__file__).resolve().parents[1] / 'examples' / 'validate_all.py'


def _load_harness():
    spec = importlib.util.spec_from_file_location('validation_harness', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness():
    return _load_harness()


def test_import_does_not_probe_backends(monkeypatch):
    calls = []
    monkeypatch.setattr(tack, 'init', lambda arch: calls.append(arch))
    _load_harness()
    assert calls == []


@pytest.mark.parametrize('bad_call', range(4), ids=['cold', 'slow-warm', 'fast-warm',
                                                  'last-warm'])
def test_report_rejects_every_incorrect_call(harness, monkeypatch, bad_call):
    # The old implementation verifies only call 3, the fastest warm call,
    # and silently passes when any other call returns a wrong result.
    runs = iter([(9.0, (0,)), (7.0, (1,)), (1.0, (2,)), (5.0, (3,))])
    monkeypatch.setattr(harness, '_time_call', lambda setup, call: next(runs))
    verified = []

    def verify(result):
        verified.append(result)
        return result != bad_call

    kind = 'cold' if bad_call == 0 else 'warm'
    with pytest.raises(AssertionError, match=rf'Vector Add.*cpu.*call {bad_call + 1}.*{kind}'):
        harness._report('Vector Add', 'cpu', None, None, verify)
    assert verified == list(range(bad_call + 1))


def test_report_checks_fresh_results_outside_kernel_timing(harness, monkeypatch, capsys):
    clock = [0.0]
    durations = iter([0.009, 0.007, 0.001, 0.005])
    setups, executed, verified = [], [], []
    monkeypatch.setattr(harness.time, 'perf_counter', lambda: clock[0])

    def setup():
        clock[0] += 100.0
        fields = object()
        setups.append(fields)
        return (fields,)

    def call(fields):
        clock[0] += next(durations)
        executed.append(fields)

    def verify(fields):
        clock[0] += 1000.0
        verified.append(fields)
        return True

    harness._report('Vector Add', 'cpu', setup, call, verify)
    assert len(setups) == 4
    assert executed == verified == setups
    output = capsys.readouterr().out
    assert 'first     9.00 ms' in output
    assert 'warm    1.000 ms' in output
    assert '[OK]' in output


@pytest.mark.parametrize('stage', ['call', 'verification'])
def test_report_identifies_raised_error(harness, stage):
    def broken(result):
        raise ValueError('bad buffer')

    call = broken if stage == 'call' else lambda result: None
    verify = broken if stage == 'verification' else lambda result: True
    with pytest.raises(RuntimeError, match=r'Jacobi.*metal.*call 1.*cold') as error:
        harness._report('Jacobi', 'metal', lambda: (object(),), call, verify)
    assert isinstance(error.value.__cause__, ValueError)


def _fake_backends(harness, monkeypatch, available):
    current = SimpleNamespace(name=None)
    attempts = []

    def init(arch):
        attempts.append(arch)
        if arch not in available:
            raise RuntimeError(f'{arch} device unavailable')
        current.name = arch

    monkeypatch.setattr(harness.tack, 'init', init)
    monkeypatch.setattr(harness, 'get_backend', lambda: current)
    return current, attempts


def _record_workloads(harness, monkeypatch):
    seen = []

    def report(name, backend, setup, call, verify):
        assert harness.get_backend().name == backend
        seen.append((name, backend))

    monkeypatch.setattr(harness, '_report', report)
    return seen


@pytest.mark.parametrize('arch', ['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
def test_explicit_selection_runs_only_requested_backend(harness, monkeypatch, capsys, arch):
    _, attempts = _fake_backends(harness, monkeypatch, {arch})
    seen = _record_workloads(harness, monkeypatch)
    harness.main(['--arch', arch])
    assert set(attempts) == {arch}
    assert len(seen) == 7
    assert {backend for name, backend in seen} == {arch}
    output = capsys.readouterr().out
    assert f'Backends: {arch}' in output
    assert 'All validations passed!' in output


def test_default_selection_runs_all_available_backends(harness, monkeypatch, capsys):
    _, attempts = _fake_backends(harness, monkeypatch, {'cpu', 'metal'})
    seen = _record_workloads(harness, monkeypatch)
    harness.main([])
    assert set(attempts) == {'cpu', 'metal', 'cuda', 'hip', 'level_zero'}
    assert len(seen) == 14
    assert {backend for name, backend in seen} == {'cpu', 'metal'}
    assert 'Backends: cpu, metal' in capsys.readouterr().out


@pytest.mark.parametrize('arch', ['cpu', 'metal', 'cuda', 'hip', 'level_zero'])
def test_unavailable_explicit_backend_fails_without_fallback(harness, monkeypatch, capsys, arch):
    _, attempts = _fake_backends(harness, monkeypatch, set())
    seen = _record_workloads(harness, monkeypatch)
    with pytest.raises(SystemExit) as error:
        harness.main(['--arch', arch])
    assert error.value.code == 2
    assert attempts == [arch]
    assert seen == []
    output = capsys.readouterr()
    assert f'{arch} device unavailable' in output.err
    assert 'All validations passed!' not in output.out


def test_empty_discovery_fails(harness, monkeypatch, capsys):
    _fake_backends(harness, monkeypatch, set())
    seen = _record_workloads(harness, monkeypatch)
    with pytest.raises(SystemExit) as error:
        harness.main([])
    assert error.value.code == 2
    assert seen == []
    output = capsys.readouterr()
    assert 'No available Tack backends' in output.err
    assert 'All validations passed!' not in output.out


def test_backend_mismatch_cannot_be_reported_as_gpu_success(harness, monkeypatch, capsys):
    _fake_backends(harness, monkeypatch, {'cpu'})
    monkeypatch.setattr(harness.tack, 'init', lambda arch: None)
    monkeypatch.setattr(harness, 'get_backend', lambda: SimpleNamespace(name='cpu'))
    with pytest.raises(SystemExit) as error:
        harness.main(['--arch', 'metal'])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert 'metal' in output.err and 'cpu' in output.err
    assert 'All validations passed!' not in output.out


def test_unknown_arch_is_rejected_before_initialization(harness, monkeypatch, capsys):
    _, attempts = _fake_backends(harness, monkeypatch, {'cpu'})
    with pytest.raises(SystemExit) as error:
        harness.main(['--arch', 'unknown'])
    assert error.value.code == 2
    assert attempts == []
    assert "invalid choice: 'unknown'" in capsys.readouterr().err

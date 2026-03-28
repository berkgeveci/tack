"""Tests for GPU statistics and analysis (tack.algorithms.stats)."""

import math
import numpy as np
import pytest
import tack
from tack.algorithms import var, std, norm, absmax, count_nonzero, dot, histogram

_backends = []
for _arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


# --- field.mean() ---

def test_mean(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    assert f.mean() == pytest.approx(2.5)


def test_mean_large(backend):
    n = 10000
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(np.ones(n, dtype=np.float32) * 7.0)
    assert f.mean() == pytest.approx(7.0, rel=1e-5)


# --- var / std ---

def test_var(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    expected = np.var([1, 2, 3, 4])
    assert var(f) == pytest.approx(expected, rel=1e-4)


def test_std(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    expected = np.std([1, 2, 3, 4])
    assert std(f) == pytest.approx(expected, rel=1e-4)


def test_var_constant(backend):
    """Variance of constant field is 0."""
    f = tack.field(dtype=tack.f32, shape=(100,))
    f.fill(5.0)
    assert var(f) == pytest.approx(0.0, abs=1e-4)


# --- norm ---

def test_norm_l2(backend):
    f = tack.field(dtype=tack.f32, shape=(3,))
    f.from_numpy(np.array([3, 4, 0], dtype=np.float32))
    assert norm(f, ord=2) == pytest.approx(5.0, rel=1e-5)


def test_norm_l1(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, -2, 3, -4], dtype=np.float32))
    assert norm(f, ord=1) == pytest.approx(10.0, rel=1e-4)


def test_norm_linf(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, -5, 3, 2], dtype=np.float32))
    assert norm(f, ord=float('inf')) == pytest.approx(5.0, rel=1e-5)


# --- absmax ---

def test_absmax(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, -7, 3, 2], dtype=np.float32))
    assert absmax(f) == pytest.approx(7.0, rel=1e-5)


def test_absmax_positive(backend):
    f = tack.field(dtype=tack.f32, shape=(3,))
    f.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    assert absmax(f) == pytest.approx(3.0, rel=1e-5)


# --- count_nonzero ---

def test_count_nonzero(backend):
    f = tack.field(dtype=tack.f32, shape=(6,))
    f.from_numpy(np.array([0, 1, 0, 0, 5, 3], dtype=np.float32))
    assert count_nonzero(f) == 3


def test_count_nonzero_all(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([1, 2, 3, 4], dtype=np.float32))
    assert count_nonzero(f) == 4


def test_count_nonzero_none(backend):
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.fill(0.0)
    assert count_nonzero(f) == 0


# --- dot ---

def test_dot_basic(backend):
    a = tack.field(dtype=tack.f32, shape=(3,))
    b = tack.field(dtype=tack.f32, shape=(3,))
    a.from_numpy(np.array([1, 2, 3], dtype=np.float32))
    b.from_numpy(np.array([4, 5, 6], dtype=np.float32))
    # 1*4 + 2*5 + 3*6 = 32
    assert dot(a, b) == pytest.approx(32.0, rel=1e-4)


def test_dot_orthogonal(backend):
    """Dot product of orthogonal vectors is 0."""
    a = tack.field(dtype=tack.f32, shape=(3,))
    b = tack.field(dtype=tack.f32, shape=(3,))
    a.from_numpy(np.array([1, 0, 0], dtype=np.float32))
    b.from_numpy(np.array([0, 1, 0], dtype=np.float32))
    assert dot(a, b) == pytest.approx(0.0, abs=1e-6)


# --- histogram ---

def test_histogram_uniform(backend):
    """Uniform distribution should have roughly equal bin counts."""
    n = 10000
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(np.linspace(0, 1, n, dtype=np.float32))
    counts, edges = histogram(f, bins=10, range=(0, 1))

    counts_np = counts.to_numpy()
    assert counts_np.shape == (10,)
    assert edges.shape == (11,)
    assert int(counts_np.sum()) == n
    # Each bin should have ~1000
    for c in counts_np:
        assert 900 < c < 1100


def test_histogram_single_value(backend):
    """All values in one bin."""
    f = tack.field(dtype=tack.f32, shape=(100,))
    f.fill(0.5)
    counts, edges = histogram(f, bins=10, range=(0, 1))
    counts_np = counts.to_numpy()
    assert int(counts_np.sum()) == 100
    # All in the middle bin
    assert counts_np[5] == 100


def test_histogram_auto_range(backend):
    """Range is auto-detected from data."""
    f = tack.field(dtype=tack.f32, shape=(4,))
    f.from_numpy(np.array([10, 20, 30, 40], dtype=np.float32))
    counts, edges = histogram(f, bins=3)
    counts_np = counts.to_numpy()
    assert int(counts_np.sum()) == 4
    assert edges[0] == pytest.approx(10.0)
    assert edges[-1] == pytest.approx(40.0)


# --- End-to-end workflow ---

def test_analysis_workflow(backend):
    """Full analysis pipeline: create data, compute stats, histogram."""
    n = 1000
    np.random.seed(42)
    data_np = np.random.randn(n).astype(np.float32)
    f = tack.field(dtype=tack.f32, shape=(n,))
    f.from_numpy(data_np)

    # Stats
    assert f.mean() == pytest.approx(np.mean(data_np), rel=1e-3)
    assert var(f) == pytest.approx(np.var(data_np), rel=1e-2)
    assert norm(f, ord=2) == pytest.approx(np.linalg.norm(data_np), rel=1e-3)

    # Histogram
    counts, edges = histogram(f, bins=20, range=(-3, 3))
    assert int(counts.to_numpy().sum()) <= n  # some may be outside range
    assert edges[0] == pytest.approx(-3.0)
    assert edges[-1] == pytest.approx(3.0)

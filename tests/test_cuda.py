"""Tests for the CUDA compute backend."""

import numpy as np
import pytest

import pgc


@pytest.fixture(autouse=True)
def cuda_backend():
    """Initialize CUDA backend for all tests in this module."""
    try:
        pgc.init(arch=pgc.cuda)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"CUDA not available: {e}")


# --- Basic correctness ---

def test_vector_add():
    n = 1024
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @pgc.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 2.0
    assert np.allclose(result, expected)


def test_saxpy():
    n = 1024
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 1.0)

    @pgc.kernel
    def saxpy(x, y, out):
        for i in range(x.shape[0]):
            out[i] = 2.0 * x[i] + y[i]

    saxpy(x, y, out)

    result = out.to_numpy()
    expected = 2.0 * 3.0 + 1.0  # 7.0
    assert np.allclose(result, expected)


def test_subtraction():
    n = 512
    a = pgc.field(dtype=pgc.f32, shape=(n,))
    b = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))

    @pgc.kernel
    def sub(a, b, out):
        for i in range(a.shape[0]):
            out[i] = a[i] - b[i]

    sub(a, b, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)


def test_conditional():
    n = 1024
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    data = np.arange(n, dtype=np.float32) - 512.0
    x.from_numpy(data)

    @pgc.kernel
    def relu(x, out):
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                out[i] = x[i]
            else:
                out[i] = 0.0

    relu(x, out)

    result = out.to_numpy()
    expected = np.maximum(data, 0.0)
    assert np.allclose(result, expected)


def test_negation():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def neg(x, out):
        for i in range(x.shape[0]):
            out[i] = -x[i]

    neg(x, out)

    result = out.to_numpy()
    expected = -np.arange(n, dtype=np.float32)
    assert np.allclose(result, expected)


def test_multiple_ops():
    n = 512
    a = pgc.field(dtype=pgc.f32, shape=(n,))
    b = pgc.field(dtype=pgc.f32, shape=(n,))
    c = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))
    c.from_numpy(np.full(n, 2.0, dtype=np.float32))

    @pgc.kernel
    def kern(a, b, c, out):
        for i in range(a.shape[0]):
            out[i] = (a[i] - b[i]) * c[i] / 2.0

    kern(a, b, c, out)

    result = out.to_numpy()
    expected = (10.0 - 3.0) * 2.0 / 2.0  # 7.0
    assert np.allclose(result, expected)


def test_math_sqrt():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    data = np.arange(1, n + 1, dtype=np.float32)
    x.from_numpy(data)

    @pgc.kernel
    def kern(x, out):
        for i in range(x.shape[0]):
            out[i] = sqrt(x[i])

    kern(x, out)

    result = out.to_numpy()
    expected = np.sqrt(data)
    assert np.allclose(result, expected, rtol=1e-5)


def test_cached_reuse():
    """Second call should use cached pipeline."""
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @pgc.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)
    assert np.allclose(out.to_numpy(), 3.0)

    # Second call — cached
    x.from_numpy(np.ones(n, dtype=np.float32) * 5.0)
    add(x, y, out)
    assert np.allclose(out.to_numpy(), 7.0)


def test_large_array():
    """Test with a larger array to exercise multiple thread blocks."""
    n = 1_000_000
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 4.0)

    @pgc.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)

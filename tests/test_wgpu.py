"""Tests for the WebGPU compute backend."""

import numpy as np
import pytest

import pgc


@pytest.fixture(autouse=True)
def wgpu_backend():
    """Initialize WebGPU backend for all tests in this module."""
    try:
        pgc.init(arch=pgc.wgpu)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"WebGPU not available: {e}")


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
    np.testing.assert_allclose(out.to_numpy(), np.arange(n, dtype=np.float32) + 2.0)


def test_saxpy():
    n = 512
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 3.0)

    @pgc.kernel
    def saxpy(x, y, out, alpha, n):
        for i in range(n):
            out[i] = alpha * x[i] + y[i]

    saxpy(x, y, out, 2.5, n)
    expected = 2.5 * np.arange(n, dtype=np.float32) + 3.0
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_math_builtins():
    n = 64
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def compute(out, n):
        for i in range(n):
            out[i] = sqrt(float(i))

    compute(out, n)
    expected = np.sqrt(np.arange(n, dtype=np.float32))
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_if_else():
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def threshold(data, out, thresh, n):
        for i in range(n):
            if data[i] > thresh:
                out[i] = 1.0
            else:
                out[i] = 0.0

    threshold(data, out, 128.0, n)
    result = out.to_numpy()
    assert result[127] == 0.0
    assert result[129] == 1.0


def test_sequential_loop():
    n = 32
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def sum_range(out, n):
        for i in range(n):
            total = 0.0
            for k in range(10):
                total = total + float(k)
            out[i] = total

    sum_range(out, n)
    np.testing.assert_allclose(out.to_numpy(), 45.0)


def test_template():
    n = 64
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.data_oriented
    class Scaler:
        def __init__(self, factor):
            self.factor = factor

        @pgc.func
        def apply(self, val):
            return val * self.factor

    @pgc.kernel
    def transform(cfg: pgc.template(), data, out, n):
        for i in range(n):
            out[i] = cfg.apply(data[i])

    transform(Scaler(3.0), data, out, n)
    np.testing.assert_allclose(out.to_numpy(), np.arange(n, dtype=np.float32) * 3.0)


def test_scalar_packing():
    n = 64
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def many_scalars(out, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, count):
        for i in range(count):
            out[i] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9

    many_scalars(out, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, n)
    np.testing.assert_allclose(out.to_numpy(), 55.0, rtol=1e-4)


def test_local_array():
    n = 32
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def use_local(out, n):
        for i in range(n):
            arr = pgc.local_array(pgc.f32, 4)
            arr[0] = 1.0
            arr[1] = 2.0
            arr[2] = 3.0
            arr[3] = 4.0
            out[i] = arr[0] + arr[1] + arr[2] + arr[3]

    use_local(out, n)
    np.testing.assert_allclose(out.to_numpy(), 10.0)


def test_field_like():
    data = pgc.field_like(np.arange(100, dtype=np.float32))
    assert data.shape == (100,)
    np.testing.assert_allclose(data.to_numpy(), np.arange(100, dtype=np.float32))

"""Tests for pgc.local_array — per-thread private arrays."""

import numpy as np
import pytest
import pgc


# Test on all available backends
def _available_backends():
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "vulkan", "level_zero"]:
        try:
            pgc.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    pgc.init(arch="cpu")
    return backends

_backends = _available_backends()


@pytest.fixture(params=_backends)
def backend(request):
    try:
        pgc.init(arch=request.param)
    except (ImportError, RuntimeError, OSError) as e:
        pytest.skip(f"{request.param} not available: {e}")
    return request.param


def test_local_array_store_load(backend):
    """Write to local array, read back, write to output."""
    n = 64
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
    result = out.to_numpy()
    np.testing.assert_allclose(result, 10.0)


def test_local_array_loop_fill(backend):
    """Fill local array in a loop, sum it."""
    n = 32
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def loop_fill(out, n):
        for i in range(n):
            buf = pgc.local_array(pgc.f32, 8)
            for k in range(8):
                buf[k] = float(k) * 0.5
            total = 0.0
            for k in range(8):
                total = total + buf[k]
            out[i] = total

    loop_fill(out, n)
    expected = sum(k * 0.5 for k in range(8))
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_local_array_int(backend):
    """Local array with integer type."""
    n = 16
    out = pgc.field(dtype=pgc.i32, shape=(n,))

    @pgc.kernel
    def int_local(out, n):
        for i in range(n):
            idx = pgc.local_array(pgc.i32, 3)
            idx[0] = 10
            idx[1] = 20
            idx[2] = 30
            out[i] = idx[0] + idx[1] + idx[2]

    int_local(out, n)
    np.testing.assert_array_equal(out.to_numpy(), 60)


def test_local_array_with_template_size(backend):
    """Local array size from a template parameter (compile-time constant)."""
    n = 16
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.data_oriented
    class Config:
        def __init__(self, sz):
            self.size = sz

    @pgc.kernel
    def tmpl_local(cfg: pgc.template(), out, n):
        for i in range(n):
            buf = pgc.local_array(pgc.f32, cfg.size)
            for k in range(cfg.size):
                buf[k] = float(k)
            total = 0.0
            for k in range(cfg.size):
                total = total + buf[k]
            out[i] = total

    cfg = Config(5)
    tmpl_local(cfg, out, n)
    expected = sum(range(5))
    np.testing.assert_allclose(out.to_numpy(), float(expected), rtol=1e-5)

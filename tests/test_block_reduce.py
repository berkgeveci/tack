"""Tests for pgc.block_sum, pgc.block_max, pgc.block_min."""

import numpy as np
import pytest
import pgc


def _available_backends():
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "level_zero"]:
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


def test_block_sum(backend):
    """Each workgroup sums its elements and writes to partial_sums."""
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    partial = pgc.field(dtype=pgc.f32, shape=(1,))
    data.from_numpy(np.ones(n, dtype=np.float32))

    @pgc.kernel
    def reduce_sum(data, partial, n):
        for i in range(n):
            total = pgc.block_sum(data[i])
            if pgc.thread_id() == 0:
                pgc.atomic_add(partial, 0, total)

    reduce_sum(data, partial, n)
    result = partial.to_numpy()[0]
    # One workgroup of 256 threads, each contributing 1.0
    np.testing.assert_allclose(result, 256.0, rtol=1e-4)


def test_block_max(backend):
    """Block max finds the maximum in each workgroup."""
    if backend == "cpu":
        pytest.skip("block_max with thread_id guard is a GPU-only pattern")
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def reduce_max(data, out, n):
        for i in range(n):
            mx = pgc.block_max(data[i])
            if pgc.thread_id() == 0:
                out[0] = mx

    reduce_max(data, out, n)
    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, 255.0, rtol=1e-4)


def test_block_min(backend):
    """Block min finds the minimum in each workgroup."""
    if backend == "cpu":
        pytest.skip("block_min with thread_id guard is a GPU-only pattern")
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))
    data.from_numpy(np.arange(n, dtype=np.float32) + 10.0)

    @pgc.kernel
    def reduce_min(data, out, n):
        for i in range(n):
            mn = pgc.block_min(data[i])
            if pgc.thread_id() == 0:
                out[0] = mn

    reduce_min(data, out, n)
    result = out.to_numpy()[0]
    np.testing.assert_allclose(result, 10.0, rtol=1e-4)


def test_block_sum_multi_workgroup(backend):
    """Sum across multiple workgroups using atomic accumulation."""
    n = 1024  # 4 workgroups of 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    partial = pgc.field(dtype=pgc.f32, shape=(1,))
    data.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @pgc.kernel
    def multi_sum(data, partial, n):
        for i in range(n):
            total = pgc.block_sum(data[i])
            if pgc.thread_id() == 0:
                pgc.atomic_add(partial, 0, total)

    multi_sum(data, partial, n)
    result = partial.to_numpy()[0]
    # 1024 elements * 2.0 = 2048, accumulated across 4 workgroup block_sums
    np.testing.assert_allclose(result, 2048.0, rtol=1e-3)

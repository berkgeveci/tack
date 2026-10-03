"""Tests for the Metal compute backend."""

import numpy as np
import pytest

import tack


@pytest.fixture(autouse=True)
def metal_backend():
    """Initialize Metal backend for all tests in this module."""
    try:
        tack.init(arch=tack.metal)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Metal not available: {e}")


# --- Basic correctness ---

@pytest.mark.parametrize("dtype", [tack.i8, tack.u8, tack.i16, tack.u16,
                                   tack.i32, tack.u32, tack.i64, tack.u64, tack.f32])
def test_imported_buffer_alias_preserves_order(dtype):
    """Distinct wrappers of an MTLBuffer preserve ordered typed accesses."""
    a = tack.field(dtype=dtype, shape=(7,))
    alias = tack.field_from_ptr(a._buffer.metal_buffer, dtype, (7,), writable=True)
    assert alias._buffer is not a._buffer
    out = tack.field(dtype=dtype, shape=(7,))
    a.fill(19)

    @tack.kernel
    def ordered(a, b, out):
        for i in range(out.shape[0]):
            before = a[i]
            b[i] = before + 2
            out[i] = a[i] - before

    ordered(a, alias, out)
    np.testing.assert_array_equal(a.to_numpy(), np.full(7, 21))
    np.testing.assert_array_equal(out.to_numpy(), np.full(7, 2))


def test_alias_bindings_with_mixed_scalar_packs():
    """Both scalar pack buffers stay correctly bound alongside aliases."""
    a = tack.field(dtype=tack.f32, shape=(7,))
    alias = a.reshape((7,))
    out = tack.field(dtype=tack.f32, shape=(7,))

    @tack.kernel
    def ordered(a, scale, b, increment, out):
        for i in range(out.shape[0]):
            a[i] = scale
            b[i] = a[i] + increment
            out[i] = a[i]

    for scale, increment in ((1.5, 2), (3.5, 4), (0.5, 1)):
        ordered(a, scale, alias, increment, out)
        np.testing.assert_array_equal(out.to_numpy(), np.full(7, scale + increment))


def test_atomic_updates_through_aliases():
    """Indirect buffer bindings keep atomic updates visible through aliases."""
    out = tack.field(dtype=tack.i32, shape=(1,))
    alias = out.reshape((1,))
    out.fill(0)

    @tack.kernel
    def increment(a, b, n):
        for i in range(n):
            tack.atomic_add(a, 0, 1)
            tack.atomic_add(b, 0, 1)

    increment(out, alias, 257)
    np.testing.assert_array_equal(out.to_numpy(), [514])


def test_vector_add():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    expected = np.arange(n, dtype=np.float32) + 2.0
    assert np.allclose(result, expected)


def test_saxpy():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 1.0)

    @tack.kernel
    def saxpy(x, y, out):
        for i in range(x.shape[0]):
            out[i] = 2.0 * x[i] + y[i]

    saxpy(x, y, out)

    result = out.to_numpy()
    expected = 2.0 * 3.0 + 1.0  # 7.0
    assert np.allclose(result, expected)


def test_subtraction():
    n = 512
    a = tack.field(dtype=tack.f32, shape=(n,))
    b = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))

    @tack.kernel
    def sub(a, b, out):
        for i in range(a.shape[0]):
            out[i] = a[i] - b[i]

    sub(a, b, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)


def test_conditional():
    n = 1024
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    data = np.arange(n, dtype=np.float32) - 512.0
    x.from_numpy(data)

    @tack.kernel
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
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.arange(n, dtype=np.float32))

    @tack.kernel
    def neg(x, out):
        for i in range(x.shape[0]):
            out[i] = -x[i]

    neg(x, out)

    result = out.to_numpy()
    expected = -np.arange(n, dtype=np.float32)
    assert np.allclose(result, expected)


def test_multiple_ops():
    n = 512
    a = tack.field(dtype=tack.f32, shape=(n,))
    b = tack.field(dtype=tack.f32, shape=(n,))
    c = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    a.from_numpy(np.full(n, 10.0, dtype=np.float32))
    b.from_numpy(np.full(n, 3.0, dtype=np.float32))
    c.from_numpy(np.full(n, 2.0, dtype=np.float32))

    @tack.kernel
    def kern(a, b, c, out):
        for i in range(a.shape[0]):
            out[i] = (a[i] - b[i]) * c[i] / 2.0

    kern(a, b, c, out)

    result = out.to_numpy()
    expected = (10.0 - 3.0) * 2.0 / 2.0  # 7.0
    assert np.allclose(result, expected)


def test_math_sqrt():
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    data = np.arange(1, n + 1, dtype=np.float32)
    x.from_numpy(data)

    @tack.kernel
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
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    y.from_numpy(np.ones(n, dtype=np.float32) * 2.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)
    assert np.allclose(out.to_numpy(), 3.0)

    # Second call — cached
    x.from_numpy(np.ones(n, dtype=np.float32) * 5.0)
    add(x, y, out)
    assert np.allclose(out.to_numpy(), 7.0)


def test_atomic_add():
    """Test atomic_add on Metal."""
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(1,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    out.from_numpy(np.zeros(1, dtype=np.float32))

    @tack.kernel
    def sum_kernel(x, out):
        for i in range(x.shape[0]):
            tack.atomic_add(out, 0, x[i])

    sum_kernel(x, out)
    np.testing.assert_allclose(out.to_numpy()[0], 256.0, rtol=1e-5)


def test_atomic_min_max():
    """Test atomic_min and atomic_max on Metal."""
    n = 64
    x = tack.field(dtype=tack.f32, shape=(n,))
    min_out = tack.field(dtype=tack.f32, shape=(1,))
    max_out = tack.field(dtype=tack.f32, shape=(1,))

    data = np.arange(n, dtype=np.float32) - 20.0
    x.from_numpy(data)
    min_out.from_numpy(np.array([1e10], dtype=np.float32))
    max_out.from_numpy(np.array([-1e10], dtype=np.float32))

    @tack.kernel
    def minmax_kernel(x, min_out, max_out):
        for i in range(x.shape[0]):
            tack.atomic_min(min_out, 0, x[i])
            tack.atomic_max(max_out, 0, x[i])

    minmax_kernel(x, min_out, max_out)
    np.testing.assert_allclose(min_out.to_numpy()[0], -20.0)
    np.testing.assert_allclose(max_out.to_numpy()[0], 43.0)


def test_range_with_step():
    """Test for-loop with step on Metal."""
    n = 64
    out = tack.field(dtype=tack.f32, shape=(n,))

    @tack.kernel
    def step_test(out):
        for i in range(out.shape[0]):
            out[i] = 0.0
            for j in range(0, 10, 3):
                out[i] = out[i] + float(j)

    step_test(out)
    # 0 + 3 + 6 + 9 = 18
    np.testing.assert_allclose(out.to_numpy(), 18.0)


def test_shared_memory():
    """Test shared memory, thread_id, and barrier on Metal."""
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))

    @tack.kernel
    def shared_test(x, out):
        smem = tack.shared(tack.f32, 256)
        for i in range(x.shape[0]):
            tid = tack.thread_id()
            smem[tid] = x[i] * 2.0
            tack.barrier()
            out[i] = smem[tid]

    shared_test(x, out)
    expected = np.arange(n, dtype=np.float32) * 2.0
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_gpu_reductions():
    """Test GPU-side field.sum(), field.min(), field.max()."""
    n = 10000
    x = tack.field(dtype=tack.f32, shape=(n,))
    data = np.arange(n, dtype=np.float32)
    x.from_numpy(data)

    np.testing.assert_allclose(x.sum(), data.sum(), rtol=1e-5)
    np.testing.assert_allclose(x.min(), data.min())
    np.testing.assert_allclose(x.max(), data.max())

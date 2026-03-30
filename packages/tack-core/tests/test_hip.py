"""Tests for the HIP compute backend (AMD GPUs)."""

import numpy as np
import pytest

import tack


@pytest.fixture(autouse=True)
def hip_backend():
    """Initialize HIP backend for all tests in this module."""
    try:
        tack.init(arch=tack.hip)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"HIP not available: {e}")


# --- Basic correctness ---

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


def test_large_array():
    """Test with a larger array to exercise multiple thread blocks."""
    n = 1_000_000
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    y.from_numpy(np.ones(n, dtype=np.float32) * 4.0)

    @tack.kernel
    def add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    add(x, y, out)

    result = out.to_numpy()
    assert np.allclose(result, 7.0)


# --- Loops and control flow ---

def test_nested_loops():
    """Test matrix-like operation with nested loops."""
    n = 16
    a = tack.field(dtype=tack.f32, shape=(n * n,))
    out = tack.field(dtype=tack.f32, shape=(n * n,))

    np_a = np.arange(n * n, dtype=np.float32)
    a.from_numpy(np_a)

    @tack.kernel
    def scale_2d(a, out):
        for i in range(16):
            for j in range(16):
                out[i * 16 + j] = a[i * 16 + j] * 2.0

    scale_2d(a, out)
    np.testing.assert_allclose(out.to_numpy(), np_a * 2.0)


def test_while_loop():
    n = 10
    out = tack.field(dtype=tack.f32, shape=(n,))

    @tack.kernel
    def count_up(out):
        for i in range(10):
            val = 0.0
            j = 0
            while j < 5:
                val = val + 1.0
                j = j + 1
            out[i] = val

    count_up(out)
    np.testing.assert_allclose(out.to_numpy(), 5.0)


def test_range_with_step_sequential():
    """Test nested for-loop with step: range(start, end, step)."""
    n = 10
    out = tack.field(dtype=tack.f32, shape=(n,))

    @tack.kernel
    def step_sum(out):
        for i in range(out.shape[0]):
            out[i] = 0.0
            for j in range(0, 10, 2):
                out[i] = out[i] + float(j)

    step_sum(out)
    # 0 + 2 + 4 + 6 + 8 = 20
    np.testing.assert_allclose(out.to_numpy(), 20.0)


def test_range_with_step_parallel():
    """Test top-level parallel for-loop with step."""
    out = tack.field(dtype=tack.f32, shape=(50,))

    @tack.kernel
    def parallel_step(out):
        for i in range(0, 100, 2):
            out[i // 2] = float(i)

    parallel_step(out)
    expected = np.arange(0, 100, 2, dtype=np.float32)
    np.testing.assert_allclose(out.to_numpy(), expected)


# --- Math builtins ---

def test_min_max():
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    y = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    np_x = np.random.randn(n).astype(np.float32)
    np_y = np.random.randn(n).astype(np.float32)
    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @tack.kernel
    def element_min(x, y, out):
        for i in range(x.shape[0]):
            out[i] = min(x[i], y[i])

    element_min(x, y, out)
    np.testing.assert_allclose(out.to_numpy(), np.minimum(np_x, np_y))


def test_abs():
    n = 256
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    np_x = np.random.randn(n).astype(np.float32)
    x.from_numpy(np_x)

    @tack.kernel
    def apply_abs(x, out):
        for i in range(x.shape[0]):
            out[i] = abs(x[i])

    apply_abs(x, out)
    np.testing.assert_allclose(out.to_numpy(), np.abs(np_x))


# --- Augmented assignment ---

def test_augmented_assignment():
    n = 100
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    out.from_numpy(np.arange(n, dtype=np.float32))

    @tack.kernel
    def add_in_place(x, out):
        for i in range(x.shape[0]):
            out[i] += x[i]

    add_in_place(x, out)
    expected = np.arange(n, dtype=np.float32) + 1.0
    np.testing.assert_allclose(out.to_numpy(), expected)


# --- Atomics ---

def test_atomic_add():
    """Test atomic_add on HIP."""
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
    """Test atomic_min and atomic_max on HIP."""
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


# --- Shared memory ---

def test_shared_memory():
    """Test shared memory, thread_id, and barrier on HIP."""
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


# --- Reductions ---

def test_field_reductions():
    """Test GPU-side field.sum(), field.min(), field.max()."""
    n = 10000
    x = tack.field(dtype=tack.f32, shape=(n,))
    data = np.arange(n, dtype=np.float32)
    x.from_numpy(data)

    np.testing.assert_allclose(x.sum(), data.sum(), rtol=1e-5)
    np.testing.assert_allclose(x.min(), data.min())
    np.testing.assert_allclose(x.max(), data.max())


# --- Print ---

def test_print_in_kernel():
    """Test that print() in kernel doesn't crash (HIP supports printf)."""
    n = 3
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(x.shape[0]):
            print("val:", x[i])
            out[i] = x[i] * 2.0

    kern(x, out)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 4.0, 6.0])

"""Tests for the CPU JIT backend — validates end-to-end compilation and execution."""

import numpy as np
import pytest
import pgc


@pytest.fixture(autouse=True)
def init_cpu():
    pgc.init(arch=pgc.cpu)


def test_vector_add():
    n = 1024
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(n, dtype=np.float32)
    np_y = np.ones(n, dtype=np.float32) * 2.0
    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @pgc.kernel
    def vector_add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    vector_add(x, y, out)

    result = out.to_numpy()
    expected = np_x + np_y
    np.testing.assert_allclose(result, expected)


def test_saxpy():
    n = 512
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(n, dtype=np.float32)
    np_y = np.ones(n, dtype=np.float32) * 3.0
    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @pgc.kernel
    def saxpy(x, y, out):
        for i in range(x.shape[0]):
            out[i] = 2.0 * x[i] + y[i]

    saxpy(x, y, out)

    result = out.to_numpy()
    expected = 2.0 * np_x + np_y
    np.testing.assert_allclose(result, expected)


def test_negation():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(n, dtype=np.float32) - 128.0
    x.from_numpy(np_x)

    @pgc.kernel
    def negate(x, out):
        for i in range(x.shape[0]):
            out[i] = -x[i]

    negate(x, out)
    np.testing.assert_allclose(out.to_numpy(), -np_x)


def test_constant_range():
    """Test kernel with a constant loop range (not derived from field shape)."""
    n = 100
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def fill(out):
        for i in range(100):
            out[i] = 42.0

    fill(out)
    np.testing.assert_allclose(out.to_numpy(), 42.0)


def test_conditional():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(n, dtype=np.float32) - 128.0
    x.from_numpy(np_x)

    @pgc.kernel
    def clamp_positive(x, out):
        for i in range(x.shape[0]):
            if x[i] > 0.0:
                out[i] = x[i]
            else:
                out[i] = 0.0

    clamp_positive(x, out)
    expected = np.maximum(np_x, 0.0)
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_nested_loops():
    """Test matrix-like operation with nested loops."""
    n = 16
    a = pgc.field(dtype=pgc.f32, shape=(n * n,))
    out = pgc.field(dtype=pgc.f32, shape=(n * n,))

    np_a = np.arange(n * n, dtype=np.float32)
    a.from_numpy(np_a)

    @pgc.kernel
    def scale_2d(a, out):
        for i in range(16):
            for j in range(16):
                out[i * 16 + j] = a[i * 16 + j] * 2.0

    scale_2d(a, out)
    np.testing.assert_allclose(out.to_numpy(), np_a * 2.0)


def test_while_loop():
    n = 10
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
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


def test_math_sqrt():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.arange(1, n + 1, dtype=np.float32)
    x.from_numpy(np_x)

    @pgc.kernel
    def apply_sqrt(x, out):
        for i in range(x.shape[0]):
            out[i] = sqrt(x[i])

    apply_sqrt(x, out)
    expected = np.sqrt(np_x)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-5)


def test_min_max():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.random.randn(n).astype(np.float32)
    np_y = np.random.randn(n).astype(np.float32)
    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @pgc.kernel
    def element_min(x, y, out):
        for i in range(x.shape[0]):
            out[i] = min(x[i], y[i])

    element_min(x, y, out)
    np.testing.assert_allclose(out.to_numpy(), np.minimum(np_x, np_y))


def test_abs():
    n = 256
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.random.randn(n).astype(np.float32)
    x.from_numpy(np_x)

    @pgc.kernel
    def apply_abs(x, out):
        for i in range(x.shape[0]):
            out[i] = abs(x[i])

    apply_abs(x, out)
    np.testing.assert_allclose(out.to_numpy(), np.abs(np_x))


def test_augmented_assignment():
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    out.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def add_in_place(x, out):
        for i in range(x.shape[0]):
            out[i] += x[i]

    add_in_place(x, out)
    expected = np.arange(n, dtype=np.float32) + 1.0
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_large_parallel():
    """Test with enough elements to trigger multi-threaded execution."""
    n = 100_000
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    y = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    np_x = np.random.randn(n).astype(np.float32)
    np_y = np.random.randn(n).astype(np.float32)
    x.from_numpy(np_x)
    y.from_numpy(np_y)

    @pgc.kernel
    def vector_add(x, y, out):
        for i in range(x.shape[0]):
            out[i] = x[i] + y[i]

    vector_add(x, y, out)
    np.testing.assert_allclose(out.to_numpy(), np_x + np_y)


def test_cached_compilation():
    """Calling the same kernel twice should use the cache."""
    n = 64
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.ones(n, dtype=np.float32))

    @pgc.kernel
    def double(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i] * 2.0

    double(x, out)
    np.testing.assert_allclose(out.to_numpy(), 2.0)

    # Call again — should hit cache
    x.from_numpy(np.ones(n, dtype=np.float32) * 3.0)
    double(x, out)
    np.testing.assert_allclose(out.to_numpy(), 6.0)


def test_atomic_add():
    """Test atomic_add on a field."""
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(1,))

    x.from_numpy(np.ones(n, dtype=np.float32))
    out.from_numpy(np.zeros(1, dtype=np.float32))

    @pgc.kernel
    def sum_kernel(x, out):
        for i in range(x.shape[0]):
            pgc.atomic_add(out, 0, x[i])

    sum_kernel(x, out)
    np.testing.assert_allclose(out.to_numpy()[0], 100.0, rtol=1e-5)


def test_atomic_min_max():
    """Test atomic_min and atomic_max on fields."""
    n = 64
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    min_out = pgc.field(dtype=pgc.f32, shape=(1,))
    max_out = pgc.field(dtype=pgc.f32, shape=(1,))

    data = np.arange(n, dtype=np.float32) - 20.0  # -20 to 43
    x.from_numpy(data)
    min_out.from_numpy(np.array([1e10], dtype=np.float32))
    max_out.from_numpy(np.array([-1e10], dtype=np.float32))

    @pgc.kernel
    def minmax_kernel(x, min_out, max_out):
        for i in range(x.shape[0]):
            pgc.atomic_min(min_out, 0, x[i])
            pgc.atomic_max(max_out, 0, x[i])

    minmax_kernel(x, min_out, max_out)
    np.testing.assert_allclose(min_out.to_numpy()[0], -20.0)
    np.testing.assert_allclose(max_out.to_numpy()[0], 43.0)


def test_field_reductions():
    """Test field.sum(), field.min(), field.max() reduction builtins."""
    n = 100
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    data = np.arange(n, dtype=np.float32)
    x.from_numpy(data)

    assert x.sum() == pytest.approx(data.sum())
    assert x.min() == pytest.approx(data.min())
    assert x.max() == pytest.approx(data.max())


def test_range_with_step_sequential():
    """Test nested for-loop with step: range(start, end, step)."""
    n = 10
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
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
    out = pgc.field(dtype=pgc.f32, shape=(50,))

    @pgc.kernel
    def parallel_step(out):
        for i in range(0, 100, 2):
            out[i // 2] = float(i)

    parallel_step(out)
    expected = np.arange(0, 100, 2, dtype=np.float32)
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_shared_memory():
    """Test shared memory alloc, thread_id, and barrier on CPU."""
    n = 10
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def shared_test(x, out):
        smem = pgc.shared(pgc.f32, 256)
        for i in range(x.shape[0]):
            tid = pgc.thread_id()
            smem[tid] = x[i] * 2.0
            pgc.barrier()
            out[i] = smem[tid]

    shared_test(x, out)
    expected = np.arange(n, dtype=np.float32) * 2.0
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_print_in_kernel():
    """Test that print() in kernel doesn't crash (CPU)."""
    n = 3
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    @pgc.kernel
    def kern(x, out):
        for i in range(x.shape[0]):
            print("val:", x[i])
            out[i] = x[i] * 2.0

    kern(x, out)
    np.testing.assert_allclose(out.to_numpy(), [2.0, 4.0, 6.0])

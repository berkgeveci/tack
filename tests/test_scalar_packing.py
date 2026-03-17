"""Tests for automatic scalar parameter packing into constant buffers.

GPU backends pack scalar params into typed field buffers to reduce buffer
bindings. CPU passes scalars directly as register values (no packing).
Both paths are tested here.
"""

import numpy as np
import pytest
import pgc


# --- Backend parametrization ---

def _available_backends():
    """Return list of available backends for parametrized tests."""
    backends = ["cpu"]
    for arch in ["metal", "cuda", "hip", "vulkan", "level_zero"]:
        try:
            pgc.init(arch=arch)
            backends.append(arch)
        except (ImportError, RuntimeError, OSError):
            pass
    pgc.init(arch="cpu")  # restore
    return backends


_backends = _available_backends()


@pytest.fixture(params=_backends)
def backend(request):
    """Parametrized fixture that runs each test on all available backends."""
    try:
        pgc.init(arch=request.param)
    except (ImportError, RuntimeError, OSError) as e:
        pytest.skip(f"{request.param} not available: {e}")
    return request.param


# --- Tests ---

def test_many_float_scalars(backend):
    """Kernel with 20 float scalar params — packed into __pack_f32__ on GPU."""
    n = 100
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def add_scalars(out, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9,
                    s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, count):
        for i in range(count):
            out[i] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10 + s11 + s12 + s13 + s14 + s15 + s16 + s17 + s18 + s19

    values = [float(i) * 0.5 for i in range(20)]
    add_scalars(out, *values, n)

    result = out.to_numpy()
    expected = sum(values)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_mixed_int_and_float_scalars(backend):
    """Kernel with both int and float scalars — packed into separate buffers."""
    n = 64
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def mixed(out, scale, offset, stride, count):
        for i in range(count):
            out[i] = scale * float(i * stride) + offset

    mixed(out, 2.5, 10.0, 3, n)
    result = out.to_numpy()
    expected = np.array([2.5 * float(i * 3) + 10.0 for i in range(n)], dtype=np.float32)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_scalar_packing_called_twice(backend):
    """Verify packed kernels work on repeated calls with different scalar values."""
    n = 32
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def scale(out, a, b, count):
        for i in range(count):
            out[i] = a * float(i) + b

    scale(out, 3.0, 1.0, n)
    r1 = out.to_numpy().copy()

    scale(out, -1.0, 100.0, n)
    r2 = out.to_numpy().copy()

    np.testing.assert_allclose(r1, 3.0 * np.arange(n, dtype=np.float32) + 1.0, rtol=1e-4)
    np.testing.assert_allclose(r2, -1.0 * np.arange(n, dtype=np.float32) + 100.0, rtol=1e-4)


def test_no_scalars_no_packing(backend):
    """Kernel with only field params — packing is a no-op."""
    n = 64
    x = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    x.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def copy(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i]

    copy(x, out)
    np.testing.assert_allclose(out.to_numpy(), np.arange(n, dtype=np.float32))


def test_numpy_scalar_types(backend):
    """numpy scalar types (np.float32, np.int32) work as kernel args."""
    n = 16
    out = pgc.field(dtype=pgc.f32, shape=(n,))

    @pgc.kernel
    def fill(out, val, count):
        for i in range(count):
            out[i] = val

    fill(out, np.float32(42.5), np.int32(n))
    np.testing.assert_allclose(out.to_numpy(), 42.5)

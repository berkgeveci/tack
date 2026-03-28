"""Tests for scalar parameter matching (Layer 3 of type system refactor).

Validates that Python float scalars match the field dtype context:
- f32 fields + float scalar → f32 parameter
- f64 fields + float scalar → f64 parameter
"""

import numpy as np
import pytest
import tack
from tack.lang import ir
from tack.lang.types import f32, f64, i32, i64
from tack.lang.type_inference import infer_param_types


# --- IR-level tests ---

class TestScalarParamInference:
    @pytest.fixture(autouse=True)
    def _init_cpu(self):
        tack.init(arch=tack.cpu)

    def test_float_with_f32_field(self):
        """Float scalar stays f32 when all fields are f32."""
        func = ir.IRFunction("k", [ir.IRParam("data"), ir.IRParam("alpha")], [])
        data = tack.field(dtype=tack.f32, shape=(10,))
        types = infer_param_types(func, (data, 0.5))
        assert types[1] is f32
        assert func.params[1].type_annotation is f32

    def test_float_with_f64_field(self):
        """Float scalar promotes to f64 when any field is f64."""
        func = ir.IRFunction("k", [ir.IRParam("data"), ir.IRParam("alpha")], [])
        data = tack.field(dtype=tack.f64, shape=(10,))
        types = infer_param_types(func, (data, 0.5))
        assert types[1] is f64
        assert func.params[1].type_annotation is f64

    def test_float_with_mixed_fields(self):
        """Float scalar promotes to f64 if any field is f64."""
        func = ir.IRFunction("k", [
            ir.IRParam("x"), ir.IRParam("y"), ir.IRParam("alpha")
        ], [])
        x = tack.field(dtype=tack.f32, shape=(10,))
        y = tack.field(dtype=tack.f64, shape=(10,))
        types = infer_param_types(func, (x, y, 0.5))
        assert types[2] is f64

    def test_int_unaffected_by_f64_context(self):
        """Integer scalars are not affected by f64 field context."""
        func = ir.IRFunction("k", [ir.IRParam("data"), ir.IRParam("n")], [])
        data = tack.field(dtype=tack.f64, shape=(10,))
        types = infer_param_types(func, (data, 42))
        assert types[1] is i32

    def test_multiple_float_scalars(self):
        """All float scalars in the same call get promoted together."""
        func = ir.IRFunction("k", [
            ir.IRParam("data"), ir.IRParam("a"), ir.IRParam("b")
        ], [])
        data = tack.field(dtype=tack.f64, shape=(10,))
        types = infer_param_types(func, (data, 0.5, 1.0))
        assert types[1] is f64
        assert types[2] is f64

    def test_no_fields_stays_f32(self):
        """Without any fields, float scalars stay f32."""
        func = ir.IRFunction("k", [ir.IRParam("a"), ir.IRParam("b")], [])
        types = infer_param_types(func, (1.0, 2.0))
        assert types[0] is f32
        assert types[1] is f32

    def test_i32_field_doesnt_promote_float(self):
        """Integer fields don't promote float scalars."""
        func = ir.IRFunction("k", [ir.IRParam("idx"), ir.IRParam("t")], [])
        idx = tack.field(dtype=tack.i32, shape=(10,))
        types = infer_param_types(func, (idx, 0.5))
        assert types[1] is f32


# --- End-to-end f64 tests (all backends except Metal) ---

f64_backends = []
for _a in ["cpu", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _a))
        from tack.runtime.dispatch import get_backend as _get_backend
        _be = _get_backend()
        if getattr(_be, 'supports_f64', True):
            f64_backends.append(_a)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=f64_backends)
def f64_backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


def test_saxpy_f64_precision(f64_backend):
    """Float scalar gets f64 precision when used with f64 fields."""
    n = 4
    x = tack.field(dtype=tack.f64, shape=(n,))
    y = tack.field(dtype=tack.f64, shape=(n,))

    # Use values that lose precision in f32
    x.from_numpy(np.array([1.0, 1e-8, 1e15, 1.23456789012345], dtype=np.float64))
    y.from_numpy(np.zeros(n, dtype=np.float64))

    @tack.kernel
    def scale(x, y, alpha):
        for i in range(x.shape[0]):
            y[i] = alpha * x[i]

    scale(x, y, 1.23456789012345)
    result = y.to_numpy()

    # If alpha were f32, this would lose precision
    expected = 1.23456789012345 * np.array([1.0, 1e-8, 1e15, 1.23456789012345])
    np.testing.assert_allclose(result, expected, rtol=1e-14)


def test_f32_fields_keep_f32_scalar(backend):
    """Float scalars stay f32 with f32 fields (no unnecessary promotion)."""
    n = 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @tack.kernel
    def scale(x, out, alpha):
        for i in range(x.shape[0]):
            out[i] = alpha * x[i]

    scale(x, out, 2.0)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [2.0, 4.0, 6.0, 8.0])
    assert result.dtype == np.float32


# --- All-backend tests ---

backends = []
for _b in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _b))
        backends.append(_b)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


def test_f32_scalar_matching_both_backends(backend):
    """Basic f32 scalar matching works on both CPU and Metal."""
    n = 4
    x = tack.field(dtype=tack.f32, shape=(n,))
    out = tack.field(dtype=tack.f32, shape=(n,))
    x.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @tack.kernel
    def add_scalar(x, out, val):
        for i in range(x.shape[0]):
            out[i] = x[i] + val

    add_scalar(x, out, 10.0)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [11.0, 12.0, 13.0, 14.0])

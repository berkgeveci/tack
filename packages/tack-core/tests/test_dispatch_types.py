"""Tests for dispatch-time type checking (Layer 5 of type system refactor).

Validates that unsupported array dtypes are caught with clear errors
before hitting the compiler.
"""

import numpy as np
import pytest

import tack
from tack.lang import ir
from tack.lang.type_inference import check_dispatch_types, infer_param_types
from tack.lang.types import f32, f64, i32, i64, u32, u64

# --- Unit tests for check_dispatch_types ---

class TestCheckDispatchTypes:
    @pytest.fixture(autouse=True)
    def _init_cpu(self):
        tack.set_default_device("cpu")

    def test_supported_dtype_passes(self):
        """f32 array should pass when f32 is supported."""
        func = ir.IRFunction("k", [ir.IRParam("data")], [])
        data = tack.empty((10,))
        infer_param_types(func, (data,))
        # Should not raise
        check_dispatch_types(func, (data,),
                             supported_dtypes={f32, i32},
                             backend_name="Test")

    def test_unsupported_dtype_raises(self):
        """f64 array should raise when f64 is not supported."""
        func = ir.IRFunction("k", [ir.IRParam("data")], [])
        data = tack.empty((10,), tack.f64)
        infer_param_types(func, (data,))
        with pytest.raises(TypeError, match="f64.*not supported.*TestBackend"):
            check_dispatch_types(func, (data,),
                                 supported_dtypes={f32, i32},
                                 backend_name="TestBackend")

    def test_error_names_parameter(self):
        """Error message should include the parameter name."""
        func = ir.IRFunction("k", [ir.IRParam("my_data")], [])
        data = tack.empty((10,), tack.f64)
        infer_param_types(func, (data,))
        with pytest.raises(TypeError, match="my_data"):
            check_dispatch_types(func, (data,),
                                 supported_dtypes={f32},
                                 backend_name="Test")

    def test_error_names_kernel(self):
        """Error message should include the kernel name."""
        func = ir.IRFunction("my_kernel", [ir.IRParam("data")], [])
        data = tack.empty((10,), tack.f64)
        infer_param_types(func, (data,))
        with pytest.raises(TypeError, match="my_kernel"):
            check_dispatch_types(func, (data,),
                                 supported_dtypes={f32},
                                 backend_name="Test")

    def test_scalar_args_not_checked(self):
        """Scalar arguments should not trigger dtype checks."""
        func = ir.IRFunction("k", [ir.IRParam("n")], [])
        infer_param_types(func, (42,))
        # Should not raise even with restrictive supported_dtypes
        check_dispatch_types(func, (42,),
                             supported_dtypes={f32},
                             backend_name="Test")

    def test_multiple_arrays_first_bad(self):
        """First unsupported array should trigger the error."""
        func = ir.IRFunction("k", [ir.IRParam("a"), ir.IRParam("b")], [])
        a = tack.empty((10,), tack.f64)
        b = tack.empty((10,))
        infer_param_types(func, (a, b))
        with pytest.raises(TypeError, match="'a'.*f64"):
            check_dispatch_types(func, (a, b),
                                 supported_dtypes={f32, i32},
                                 backend_name="Test")

    def test_no_supported_dtypes_skips_check(self):
        """When supported_dtypes is None, no check is performed."""
        func = ir.IRFunction("k", [ir.IRParam("data")], [])
        data = tack.empty((10,), tack.f64)
        infer_param_types(func, (data,))
        # Should not raise
        check_dispatch_types(func, (data,),
                             supported_dtypes=None,
                             backend_name="Test")

    def test_all_supported_types_pass(self):
        """All standard types should pass on CPU (full support)."""
        from tack.lang.types import f32, i32
        all_dtypes = {f32, f64, i32, i64, u32, u64}
        for dtype in [tack.f32, tack.f64, tack.i32, tack.i64, tack.u32, tack.u64]:
            func = ir.IRFunction("k", [ir.IRParam("data")], [])
            data = tack.empty((10,), dtype)
            infer_param_types(func, (data,))
            check_dispatch_types(func, (data,),
                                 supported_dtypes=all_dtypes,
                                 backend_name="CPU")


# --- End-to-end: Metal rejects f64 ---

def test_metal_rejects_f64_array():
    """Metal backend should reject f64 arrays with a clear error."""
    try:
        tack.set_default_device("metal")
    except Exception:
        pytest.skip("Metal not available")

    data = tack.empty((10,), tack.f64)
    out = tack.empty((10,), tack.f64)

    @tack.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            out[i] = data[i]

    with pytest.raises(TypeError, match="f64.*not supported.*Metal"):
        kern(data, out)


# --- End-to-end: CPU accepts all types ---

backends = []
try:
    tack.set_default_device("cpu")
    backends.append("cpu")
except Exception:
    pass
try:
    tack.set_default_device("metal")
    backends.append("metal")
except Exception:
    pass


@pytest.fixture(params=backends)
def backend(request):
    tack.set_default_device(request.param)
    return request.param


def test_valid_dtypes_execute_successfully(backend):
    """Valid dtypes should execute without type errors on both backends."""
    n = 4
    x = tack.empty((n,))
    out = tack.empty((n,))
    tack.copyto(x, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @tack.kernel
    def copy(x, out):
        for i in range(x.shape[0]):
            out[i] = x[i]

    copy(x, out)
    np.testing.assert_array_equal(out.numpy(), x.numpy())

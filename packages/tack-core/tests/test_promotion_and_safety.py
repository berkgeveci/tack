"""Tests for signed/unsigned promotion, C keyword escaping, and unsigned casts."""

import numpy as np
import pytest

import tack
from tack.lang.type_inference import promote_types
from tack.lang.types import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

# --- Signed/unsigned promotion rules ---

class TestMixedSignPromotion:
    def test_u8_i8_promotes_to_i16(self):
        assert promote_types(u8, i8) is i16
        assert promote_types(i8, u8) is i16

    def test_u16_i16_promotes_to_i32(self):
        assert promote_types(u16, i16) is i32
        assert promote_types(i16, u16) is i32

    def test_u32_i32_promotes_to_i64(self):
        assert promote_types(u32, i32) is i64
        assert promote_types(i32, u32) is i64

    def test_u64_i64_promotes_to_i64(self):
        """No wider signed type — stays i64."""
        assert promote_types(u64, i64) is i64
        assert promote_types(i64, u64) is i64

    def test_same_sign_same_width(self):
        """Same type returns itself."""
        assert promote_types(i8, i8) is i8
        assert promote_types(u32, u32) is u32
        assert promote_types(f32, f32) is f32

    def test_different_width_signed(self):
        assert promote_types(i8, i32) is i32
        assert promote_types(i16, i64) is i64

    def test_different_width_unsigned(self):
        assert promote_types(u8, u32) is u32
        assert promote_types(u16, u64) is u64

    def test_int_float_promotes_to_float(self):
        assert promote_types(i32, f32) is f32
        assert promote_types(u8, f32) is f32
        assert promote_types(i64, f64) is f64

    def test_f32_f64_promotes_to_f64(self):
        assert promote_types(f32, f64) is f64

    def test_u8_i32_promotes_to_i32(self):
        """Different widths: u8 (rank 0) + i32 (rank 2) → i32."""
        assert promote_types(u8, i32) is i32

    def test_u16_i8_promotes_to_u16(self):
        """u16 (rank 1) > i8 (rank 0) → u16."""
        assert promote_types(u16, i8) is u16


# --- C keyword kernel name escaping ---

def test_kernel_named_double(backend):
    """Kernel named 'double' (C reserved word) should compile and run."""
    out = tack.field(dtype=tack.f32, shape=(4,))

    @tack.kernel
    def double(out):
        for i in range(out.shape[0]):
            out[i] = float(i) * 2.0

    double(out)
    np.testing.assert_allclose(out.to_numpy(), [0.0, 2.0, 4.0, 6.0])


def test_kernel_named_float(backend):
    """Kernel named 'float' (C reserved word) should compile and run."""
    out = tack.field(dtype=tack.i32, shape=(4,))

    @tack.kernel
    def float(out):
        for i in range(out.shape[0]):
            out[i] = i + 1

    float(out)
    np.testing.assert_array_equal(out.to_numpy(), [1, 2, 3, 4])


def test_kernel_named_int(backend):
    """Kernel named 'int' should compile and run."""
    out = tack.field(dtype=tack.f32, shape=(4,))

    # Use globals trick since 'int' shadows the builtin
    @tack.kernel
    def int(out):
        for i in range(out.shape[0]):
            out[i] = 1.0

    int(out)
    np.testing.assert_allclose(out.to_numpy(), [1.0, 1.0, 1.0, 1.0])


def test_codegen_escapes_reserved_name():
    """Verify codegen output uses _tack_ prefix for reserved names."""
    from tack.codegen.cuda_gen import generate_cuda_source
    from tack.lang import ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types

    tack.init(arch=tack.cpu)
    out = tack.field(dtype=tack.f32, shape=(4,))

    p = ir.IRParam("out")
    store = ir.IRFieldStore(ir.IRName("out"), ir.IRName("i"), ir.IRConstant(1.0))
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(4), [store])
    func = ir.IRFunction("double", [p], [pfor])
    infer_param_types(func, (out,))
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "_tack_double" in src
    assert "void double(" not in src


# --- u32→f32 unsigned cast on CPU ---

def test_u32_to_f32_cast(backend):
    """u32 values > 2^31 should cast correctly to f32 (uitofp, not sitofp)."""
    data = tack.field(dtype=tack.u32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    data.from_numpy(np.array([0, 1, 2**31, 2**32 - 1], dtype=np.uint32))

    @tack.kernel
    def cast_u32(data, out):
        for i in range(data.shape[0]):
            out[i] = tack.f32(data[i])

    cast_u32(data, out)
    result = out.to_numpy()
    assert result[0] == 0.0
    assert result[1] == 1.0
    assert result[2] == pytest.approx(2**31, rel=1e-6)
    assert result[3] == pytest.approx(2**32 - 1, rel=1e-6)


def test_u64_to_f64_cast():
    """u64 values should cast correctly to f64."""
    tack.init(arch=tack.cpu)
    data = tack.field(dtype=tack.u64, shape=(3,))
    out = tack.field(dtype=tack.f64, shape=(3,))
    data.from_numpy(np.array([0, 2**32, 2**63], dtype=np.uint64))

    @tack.kernel
    def cast_u64(data, out):
        for i in range(data.shape[0]):
            out[i] = tack.f64(data[i])

    cast_u64(data, out)
    result = out.to_numpy()
    assert result[0] == 0.0
    assert result[1] == pytest.approx(2**32)
    assert result[2] == pytest.approx(2**63, rel=1e-10)

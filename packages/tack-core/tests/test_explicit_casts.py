"""Tests for explicit type casts (Layer 2 of type system refactor).

Tests tack.f32(), tack.f64(), tack.i32(), tack.i64(), tack.u32(), tack.u64()
casts in kernels, running on actual backends.
"""

import numpy as np
import pytest
import tack

backends = []
f64_backends = []
for _arch in ["cpu", "metal", "cuda", "hip", "level_zero"]:
    try:
        tack.init(arch=getattr(tack, _arch))
        backends.append(_arch)
        if _arch == "metal":
            continue  # Metal lacks f64
        from tack.runtime.dispatch import get_backend as _get_backend
        _be = _get_backend()
        if getattr(_be, 'supports_f64', True):
            f64_backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=backends)
def backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


@pytest.fixture(params=f64_backends)
def f64_backend(request):
    tack.init(arch=getattr(tack, request.param))
    return request.param


# --- int() and float() still work ---

def test_int_cast(backend):
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.i32, shape=(4,))
    x.from_numpy(np.array([1.9, 2.1, -0.5, 3.7], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = int(x[i])

    kern(x, out)
    result = out.to_numpy()
    assert result[0] == 1
    assert result[1] == 2
    assert result[3] == 3


def test_float_cast(backend):
    x = tack.field(dtype=tack.i32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = float(x[i]) * 0.5

    kern(x, out)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [0.5, 1.0, 1.5, 2.0])


# --- tack.i32() ---

def test_tack_i32_cast(backend):
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.i32, shape=(4,))
    x.from_numpy(np.array([1.9, 2.1, -0.5, 3.7], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.i32(x[i])

    kern(x, out)
    result = out.to_numpy()
    assert result[0] == 1
    assert result[1] == 2
    assert result[3] == 3


# --- tack.f32() ---

def test_tack_f32_cast(backend):
    x = tack.field(dtype=tack.i32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    x.from_numpy(np.array([1, 2, 3, 4], dtype=np.int32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.f32(x[i]) * 0.5

    kern(x, out)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [0.5, 1.0, 1.5, 2.0])


# --- tack.f64() (all backends except Metal) ---

def test_tack_f64_cast(f64_backend):
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.f64, shape=(4,))
    x.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.f64(x[i])

    kern(x, out)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])
    assert result.dtype == np.float64


# --- tack.i64() ---

def test_tack_i64_cast(backend):
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.i64, shape=(4,))
    x.from_numpy(np.array([1.9, -2.1, 100.7, 0.1], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.i64(x[i])

    kern(x, out)
    result = out.to_numpy()
    assert result[0] == 1
    assert result[1] == -2
    assert result[2] == 100
    assert result[3] == 0


# --- tack.u32() ---

def test_tack_u32_cast(backend):
    x = tack.field(dtype=tack.i32, shape=(4,))
    out = tack.field(dtype=tack.u32, shape=(4,))
    x.from_numpy(np.array([0, 1, 255, 1000], dtype=np.int32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.u32(x[i])

    kern(x, out)
    result = out.to_numpy()
    np.testing.assert_array_equal(result, [0, 1, 255, 1000])
    assert result.dtype == np.uint32


# --- Cast in expressions ---

def test_cast_in_expression(backend):
    """Cast inside a larger expression."""
    x = tack.field(dtype=tack.f32, shape=(4,))
    out = tack.field(dtype=tack.f32, shape=(4,))
    x.from_numpy(np.array([1.5, 2.7, 3.1, 4.9], dtype=np.float32))

    @tack.kernel
    def kern(x, out):
        for i in range(4):
            out[i] = tack.f32(tack.i32(x[i])) + 0.5

    kern(x, out)
    result = out.to_numpy()
    np.testing.assert_allclose(result, [1.5, 2.5, 3.5, 4.5])


# --- Codegen-level tests (no backend needed) ---

def test_cuda_codegen_explicit_cast():
    """CUDA codegen emits correct C types for explicit casts."""
    from tack.lang import ir
    from tack.lang.types import f32, f64, i32, i64, u32
    from tack.lang.ir_type_annotate import annotate_types
    from tack.codegen.cuda_gen import generate_cuda_source

    p = ir.IRParam("x", f32)
    p._is_field = True
    body = [
        ir.IRAssign("a", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), f64)),
        ir.IRAssign("b", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), i64)),
        ir.IRAssign("c", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), u32)),
    ]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p], [pfor])
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "((double)" in src
    assert "((long long)" in src
    assert "((unsigned int)" in src


def test_msl_codegen_explicit_cast():
    """MSL codegen emits correct types for explicit casts."""
    from tack.lang import ir
    from tack.lang.types import f32, i32, i64, u32
    from tack.lang.ir_type_annotate import annotate_types
    from tack.codegen.msl_gen import generate_msl_source

    p = ir.IRParam("x", f32)
    p._is_field = True
    body = [
        ir.IRAssign("a", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), i64)),
        ir.IRAssign("b", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), u32)),
    ]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p], [pfor])
    annotate_types(func)
    src = generate_msl_source(func)

    assert "((long)" in src
    assert "((uint)" in src


def test_msl_codegen_rejects_f64():
    """MSL codegen should reject f64 casts (Apple GPUs don't support double)."""
    from tack.lang import ir
    from tack.lang.types import f32, f64
    from tack.lang.ir_type_annotate import annotate_types
    from tack.codegen.msl_gen import generate_msl_source

    p = ir.IRParam("x", f32)
    p._is_field = True
    body = [
        ir.IRAssign("a", ir.IRCast(ir.IRFieldLoad(ir.IRName("x"), ir.IRName("i")), f64)),
    ]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p], [pfor])
    annotate_types(func)
    with pytest.raises(NotImplementedError, match="f64"):
        generate_msl_source(func)

"""Tests for i8/u8/i16/u16 scalar types."""

import numpy as np
import pytest
import pgc

_backends = []
for _arch in ["cpu", "metal"]:
    try:
        pgc.init(arch=getattr(pgc, _arch))
        _backends.append(_arch)
    except (ImportError, RuntimeError, OSError):
        pass


@pytest.fixture(params=_backends)
def backend(request):
    pgc.init(arch=getattr(pgc, request.param))
    return request.param


# --- Field creation and read/write ---

def test_u8_field(backend):
    n = 16
    data = pgc.field(dtype=pgc.u8, shape=(n,))
    out = pgc.field(dtype=pgc.u8, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.uint8))

    @pgc.kernel
    def copy(data, out):
        for i in range(data.shape[0]):
            out[i] = data[i]

    copy(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.uint8))


def test_i8_field(backend):
    n = 16
    data = pgc.field(dtype=pgc.i8, shape=(n,))
    out = pgc.field(dtype=pgc.i8, shape=(n,))
    data.from_numpy(np.arange(-8, 8, dtype=np.int8))

    @pgc.kernel
    def copy(data, out):
        for i in range(data.shape[0]):
            out[i] = data[i]

    copy(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(-8, 8, dtype=np.int8))


def test_u16_field(backend):
    n = 16
    data = pgc.field(dtype=pgc.u16, shape=(n,))
    out = pgc.field(dtype=pgc.u16, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.uint16) * 1000)

    @pgc.kernel
    def copy(data, out):
        for i in range(data.shape[0]):
            out[i] = data[i]

    copy(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.uint16) * 1000)


def test_i16_field(backend):
    n = 16
    data = pgc.field(dtype=pgc.i16, shape=(n,))
    out = pgc.field(dtype=pgc.i16, shape=(n,))
    data.from_numpy(np.arange(-8, 8, dtype=np.int16) * 100)

    @pgc.kernel
    def copy(data, out):
        for i in range(data.shape[0]):
            out[i] = data[i]

    copy(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(-8, 8, dtype=np.int16) * 100)


# --- Arithmetic ---

def test_u8_arithmetic(backend):
    n = 10
    a = pgc.field(dtype=pgc.u8, shape=(n,))
    out = pgc.field(dtype=pgc.u8, shape=(n,))
    a.from_numpy(np.arange(n, dtype=np.uint8))

    @pgc.kernel
    def double_u8(a, out):
        for i in range(a.shape[0]):
            out[i] = a[i] * 2

    double_u8(a, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.uint8) * 2)


def test_i16_arithmetic(backend):
    n = 10
    a = pgc.field(dtype=pgc.i16, shape=(n,))
    out = pgc.field(dtype=pgc.i16, shape=(n,))
    a.from_numpy(np.arange(n, dtype=np.int16) * 100)

    @pgc.kernel
    def negate(a, out):
        for i in range(a.shape[0]):
            out[i] = 0 - a[i]

    negate(a, out)
    np.testing.assert_array_equal(out.to_numpy(), -np.arange(n, dtype=np.int16) * 100)


# --- Explicit casts ---

def test_cast_f32_to_u8(backend):
    n = 8
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.u8, shape=(n,))
    data.from_numpy(np.array([0.0, 1.9, 50.5, 127.0, 200.0, 254.9, 255.0, 0.1],
                              dtype=np.float32))

    @pgc.kernel
    def cast_kern(data, out):
        for i in range(data.shape[0]):
            out[i] = pgc.u8(data[i])

    cast_kern(data, out)
    result = out.to_numpy()
    assert result[0] == 0
    assert result[1] == 1
    assert result[3] == 127
    assert result[6] == 255


def test_cast_i32_to_i16(backend):
    n = 4
    data = pgc.field(dtype=pgc.i32, shape=(n,))
    out = pgc.field(dtype=pgc.i16, shape=(n,))
    data.from_numpy(np.array([0, 100, -100, 32000], dtype=np.int32))

    @pgc.kernel
    def cast_kern(data, out):
        for i in range(data.shape[0]):
            out[i] = pgc.i16(data[i])

    cast_kern(data, out)
    np.testing.assert_array_equal(out.to_numpy(), [0, 100, -100, 32000])


def test_cast_u8_to_f32(backend):
    n = 4
    data = pgc.field(dtype=pgc.u8, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.array([0, 128, 255, 1], dtype=np.uint8))

    @pgc.kernel
    def cast_kern(data, out):
        for i in range(data.shape[0]):
            out[i] = pgc.f32(data[i])

    cast_kern(data, out)
    np.testing.assert_allclose(out.to_numpy(), [0.0, 128.0, 255.0, 1.0])


# --- Mixed-type operations (promotion) ---

def test_u8_plus_i32_promotes(backend):
    """u8 + i32 should promote to i32."""
    n = 4
    a = pgc.field(dtype=pgc.u8, shape=(n,))
    b = pgc.field(dtype=pgc.i32, shape=(n,))
    out = pgc.field(dtype=pgc.i32, shape=(n,))
    a.from_numpy(np.array([10, 20, 30, 40], dtype=np.uint8))
    b.from_numpy(np.array([1000, 2000, 3000, 4000], dtype=np.int32))

    @pgc.kernel
    def add(a, b, out):
        for i in range(a.shape[0]):
            out[i] = pgc.i32(a[i]) + b[i]

    add(a, b, out)
    np.testing.assert_array_equal(out.to_numpy(), [1010, 2020, 3030, 4040])


# --- Codegen output checks ---

def test_cuda_codegen_i8_u8():
    """CUDA codegen emits correct C types for i8/u8."""
    from pgc.lang import ir
    from pgc.lang.types import u8, i8
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.cuda_gen import generate_cuda_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.u8, shape=(10,))
    out = pgc.field(dtype=pgc.i8, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    store = ir.IRFieldStore(ir.IRName("out"), ir.IRName("i"),
                            ir.IRFieldLoad(ir.IRName("data"), ir.IRName("i")))
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), [store])
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "unsigned char* __restrict__ data" in src
    assert "signed char* __restrict__ out" in src


def test_msl_codegen_u16():
    """MSL codegen emits correct type for u16."""
    from pgc.lang import ir
    from pgc.lang.types import u16
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.msl_gen import generate_msl_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.u16, shape=(10,))
    out = pgc.field(dtype=pgc.u16, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    store = ir.IRFieldStore(ir.IRName("out"), ir.IRName("i"),
                            ir.IRFieldLoad(ir.IRName("data"), ir.IRName("i")))
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), [store])
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    annotate_types(func)
    src = generate_msl_source(func)

    assert "device ushort* data" in src
    assert "device ushort* out" in src

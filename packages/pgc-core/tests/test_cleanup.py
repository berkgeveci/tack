"""Tests for type system cleanup: ScalarType on allocs, _NAME_TO_TYPE, registry leak."""

import numpy as np
import pytest
import pgc
from pgc.lang import ir
from pgc.lang.types import ScalarType, _NAME_TO_TYPE, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64

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


# --- _NAME_TO_TYPE completeness ---

class TestNameToType:
    def test_all_types_present(self):
        """Every ScalarType singleton has an entry in _NAME_TO_TYPE."""
        for t in [i8, u8, i16, u16, i32, u32, i64, u64, f32, f64]:
            assert t.name in _NAME_TO_TYPE
            assert _NAME_TO_TYPE[t.name] is t

    def test_roundtrip(self):
        """name → ScalarType → name roundtrips correctly."""
        for name, scalar_type in _NAME_TO_TYPE.items():
            assert scalar_type.name == name


# --- IRSharedAlloc/IRLocalAlloc dtype is ScalarType ---

class TestAllocDtypeIsScalarType:
    def test_shared_alloc_from_ast(self):
        """pgc.shared(pgc.f32, 256) produces IRSharedAlloc with ScalarType dtype."""
        import ast, textwrap
        from pgc.lang.ast_transform import transform_kernel
        source = textwrap.dedent("""
def kern(data, out):
    for i in range(10):
        smem = pgc.shared(pgc.f32, 256)
        smem[0] = data[i]
        out[i] = smem[0]
""")
        tree = ast.parse(source)
        module = transform_kernel(tree)
        pfor = module.functions[0].body[0]
        shared_alloc = pfor.body[0]
        assert isinstance(shared_alloc, ir.IRSharedAlloc)
        assert isinstance(shared_alloc.dtype, ScalarType)
        assert shared_alloc.dtype is f32

    def test_shared_alloc_i32(self):
        """pgc.shared(pgc.i32, 128) produces IRSharedAlloc with i32 dtype."""
        import ast, textwrap
        from pgc.lang.ast_transform import transform_kernel
        source = textwrap.dedent("""
def kern(data, out):
    for i in range(10):
        smem = pgc.shared(pgc.i32, 128)
        smem[0] = data[i]
        out[i] = smem[0]
""")
        tree = ast.parse(source)
        module = transform_kernel(tree)
        pfor = module.functions[0].body[0]
        shared_alloc = pfor.body[0]
        assert shared_alloc.dtype is i32

    def test_local_alloc_from_ast(self):
        """pgc.local_array(pgc.u8, 16) produces IRLocalAlloc with u8 dtype."""
        import ast, textwrap
        from pgc.lang.ast_transform import transform_kernel
        source = textwrap.dedent("""
def kern(data, out):
    for i in range(10):
        buf = pgc.local_array(pgc.u8, 16)
        buf[0] = data[i]
        out[i] = buf[0]
""")
        tree = ast.parse(source)
        module = transform_kernel(tree)
        pfor = module.functions[0].body[0]
        local_alloc = pfor.body[0]
        assert isinstance(local_alloc, ir.IRLocalAlloc)
        assert isinstance(local_alloc.dtype, ScalarType)
        assert local_alloc.dtype is u8

    def test_shared_like_resolves_to_scalar_type(self):
        """shared_like resolves to the field's ScalarType, not a C string."""
        pgc.init(arch=pgc.cpu)
        data = pgc.field(dtype=pgc.i32, shape=(10,))
        out = pgc.field(dtype=pgc.i32, shape=(10,))

        p_data = ir.IRParam("data")
        p_out = ir.IRParam("out")
        shared_alloc = ir.IRSharedAlloc("smem", dtype=None, size=ir.IRConstant(256),
                                         field_name="data")
        pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), [shared_alloc])
        func = ir.IRFunction("test", [p_data, p_out], [pfor])

        from pgc.lang.type_inference import infer_param_types
        from pgc.lang.ir_resolve import resolve_ir
        infer_param_types(func, (data, out))
        resolve_ir(func, {"data": data, "out": out})

        assert isinstance(shared_alloc.dtype, ScalarType)
        assert shared_alloc.dtype is i32


# --- Codegen emits correct C types from ScalarType ---

def test_cuda_shared_alloc_types():
    """CUDA codegen maps ScalarType to correct C types for shared/local alloc."""
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.cuda_gen import generate_cuda_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.f32, shape=(10,))
    out = pgc.field(dtype=pgc.f32, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared = ir.IRSharedAlloc("smem", dtype=f32, size=ir.IRConstant(256))
    local = ir.IRLocalAlloc("buf", dtype=i64, size=ir.IRConstant(8))
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), [shared, local])
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "__shared__ float smem[256]" in src
    assert "long long buf[8]" in src


def test_msl_shared_alloc_types():
    """MSL codegen maps ScalarType to correct MSL types."""
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.msl_gen import generate_msl_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.f32, shape=(10,))
    out = pgc.field(dtype=pgc.f32, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared = ir.IRSharedAlloc("smem", dtype=u8, size=ir.IRConstant(256))
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), [shared])
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    annotate_types(func)
    src = generate_msl_source(func)

    assert "threadgroup uchar smem[256]" in src


# --- End-to-end: shared and local alloc with ScalarType ---

def test_shared_f32_end_to_end(backend):
    """pgc.shared(pgc.f32, ...) works end-to-end."""
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            smem = pgc.shared(pgc.f32, 256)
            tid = pgc.thread_id()
            smem[tid] = data[i]
            pgc.barrier()
            out[i] = smem[tid]

    kern(data, out)
    np.testing.assert_array_equal(out.to_numpy(), np.arange(n, dtype=np.float32))


def test_local_array_u8_end_to_end(backend):
    """pgc.local_array(pgc.u8, ...) works end-to-end."""
    n = 8
    out = pgc.field(dtype=pgc.u8, shape=(n,))

    @pgc.kernel
    def kern(out):
        for i in range(out.shape[0]):
            buf = pgc.local_array(pgc.u8, 4)
            buf[0] = 42
            out[i] = buf[0]

    kern(out)
    np.testing.assert_array_equal(out.to_numpy(), np.full(n, 42, dtype=np.uint8))


# --- Template func registry cleanup ---

def test_func_registry_cleanup():
    """Template rewrite cleans up temporary _func_registry entries."""
    from pgc.lang.func import _func_registry

    pgc.init(arch=pgc.cpu)

    @pgc.data_oriented
    class MyClass:
        val = 10

        @pgc.func
        def get_val(self):
            return self.val

    @pgc.kernel
    def kern(obj: pgc.template(), out):
        for i in range(out.shape[0]):
            out[i] = obj.get_val()

    # Snapshot registry before
    keys_before = set(_func_registry.keys())

    out = pgc.field(dtype=pgc.i32, shape=(4,))
    obj = MyClass()
    kern(obj, out)

    # Registry should not have grown (temporary keys cleaned up)
    keys_after = set(_func_registry.keys())
    leaked = keys_after - keys_before
    assert len(leaked) == 0, f"Leaked registry keys: {leaked}"

    # Verify the kernel still works
    np.testing.assert_array_equal(out.to_numpy(), np.full(4, 10, dtype=np.int32))


def test_func_registry_no_leak_on_repeated_calls():
    """Repeated template calls don't accumulate registry entries."""
    from pgc.lang.func import _func_registry

    pgc.init(arch=pgc.cpu)

    @pgc.data_oriented
    class Counter:
        step = 1

        @pgc.func
        def apply(self, x):
            return x + self.step

    @pgc.kernel
    def kern(c: pgc.template(), data, out):
        for i in range(data.shape[0]):
            out[i] = c.apply(data[i])

    n = 4
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    size_before = len(_func_registry)

    # Call multiple times with different objects
    for _ in range(10):
        c = Counter()
        kern(c, data, out)

    size_after = len(_func_registry)
    assert size_after == size_before, (
        f"Registry grew by {size_after - size_before} entries over 10 calls"
    )

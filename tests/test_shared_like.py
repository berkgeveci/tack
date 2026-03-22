"""Tests for pgc.shared_like (Layer 4 of type system refactor).

Validates that pgc.shared_like(field, size) allocates shared memory
with the same dtype as the given field.
"""

import numpy as np
import pytest
import pgc


backends = []
try:
    pgc.init(arch=pgc.cpu)
    backends.append("cpu")
except Exception:
    pass
try:
    pgc.init(arch=pgc.metal)
    backends.append("metal")
except Exception:
    pass


@pytest.fixture(params=backends)
def backend(request):
    pgc.init(arch=getattr(pgc, request.param))
    return request.param


# --- AST transform level ---

def test_shared_like_produces_ir():
    """shared_like(field, size) should produce IRSharedAlloc with field_name."""
    import ast, textwrap
    from pgc.lang.ast_transform import transform_kernel
    source = textwrap.dedent("""
def kern(data, out):
    for i in range(10):
        smem = pgc.shared_like(data, 256)
        smem[0] = data[i]
        out[i] = smem[0]
""")
    tree = ast.parse(source)
    module = transform_kernel(tree)
    # Find the IRSharedAlloc in the body
    from pgc.lang import ir
    pfor = module.functions[0].body[0]
    shared_alloc = pfor.body[0]
    assert isinstance(shared_alloc, ir.IRSharedAlloc)
    assert shared_alloc.dtype is None  # not yet resolved
    assert shared_alloc.field_name == "data"


# --- End-to-end: shared_like with f32 field ---

def test_shared_like_f32(backend):
    """shared_like inherits f32 from the field."""
    n = 256
    data = pgc.field(dtype=pgc.f32, shape=(n,))
    out = pgc.field(dtype=pgc.f32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.float32))

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            smem = pgc.shared_like(data, 256)
            tid = pgc.thread_id()
            smem[tid] = data[i]
            pgc.barrier()
            out[i] = smem[tid]

    kern(data, out)
    result = out.to_numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32))


# --- End-to-end: shared_like with i32 field ---

def test_shared_like_i32(backend):
    """shared_like inherits i32 from the field."""
    n = 256
    data = pgc.field(dtype=pgc.i32, shape=(n,))
    out = pgc.field(dtype=pgc.i32, shape=(n,))
    data.from_numpy(np.arange(n, dtype=np.int32))

    @pgc.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            smem = pgc.shared_like(data, 256)
            tid = pgc.thread_id()
            smem[tid] = data[i]
            pgc.barrier()
            out[i] = smem[tid]

    kern(data, out)
    result = out.to_numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.int32))


# --- Codegen level: check generated source ---

def test_shared_like_cuda_codegen():
    """CUDA codegen emits correct shared type from shared_like."""
    from pgc.lang import ir
    from pgc.lang.types import f32, i32
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_resolve import resolve_ir
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.cuda_gen import generate_cuda_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.f32, shape=(10,))
    out = pgc.field(dtype=pgc.f32, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared_alloc = ir.IRSharedAlloc("smem", dtype=None, size=ir.IRConstant(256),
                                     field_name="data")
    body = [shared_alloc]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    resolve_ir(func, {"data": data, "out": out})
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "__shared__ float smem[256]" in src


def test_shared_like_i32_cuda_codegen():
    """CUDA codegen emits int shared type from i32 field."""
    from pgc.lang import ir
    from pgc.lang.types import i32
    from pgc.lang.type_inference import infer_param_types
    from pgc.lang.ir_resolve import resolve_ir
    from pgc.lang.ir_type_annotate import annotate_types
    from pgc.codegen.cuda_gen import generate_cuda_source

    pgc.init(arch=pgc.cpu)
    data = pgc.field(dtype=pgc.i32, shape=(10,))
    out = pgc.field(dtype=pgc.i32, shape=(10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared_alloc = ir.IRSharedAlloc("smem", dtype=None, size=ir.IRConstant(256),
                                     field_name="data")
    body = [shared_alloc]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    resolve_ir(func, {"data": data, "out": out})
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "__shared__ int smem[256]" in src

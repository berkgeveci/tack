"""Tests for tack.shared_like (Layer 4 of type system refactor).

Validates that tack.shared_like(array, size) allocates shared memory
with the same dtype as the given array.
"""

import numpy as np

import tack

# --- AST transform level ---

def test_shared_like_produces_ir():
    """shared_like(array, size) should produce IRSharedAlloc with array_name."""
    import ast
    import textwrap

    from tack.lang.ast_transform import transform_kernel
    source = textwrap.dedent("""
def kern(data, out):
    for i in range(10):
        smem = tack.shared_like(data, 256)
        smem[0] = data[i]
        out[i] = smem[0]
""")
    tree = ast.parse(source)
    module = transform_kernel(tree)
    # Find the IRSharedAlloc in the body
    from tack.lang import ir
    pfor = module.functions[0].body[0]
    shared_alloc = pfor.body[0]
    assert isinstance(shared_alloc, ir.IRSharedAlloc)
    assert shared_alloc.dtype is None  # not yet resolved
    assert shared_alloc.array_name == "data"


# --- End-to-end: shared_like with f32 array ---

def test_shared_like_f32(workgroup_backend):
    """shared_like inherits f32 from the array."""
    n = 256
    data = tack.empty((n,))
    out = tack.empty((n,))
    tack.copyto(data, np.arange(n, dtype=np.float32))

    @tack.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            smem = tack.shared_like(data, 256)
            tid = tack.thread_id()
            smem[tid] = data[i]
            tack.barrier()
            out[i] = smem[tid]

    kern(data, out)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.float32))


# --- End-to-end: shared_like with i32 array ---

def test_shared_like_i32(workgroup_backend):
    """shared_like inherits i32 from the array."""
    n = 256
    data = tack.empty((n,), tack.i32)
    out = tack.empty((n,), tack.i32)
    tack.copyto(data, np.arange(n, dtype=np.int32))

    @tack.kernel
    def kern(data, out):
        for i in range(data.shape[0]):
            smem = tack.shared_like(data, 256)
            tid = tack.thread_id()
            smem[tid] = data[i]
            tack.barrier()
            out[i] = smem[tid]

    kern(data, out)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(n, dtype=np.int32))


# --- Codegen level: check generated source ---

def test_shared_like_cuda_codegen():
    """CUDA codegen emits correct shared type from shared_like."""
    from tack.codegen.cuda_gen import generate_cuda_source
    from tack.lang import ir
    from tack.lang.ir_resolve import resolve_ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types

    tack.init(arch=tack.cpu)
    data = tack.empty((10,))
    out = tack.empty((10,))

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared_alloc = ir.IRSharedAlloc("smem", dtype=None, size=ir.IRConstant(256),
                                     array_name="data")
    body = [shared_alloc]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    resolve_ir(func, {"data": data, "out": out})
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "__shared__ float tack_var_a_smem[256]" in src


def test_shared_like_i32_cuda_codegen():
    """CUDA codegen emits int shared type from i32 array."""
    from tack.codegen.cuda_gen import generate_cuda_source
    from tack.lang import ir
    from tack.lang.ir_resolve import resolve_ir
    from tack.lang.ir_type_annotate import annotate_types
    from tack.lang.type_inference import infer_param_types

    tack.init(arch=tack.cpu)
    data = tack.empty((10,), tack.i32)
    out = tack.empty((10,), tack.i32)

    p_data = ir.IRParam("data")
    p_out = ir.IRParam("out")
    shared_alloc = ir.IRSharedAlloc("smem", dtype=None, size=ir.IRConstant(256),
                                     array_name="data")
    body = [shared_alloc]
    pfor = ir.IRParallelFor("i", ir.IRConstant(0), ir.IRConstant(10), body)
    func = ir.IRFunction("test", [p_data, p_out], [pfor])
    infer_param_types(func, (data, out))
    resolve_ir(func, {"data": data, "out": out})
    annotate_types(func)
    src = generate_cuda_source(func)

    assert "__shared__ int tack_var_a_smem[256]" in src

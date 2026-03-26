"""Inspect generated code for PGC kernels without executing them.

Usage:
    print(pgc.inspect(my_kernel, arg1, arg2))             # backend source
    print(pgc.inspect(my_kernel, arg1, arg2, mode="ir"))   # PGC IR
"""

import copy

from pgc.lang import ir
from pgc.lang.field import Field
from pgc.lang.type_inference import infer_param_types, check_dispatch_types
from pgc.lang.ir_resolve import resolve_ir
from pgc.lang.ir_optimize import optimize_ir
from pgc.lang.ir_type_annotate import annotate_types
from pgc.runtime.kernel_utils import (
    _detect_template_args,
    _expand_template_args,
    _detect_vector_fields_from_args,
    _detect_texture_fields,
)


def _prepare_ir(kernel, args):
    """Run the common IR preparation pipeline: transform, resolve, infer, optimize.

    Returns (ir_func, effective_args) with a deep-copied, fully annotated IR.
    """
    from pgc.lang.field import Texture3D

    template_args = _detect_template_args(kernel, args)
    effective_args = _expand_template_args(args, template_args)
    vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
    texture_fields = _detect_texture_fields(kernel, args, template_args)

    ir_module = kernel.get_ir(
        vector_fields,
        template_args=template_args if template_args else None,
        texture_fields=texture_fields,
    )
    ir_func = copy.deepcopy(ir_module.functions[0])

    # Resolve dimension sizes
    name_to_field = {}
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, (Field, Texture3D)):
            name_to_field[param.name] = arg
    resolve_ir(ir_func, name_to_field)

    # Type inference
    infer_param_types(ir_func, effective_args)

    # Store texture shapes on params for codegen
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, Texture3D):
            param._texture_shape = arg.shape_3d

    # Optimize
    optimize_ir(ir_func)

    # Type annotation
    annotate_types(ir_func)

    return ir_func, effective_args


def inspect(kernel, *args, mode="source"):
    """Return generated code for a kernel without executing it.

    Args:
        kernel: A @pgc.kernel decorated function.
        *args: The arguments the kernel would be called with.
        mode: What to return:
            "ir"        — PGC intermediate representation
            "source"    — backend-specific source code (MSL, CUDA C, LLVM IR, etc.)
            "optimized" — source after backend optimization (LLVM O3 on CPU)

    Returns:
        The generated code as a string.
    """
    from pgc.lang.kernel import Kernel
    if not isinstance(kernel, Kernel):
        raise TypeError(f"Expected a @pgc.kernel, got {type(kernel).__name__}")

    if mode == "ir":
        ir_func, _ = _prepare_ir(kernel, args)
        return ir.dump(ir_func)

    if mode == "source":
        return _generate_source(kernel, args)

    if mode == "optimized":
        return _generate_source(kernel, args, optimize=True)

    raise ValueError(
        f"Unknown inspect mode: '{mode}'. Use 'ir', 'source', or 'optimized'."
    )


def _generate_source(kernel, args, optimize=False):
    """Generate backend-specific source code."""
    from pgc.runtime.dispatch import get_backend
    backend = get_backend()
    backend_name = type(backend).__name__

    ir_func, effective_args = _prepare_ir(kernel, args)

    # GPU backends need scalar packing
    if backend_name in ("MetalBackend", "CUDABackend", "HIPBackend", "LevelZeroBackend"):
        from pgc.lang.ir_pack_scalars import pack_scalars
        pack_scalars(ir_func, effective_args)
        # Re-annotate after packing
        annotate_types(ir_func)

    if backend_name == "CPUBackend":
        from pgc.codegen.llvm_gen import generate_llvm_ir
        llvm_module = generate_llvm_ir(ir_func)
        llvm_ir_str = str(llvm_module)
        if optimize:
            from llvmlite import binding as llvm
            from pgc.runtime.cpu import _create_target_machine, _optimize_module
            mod = llvm.parse_assembly(llvm_ir_str)
            mod.verify()
            tm = _create_target_machine()
            _optimize_module(mod, tm)
            return str(mod)
        return llvm_ir_str

    if backend_name == "MetalBackend":
        from pgc.codegen.msl_gen import generate_msl_source, _safe_kernel_name
        ir_func.name = _safe_kernel_name(ir_func.name)
        return generate_msl_source(ir_func)

    if backend_name == "CUDABackend":
        from pgc.codegen.cuda_gen import generate_cuda_source
        return generate_cuda_source(ir_func)

    if backend_name == "HIPBackend":
        from pgc.codegen.hip_gen import generate_hip_source
        return generate_hip_source(ir_func)

    if backend_name == "LevelZeroBackend":
        from pgc.codegen.opencl_gen import generate_opencl_source
        return generate_opencl_source(ir_func)

    raise RuntimeError(f"inspect() not supported for backend: {backend_name}")

"""Inspect generated code for Tack kernels without executing them.

Usage:
    print(tack.inspect(my_kernel, arg1, arg2))             # backend source
    print(tack.inspect(my_kernel, arg1, arg2, mode="ir"))   # Tack IR
"""

from tack.lang import ir
from tack.lang.field import Field
from tack.lang.ir_optimize import optimize_ir
from tack.lang.ir_resolve import resolve_ir
from tack.lang.ir_traversal import clone_ir
from tack.lang.ir_type_annotate import annotate_types
from tack.lang.ir_verify import verify_ir
from tack.lang.type_inference import infer_param_types
from tack.lang.workgroup_participation import (
    check_workgroup_launch,
    check_workgroup_participation,
)
from tack.lang.workgroup_support import check_workgroup_support
from tack.runtime.kernel_utils import (
    _detect_template_args,
    _detect_texture_fields,
    _detect_vector_fields_from_args,
    _expand_template_args,
    _get_loop_range,
    _localize_assigned_scalar_params,
)


def _prepare_ir(kernel, args, *, backend=None):
    """Run the common IR preparation pipeline: transform, resolve, infer, optimize.

    Returns (ir_func, effective_args) with a deep-copied, fully annotated IR.
    Supply a backend to enforce target capabilities. Cross-target codegen
    tools may omit it and let their selected generator check support.
    """
    from tack.lang.field import Texture3D

    template_args = _detect_template_args(kernel, args)
    effective_args = _expand_template_args(args, template_args)
    vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
    texture_fields = _detect_texture_fields(kernel, args, template_args)

    ir_module = kernel.get_ir(
        vector_fields,
        template_args=template_args if template_args else None,
        texture_fields=texture_fields,
    )
    template = ir_module.functions[0]
    if backend is not None:
        check_workgroup_support(
            template, supports_workgroups=backend.supports_workgroups,
            backend_label=backend.label, cache_features=True,
        )
    ir_func = clone_ir(template)

    # Resolve dimension sizes
    name_to_field = {}
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, (Field, Texture3D)):
            name_to_field[param.name] = arg
    resolve_ir(ir_func, name_to_field)
    verify_ir(ir_func, 'resolved')

    # Type inference
    infer_param_types(ir_func, effective_args)
    verify_ir(ir_func, 'inferred')

    # Store texture shapes on params for codegen
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, Texture3D):
            param._texture_shape = arg.shape_3d

    _localize_assigned_scalar_params(ir_func)
    verify_ir(ir_func, 'localized')
    if backend is not None and backend.supports_workgroups:
        if check_workgroup_participation(ir_func):
            check_workgroup_launch(ir_func.name, _get_loop_range(ir_func, effective_args),
                                   backend_label=backend.label)

    # Optimize
    optimize_ir(ir_func)
    verify_ir(ir_func, 'optimized')

    # Type annotation
    annotate_types(ir_func)
    verify_ir(ir_func, 'typed')

    return ir_func, effective_args


def inspect(kernel, *args, mode="source"):
    """Return generated code for a kernel without executing it.

    Args:
        kernel: A @tack.kernel decorated function.
        *args: The arguments the kernel would be called with.
        mode: What to return:
            "ir"        — Tack intermediate representation
            "source"    — backend-specific source code (MSL, CUDA C, LLVM IR, etc.)
            "optimized" — source after backend optimization (LLVM O3 on CPU)

    Returns:
        The generated code as a string.
    """
    from tack.lang.kernel import Kernel
    if not isinstance(kernel, Kernel):
        raise TypeError(f"Expected a @tack.kernel, got {type(kernel).__name__}")

    if mode == "ir":
        from tack.runtime.dispatch import get_backend
        ir_func, _ = _prepare_ir(kernel, args, backend=get_backend())
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
    from tack.runtime.dispatch import get_backend
    backend = get_backend()
    backend_name = type(backend).__name__

    ir_func, effective_args = _prepare_ir(kernel, args, backend=backend)

    # GPU backends need scalar packing
    if backend_name in ("MetalBackend", "CUDABackend", "HIPBackend", "LevelZeroBackend"):
        from tack.lang.ir_pack_scalars import pack_scalars
        pack_scalars(ir_func, effective_args)
        verify_ir(ir_func, 'packed')
        # Re-annotate after packing
        annotate_types(ir_func)
        verify_ir(ir_func, 'typed')

    if backend_name == "CPUBackend":
        from tack.codegen.llvm_gen import generate_llvm_ir
        from tack.runtime import cpu
        from tack.runtime.kernel_utils import fields_disjoint

        # Show the variant these arguments would run.
        ir_func.disjoint_fields = (
            cpu._SPECIALIZE_DISJOINT and fields_disjoint(ir_func, effective_args))
        llvm_module = generate_llvm_ir(ir_func)
        llvm_ir_str = str(llvm_module)
        if optimize:
            from llvmlite import binding as llvm

            from tack.runtime.cpu import _create_target_machine, _optimize_module
            mod = llvm.parse_assembly(llvm_ir_str)
            mod.verify()
            tm = _create_target_machine()
            _optimize_module(mod, tm)
            return str(mod)
        return llvm_ir_str

    if backend_name == "MetalBackend":
        from tack.codegen.msl_gen import generate_msl_source
        return generate_msl_source(ir_func)

    if backend_name == "CUDABackend":
        from tack.codegen.cuda_gen import generate_cuda_source
        return generate_cuda_source(ir_func)

    if backend_name == "HIPBackend":
        from tack.codegen.hip_gen import generate_hip_source
        return generate_hip_source(ir_func)

    if backend_name == "LevelZeroBackend":
        from tack.codegen.opencl_gen import generate_opencl_source
        return generate_opencl_source(ir_func)

    raise RuntimeError(f"inspect() not supported for backend: {backend_name}")

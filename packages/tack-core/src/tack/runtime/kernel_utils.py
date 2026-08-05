"""Shared kernel utility functions used by all backends.

These helpers handle template detection, vector field detection, texture detection,
loop range resolution, and scalar packing — common pre-dispatch logic shared
across CPU, GPU, and WebGPU backends.
"""

import weakref

from tack.lang import ir
from tack.lang.field import Field


def new_kernel_cache():
    """Create a backend compiled-kernel cache.

    Maps ``Kernel`` → {variant_key: compiled}, holding the kernel weakly.
    """
    return weakref.WeakKeyDictionary()


def kernel_cache_slot(cache, kernel) -> dict:
    """Return (creating if needed) the per-kernel variant dict in ``cache``.

    Keyed on the ``Kernel`` object itself rather than ``id(kernel)``.  This is
    a correctness requirement, not a style choice: ``id()`` is a memory
    address, and a garbage-collected kernel frees its address for reuse.  A
    later kernel allocated at the same address with the same name and type
    signature would hit the previous kernel's compiled code and silently
    return wrong results.  Holding the key weakly also lets compiled code and
    its device modules be released once the kernel itself goes away.
    """
    slot = cache.get(kernel)
    if slot is None:
        slot = cache[kernel] = {}
    return slot


def kernel_variant_key(ir_func, kernel, vector_fields, template_args) -> tuple:
    """Build the cache key distinguishing compiled variants of one kernel.

    The kernel identity is carried by the enclosing per-kernel slot, so this
    only needs to separate specializations: argument types, texture shapes,
    and template constants.
    """
    type_sig = tuple(p.type_annotation for p in ir_func.params)
    tex_sig = tuple(getattr(p, '_texture_shape', None) for p in ir_func.params)
    tmpl_key = ""
    if template_args:
        tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
    return (type_sig, tex_sig, tmpl_key)


def _detect_template_args(kernel, args) -> dict[int, tuple[str, object]]:
    """Detect which arguments are @tack.data_oriented template objects.

    Returns dict: param_index -> (param_name, template_object)
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    templates = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if hasattr(arg, '_data_oriented') and arg._data_oriented:
            templates[i] = (param_name, arg)
    return templates


def _expand_template_args(args, template_args) -> tuple:
    """Replace template args with their field and runtime scalar attributes.

    Returns new args tuple with template objects removed and their
    field attributes and runtime scalars appended.  These are appended
    in reverse template index order to match the AST rewrite pass
    (which processes templates from highest index to lowest).
    """
    if not template_args:
        return args

    from tack.lang.template_rewrite import classify_template_attrs

    new_args = []
    extra = []
    # Collect template fields and runtime scalars in reverse index order
    for idx in sorted(template_args.keys(), reverse=True):
        _, obj = template_args[idx]
        _, fields, runtime_scalars = classify_template_attrs(obj)
        for attr_name in sorted(fields.keys()):
            extra.append(fields[attr_name])
        for attr_name in sorted(runtime_scalars.keys()):
            extra.append(runtime_scalars[attr_name])
    # Build non-template args in order
    for i, arg in enumerate(args):
        if i not in template_args:
            new_args.append(arg)
    new_args.extend(extra)
    return tuple(new_args)


def _detect_vector_fields(kernel, args) -> dict[str, int] | None:
    """Detect which kernel parameters are vector fields.

    Returns a dict mapping parameter names to component counts,
    or None if no vector fields are present.
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_fields = {}
    for param_name, arg in zip(params, args):
        if isinstance(arg, Field) and hasattr(arg, '_vector_n'):
            vector_fields[param_name] = arg._vector_n
    return vector_fields if vector_fields else None


def _detect_vector_fields_from_args(kernel, args, template_args) -> dict[str, int] | None:
    """Detect vector fields, accounting for template parameters.

    When template args are present, we need to skip them when matching
    parameter names to arguments.
    """
    if not template_args:
        return _detect_vector_fields(kernel, args)

    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_fields = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if i in template_args:
            continue
        if isinstance(arg, Field) and hasattr(arg, '_vector_n'):
            vector_fields[param_name] = arg._vector_n
    return vector_fields if vector_fields else None


def _detect_texture_fields(kernel, args, template_args=None) -> dict[str, tuple] | None:
    """Detect which kernel parameters are Texture3D objects."""
    from tack.lang.field import Texture3D
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    texture_fields = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if template_args and i in template_args:
            continue
        if isinstance(arg, Texture3D):
            texture_fields[param_name] = arg.shape_3d
    return texture_fields if texture_fields else None


def _resolve_range_expr(node: ir.IRNode, name_to_arg: dict) -> int:
    """Resolve a range expression to a concrete integer value."""
    if isinstance(node, ir.IRConstant):
        return int(node.value)

    # x.shape[0]  →  IRFieldLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))
    if isinstance(node, ir.IRFieldLoad):
        obj = node.field
        if isinstance(obj, ir.IRAttribute) and obj.attr == "shape":
            if isinstance(obj.obj, ir.IRName):
                arg = name_to_arg.get(obj.obj.name)
                if isinstance(arg, Field):
                    idx = _resolve_range_expr(node.index, name_to_arg)
                    return arg.shape[idx]

    # len(x)  →  IRAttribute(IRName("x"), "__len__")
    if isinstance(node, ir.IRAttribute) and node.attr == "__len__":
        if isinstance(node.obj, ir.IRName):
            arg = name_to_arg.get(node.obj.name)
            if isinstance(arg, Field):
                return arg.shape[0]

    # Binary ops on range expressions (e.g., n - 1)
    if isinstance(node, ir.IRBinOp):
        left = _resolve_range_expr(node.left, name_to_arg)
        right = _resolve_range_expr(node.right, name_to_arg)
        ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
               "*": lambda a, b: a * b, "//": lambda a, b: a // b}
        if node.op in ops:
            return ops[node.op](left, right)

    # Plain name reference (e.g., `n` passed as scalar)
    if isinstance(node, ir.IRName):
        arg = name_to_arg.get(node.name)
        if arg is not None:
            return int(arg)

    raise RuntimeError(f"Cannot resolve loop range expression: {type(node).__name__}")


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments.

    Resolves the loop end expression — supports:
      - IRConstant(N)
      - IRFieldLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))  →  x.shape[0]
      - IRAttribute(IRName("x"), "__len__")  →  len(x)
    """
    # Find the top-level parallel for
    parallel_for = None
    for stmt in ir_func.body:
        if isinstance(stmt, ir.IRParallelFor):
            parallel_for = stmt
            break

    if parallel_for is None:
        raise RuntimeError("Kernel has no parallel for-loop")

    # Build name → arg mapping
    name_to_arg = {}
    for param, arg in zip(ir_func.params, args):
        name_to_arg[param.name] = arg

    return _resolve_range_expr(parallel_for.end, name_to_arg)


def _create_pack_fields(pack_info, args, backend):
    """Create packed Field objects from scalar pack info.

    Args:
        pack_info: list of (pack_name, dtype, [(orig_name, orig_arg_idx, idx_in_pack)])
        args: the original effective_args tuple
        backend: the active backend (for allocate_field)

    Returns:
        list of Field objects, one per pack group.
    """
    import numpy as np
    from tack.lang.field import Field

    fields = []
    for pack_name, dtype, entries in pack_info:
        np_dtype = dtype.numpy_dtype
        arr = np.zeros(len(entries), dtype=np_dtype)
        for _, orig_arg_idx, idx_in_pack in entries:
            arr[idx_in_pack] = args[orig_arg_idx]
        buf = backend.allocate_field(dtype, (len(entries),))
        f = Field(dtype, (len(entries),), buf)
        f.from_numpy(arr)
        fields.append(f)
    return fields


def _update_pack_fields(pack_fields, pack_info, args):
    """Update existing packed Field objects with new scalar values.

    Reuses the allocated device buffers — only copies new values.
    """
    import numpy as np

    for field, (pack_name, dtype, entries) in zip(pack_fields, pack_info):
        np_dtype = dtype.numpy_dtype
        arr = np.zeros(len(entries), dtype=np_dtype)
        for _, orig_arg_idx, idx_in_pack in entries:
            arr[idx_in_pack] = args[orig_arg_idx]
        field.from_numpy(arr)

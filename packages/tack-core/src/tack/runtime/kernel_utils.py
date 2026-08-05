"""Shared kernel utility functions used by all backends.

These helpers handle template detection, vector field detection, texture detection,
loop range resolution, and scalar packing — common pre-dispatch logic shared
across CPU, GPU, and WebGPU backends.

Variant resolution
------------------
``resolve_variant`` is the one place that turns a call into a compiled
kernel.  It runs the IR pass pipeline only when the variant is new; a
repeat dispatch does argument detection, type inference, and a dict
lookup.  The passes are not cheap — resolve + optimize + annotate is
~16 µs against a ~28 µs CPU dispatch — and they are pure functions of
the argument types and shapes, so re-deriving them per call was waste.

The passes also *bake dispatch-time constants into the IR*: a
multi-dimensional index linearizes to ``i * dim1 + j`` with ``dim1``
substituted as a literal.  That makes those substitutions part of the
compiled code's identity, so the cache key has to include them —
see ``shape_signature`` — and it makes the pristine IR from
``kernel.get_ir()`` a template that must never be mutated in place.
"""

import copy
import weakref

from tack.lang import ir
from tack.lang.field import Field
from tack.lang.type_inference import infer_param_types, check_dispatch_types


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


def kernel_variant_key(ir_func, kernel, vector_fields, template_args,
                       shape_sig=()) -> tuple:
    """Build the cache key distinguishing compiled variants of one kernel.

    The kernel identity is carried by the enclosing per-kernel slot, so this
    only needs to separate specializations: argument types, texture shapes,
    template constants, and every dimension size the resolve pass bakes
    into the generated code (``shape_sig``, from ``shape_signature``).

    Leaving the dimension sizes out is a correctness bug, not a missed
    optimization: a kernel that indexes ``a[i, j]`` compiles the row stride
    in as a literal, so reusing that code for a differently shaped field
    reads the wrong addresses and silently returns wrong numbers.
    """
    type_sig = tuple(p.type_annotation for p in ir_func.params)
    tex_sig = tuple(getattr(p, '_texture_shape', None) for p in ir_func.params)
    tmpl_key = ""
    if template_args:
        tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
    return (type_sig, tex_sig, tmpl_key, shape_sig)


def _walk_ir(node):
    """Yield every IR node under `node`, including itself."""
    if isinstance(node, ir.IRNode):
        yield node
        for value in vars(node).values():
            yield from _walk_ir(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_ir(value)


def _static_field_aliases(ir_func) -> dict:
    """Map each alias name to the field name it ultimately refers to.

    Inlining a ``@tack.func`` emits assignments like
    ``__func_out_x0_0__ = out_x0``, so a dimension query can name an alias
    rather than a parameter. The chain is a property of the code, not of
    any dispatch, so it is resolved once.
    """
    params = {p.name for p in ir_func.params}
    aliases = {}
    for node in _walk_ir(ir_func.body):
        if isinstance(node, ir.IRAssign) and isinstance(node.value, ir.IRName):
            src = node.value.name
            if src in params:
                aliases[node.target] = src
            elif src in aliases:
                aliases[node.target] = aliases[src]
    return aliases


def _dependent_nodes(ir_func):
    """Every node whose value gets compiled into the generated code.

    Skips the outermost parallel loop's bound: it is passed to the launch
    rather than emitted, so `for i in range(x.shape[0])` must not make the
    compiled kernel depend on the array's length.
    """
    for stmt in ir_func.body:
        if isinstance(stmt, ir.IRParallelFor):
            yield from _walk_ir(stmt.start)
            yield from _walk_ir(stmt.body)
        else:
            yield from _walk_ir(stmt)


def ir_shape_deps(ir_func) -> tuple:
    """Which field dimensions the resolve pass will bake into this IR.

    Returns a tuple of ``(field_name, dim)``, memoized on the IR function —
    it depends on the kernel's source, not on the arguments. Most kernels
    index one-dimensionally and get back an empty tuple, so the per-dispatch
    cost of keying on shapes is nothing at all.
    """
    deps = ir_func.__dict__.get('_shape_deps')
    if deps is not None:
        return deps

    aliases = _static_field_aliases(ir_func)
    found = set()
    for node in _dependent_nodes(ir_func):
        if isinstance(node, ir.IRDimSize):
            found.add((aliases.get(node.field_name, node.field_name), node.dim))
        elif isinstance(node, ir.IRTextureSample):
            # The sampled extent is embedded in the generated code too.
            name = aliases.get(node.field_name, node.field_name)
            found.update((name, d) for d in range(3))
    deps = tuple(sorted(found))
    ir_func._shape_deps = deps
    return deps


def shape_signature(ir_func, name_to_field) -> tuple:
    """The concrete dimension sizes this call would bake into the code."""
    deps = ir_shape_deps(ir_func)
    if not deps:
        return ()
    sig = []
    for name, dim in deps:
        field = name_to_field.get(name)
        if field is None:
            sig.append(None)
            continue
        shape = getattr(field, 'shape_3d', None) \
            or getattr(field, '_logical_shape', None) or field.shape
        sig.append(shape[dim] if dim < len(shape) else None)
    return tuple(sig)


def dispatch_name_to_field(ir_func, effective_args) -> dict:
    """Map parameter names to the Field/Texture3D arguments bound to them."""
    from tack.lang.field import Texture3D
    mapping = {}
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, (Field, Texture3D)):
            mapping[param.name] = arg
    return mapping


def _store_texture_shapes(ir_func, effective_args):
    """Record Texture3D extents on the params, for codegen and the key."""
    from tack.lang.field import Texture3D
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, Texture3D):
            param._texture_shape = arg.shape_3d


class KernelVariant:
    """One compiled specialization of a kernel, plus the IR behind it.

    `ir` is the post-pass IR — kept so the loop range can be resolved from
    it on every dispatch without re-running the passes. `payload` is
    whatever the backend needed to cache alongside it.
    """

    __slots__ = ("ir", "payload")

    def __init__(self, ir_func, payload):
        self.ir = ir_func
        self.payload = payload


def resolve_variant(backend, kernel, args, kwargs, build,
                    store_texture_shapes=None) -> tuple:
    """Find or build the compiled variant for this call.

    On a cache hit this touches no IR beyond parameter type inference. On a
    miss it deep-copies the pristine template and runs resolve → infer →
    check → optimize on the copy, then hands it to `build`, which does the
    backend-specific tail (annotate, any packing, compile) and returns the
    payload to cache.

    `store_texture_shapes` overrides how Texture3D extents are recorded on
    the params — Level Zero falls back to software sampling on devices
    without hardware samplers, and that choice changes the generated code,
    so it has to happen before the key is built.

    Returns ``(variant, effective_args)``.
    """
    if store_texture_shapes is None:
        store_texture_shapes = _store_texture_shapes
    if kwargs:
        raise NotImplementedError("Keyword arguments not supported in kernels")

    template_args = _detect_template_args(kernel, args)
    effective_args = _expand_template_args(args, template_args)
    vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
    texture_fields = _detect_texture_fields(kernel, args, template_args)

    # The pristine IR for this specialization. Never mutated below the
    # parameter list — the passes run on a copy.
    template = kernel.get_ir(
        vector_fields,
        template_args=template_args if template_args else None,
        texture_fields=texture_fields,
    ).functions[0]

    name_to_field = dispatch_name_to_field(template, effective_args)

    # Parameter types and texture extents come from the actual arguments and
    # are part of the key, so they have to be derived before the lookup.
    # Both only write to the param objects, leaving the body pristine.
    infer_param_types(template, effective_args)
    store_texture_shapes(template, effective_args)

    key = kernel_variant_key(template, kernel, vector_fields, template_args,
                             shape_signature(template, name_to_field))

    slot = kernel_cache_slot(backend._cache, kernel)
    variant = slot.get(key)
    if variant is None:
        from tack.lang.ir_resolve import resolve_ir
        from tack.lang.ir_optimize import optimize_ir

        ir_func = copy.deepcopy(template)
        resolve_ir(ir_func, name_to_field)
        infer_param_types(ir_func, effective_args)
        check_dispatch_types(ir_func, effective_args,
                             supported_dtypes=backend.supported_dtypes,
                             backend_name=backend.label)
        store_texture_shapes(ir_func, effective_args)
        optimize_ir(ir_func)
        variant = KernelVariant(ir_func, build(ir_func, effective_args))
        slot[key] = variant

    return variant, effective_args


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

    # x.shape[k] / len(x) in the grid bound. The resolve pass deliberately
    # leaves these alone so the compiled kernel does not depend on the
    # array's length; they are evaluated here instead, per dispatch.
    if isinstance(node, ir.IRDimSize):
        arg = name_to_arg.get(node.field_name)
        if arg is not None:
            shape = getattr(arg, 'shape_3d', None) \
                or getattr(arg, '_logical_shape', None) or arg.shape
            return shape[node.dim]

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

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

import weakref

from tack.lang import ir
from tack.lang.atomic_support import check_atomic_alignment, check_atomic_support
from tack.lang.field import Field, Texture3D
from tack.lang.ir_traversal import clone_ir
from tack.lang.ir_traversal import walk_ir as _walk_ir
from tack.lang.type_inference import check_dispatch_types, infer_param_types
from tack.lang.workgroup_participation import (
    check_workgroup_launch,
    check_workgroup_participation,
)
from tack.lang.workgroup_support import check_workgroup_support


def as_address(ptr) -> int | None:
    """`ptr` as a machine address, or None if it cannot be one.

    Both GPU backends' `memory_space()` needs this, and both arrived at it
    by catching whatever the attempt happened to raise. That is the shape
    D9 is about: a handler wide enough to absorb "not an address" is wide
    enough to absorb "I called the binding wrong", and it answers both with
    the same confident `"cpu"`.

    The two cases separate cleanly if the question is asked directly.
    `int()` rejects objects and strings; the range check rejects integers
    that are the right type and still cannot be addresses -- negatives, and
    anything past 64 bits, which the bindings would otherwise refuse from
    inside their own marshalling. Doing it here means neither backend has
    to catch `OverflowError` around the API call, so the call can sit
    outside the `try` where a mistaken one is a traceback.

    Deliberately not a validity check: any 64-bit value can be an address,
    and deciding whether this one *is* is the driver's job.
    """
    try:
        addr = int(ptr)
    except (TypeError, ValueError):
        return None
    return addr if 0 <= addr < (1 << 64) else None


# Every backend's compiled cache, held weakly, so variants can be dropped
# across all of them when what they were specialized on goes away.
_kernel_caches = []


def new_kernel_cache():
    """Create a backend compiled-kernel cache.

    Maps ``Kernel`` → {variant_key: compiled}, holding the kernel weakly.
    """
    cache = weakref.WeakKeyDictionary()
    _kernel_caches[:] = [ref for ref in _kernel_caches if ref() is not None]
    _kernel_caches.append(weakref.ref(cache))
    return cache


def drop_variants(kernel, is_stale):
    """Remove `kernel`'s compiled variants whose key satisfies `is_stale`.

    Applies to every live backend cache, not only the active one: a variant
    compiled before a `tack.init()` switch is retained by the old backend.
    """
    for ref in list(_kernel_caches):
        cache = ref()
        slot = cache.get(kernel) if cache is not None else None
        if not slot:
            continue
        for key in [k for k in list(slot) if is_stale(k)]:
            slot.pop(key, None)


def kernel_cache_slot(cache, kernel) -> dict:
    """Return (creating if needed) the per-kernel variant dict in ``cache``.

    Keyed on the ``Kernel`` object itself rather than ``id(kernel)``.  This is
    a correctness requirement, not a style choice: ``id()`` is a memory
    address, and a garbage-collected kernel frees its address for reuse.  A
    later kernel allocated at the same address with the same name and type
    signature would hit the previous kernel's compiled code and silently
    return wrong results.  Holding the key weakly also lets compiled code and
    its device modules be released once the kernel itself goes away.

    `setdefault` rather than an assignment because two threads can both see
    no slot: with `cache[kernel] = {}` the second one installs a fresh dict
    over the first's, discarding a variant that had just been compiled. The
    results stay correct either way — each thread returns the variant it
    built — so this only saves the recompile, but it saves it for free.
    """
    slot = cache.get(kernel)
    if slot is None:
        slot = cache.setdefault(kernel, {})
    return slot


def kernel_variant_key(ir_func, kernel, vector_fields, template_args,
                       shape_sig=(), disjoint=False) -> tuple:
    """Build the cache key distinguishing compiled variants of one kernel.

    The kernel identity is carried by the enclosing per-kernel slot, so this
    only needs to separate specializations: argument types and categories,
    vector widths, texture shapes, template structure/constants, and every
    dimension size the resolve pass bakes into generated code (``shape_sig``,
    from ``shape_signature``).  ``disjoint`` separates code compiled with a
    no-overlap promise on its field pointers from code compiled without one;
    only backends that ask for that specialization ever pass it.

    Leaving the dimension sizes out is a correctness bug, not a missed
    optimization: a kernel that indexes ``a[i, j]`` compiles the row stride
    in as a literal, so reusing that code for a differently shaped field
    reads the wrong addresses and silently returns wrong numbers.
    """
    # One pass over the params: this runs on every dispatch, and a separate
    # walk per property cost a measurable few microseconds.
    param_sig = tuple(
        (p.type_annotation, getattr(p, '_is_field', True),
         getattr(p, '_is_texture', False), getattr(p, '_texture_shape', None))
        for p in ir_func.params)
    vec_sig = tuple(sorted(vector_fields.items())) if vector_fields else ()
    tmpl_key = ()
    if template_args:
        # Keep the structural key: stringifying it loses class identity.
        tmpl_key = kernel._make_cache_key(vector_fields, template_args)
    return (param_sig, vec_sig, tmpl_key, shape_sig, disjoint)


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


def written_field_params(ir_func):
    """Names of the field parameters this kernel may store to.

    Returns a frozenset, or ``None`` when a store goes through something
    this cannot trace back to a parameter or a local allocation, in which
    case every field has to be assumed written.  Memoized on the IR
    function: it is a property of the source, not of a dispatch.
    """
    cached = ir_func.__dict__.get('_written_params', False)
    if cached is not False:
        return cached

    params = {p.name for p in ir_func.params}
    local = set()
    copies = []
    stores = []
    for node in _walk_ir(ir_func.body):
        if isinstance(node, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            local.add(node.name)
        elif isinstance(node, ir.IRAssign) and isinstance(node.value, ir.IRName):
            copies.append((node.target, node.value.name))
        elif isinstance(node, (ir.IRFieldStore, ir.IRAtomicOp)):
            stores.append(node.field)

    # A name can be bound to different fields on different paths, so each
    # one maps to every parameter it might stand for.
    sources = {name: {name} for name in params}
    changed = True
    while changed:
        changed = False
        for target, src in copies:
            reach = sources.get(src)
            if reach and not reach <= sources.setdefault(target, set()):
                sources[target] |= reach
                changed = True

    written = set()
    for field in stores:
        name = getattr(field, 'name', None) if isinstance(field, ir.IRName) \
            else None
        if name in sources:
            written |= sources[name]
        elif name is None or name not in local:
            written = None
            break
    result = None if written is None else frozenset(written)
    ir_func._written_params = result
    return result


def _written_flags(ir_func) -> tuple:
    """`written_field_params` as one bool per parameter, memoized."""
    flags = ir_func.__dict__.get('_written_flags')
    if flags is None:
        written = written_field_params(ir_func)
        flags = tuple(written is None or p.name in written
                      for p in ir_func.params)
        ir_func._written_flags = flags
    return flags


def fields_disjoint(ir_func, effective_args) -> bool:
    """Whether this call's field storage satisfies a no-overlap promise.

    True when no field the kernel stores to shares a byte with any other
    field argument.  Fields that are only read may overlap each other
    freely -- ``dot(x, x)`` is still disjoint in the sense that matters,
    because nothing read through one pointer can change under the other.

    Needs host addresses, so it is only meaningful for buffers that have a
    ``span``; a field without one makes the answer False.

    This runs on every dispatch, hence the plain loops.
    """
    spans = []
    written = []
    for is_written, arg in zip(_written_flags(ir_func), effective_args):
        if isinstance(arg, Field):
            buf = arg._buffer
        elif isinstance(arg, Texture3D):
            buf = arg.field._buffer
        else:
            continue
        try:
            span = buf.span
        except AttributeError:
            return False
        if span[1] > span[0]:
            if is_written:
                written.append(len(spans))
            spans.append(span)
    if not written or len(spans) < 2:
        return True
    # Each written range against every other one. Kernels write to a
    # handful of fields at most, so this beats sorting the ranges.
    for w in written:
        start, end = spans[w]
        for i, (other_start, other_end) in enumerate(spans):
            if other_start < end and start < other_end and i != w:
                return False
    return True


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


class _ProbeParam:
    """A blank parameter for `_KeyProbe` to have the passes write onto.

    Nothing is carried over from the template's own param: the two passes
    assign every attribute read back below, so starting empty is both
    faithful and cheaper than copying. Slots rather than a dict because one
    of these is built per parameter per dispatch.
    """

    __slots__ = (
        "_is_field",
        "_is_texture",
        "_texture_shape",
        "name",
        "type_annotation",
    )

    def __init__(self, name):
        self.name = name
        self.type_annotation = None
        self._is_field = False
        self._is_texture = False
        self._texture_shape = None


class _KeyProbe:
    """A stand-in carrying only what the key derivation touches.

    `infer_param_types` and `store_texture_shapes` record their answers by
    writing onto the params they are handed, and the key is then read back
    off those same params. The IR template is shared by every dispatch of a
    kernel, so doing that to the template publishes one call's argument
    types to every other thread for the window between the write and the
    read — long enough for a concurrent dispatch to compute its key from
    somebody else's dtypes and fetch a variant compiled for them, which is
    silent wrong numbers.

    Standing in a fresh parameter list closes that. The body, which neither
    pass touches, stays shared.
    """

    __slots__ = ("name", "params")

    def __init__(self, ir_func):
        self.name = ir_func.name
        self.params = [_ProbeParam(p.name) for p in ir_func.params]


class KernelVariant:
    """One compiled specialization of a kernel, plus the IR behind it.

    `ir` is the post-pass IR — kept so the loop range can be resolved from
    it on every dispatch without re-running the passes. `payload` is
    whatever the backend needed to cache alongside it.
    """

    __slots__ = ("atomic_targets", "ir", "payload", "requires_full_workgroups")

    def __init__(self, ir_func, payload, *, requires_full_workgroups=False, atomic_targets=()):
        self.ir = ir_func
        self.payload = payload
        self.requires_full_workgroups = requires_full_workgroups
        self.atomic_targets = atomic_targets


def resolve_variant(backend, kernel, args, kwargs, build,
                    store_texture_shapes=None,
                    specialize_disjoint=False) -> tuple:
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

    `specialize_disjoint` asks for a separate variant when this call's field
    storage is provably non-overlapping (see `fields_disjoint`). The answer
    is part of the key and is recorded on the variant's IR as
    `disjoint_fields`, for the backend's codegen to act on.

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

    check_workgroup_support(
        template, supports_workgroups=backend.supports_workgroups,
        backend_label=backend.label, cache_features=True,
    )

    name_to_field = dispatch_name_to_field(template, effective_args)

    # Parameter types and texture extents come from the actual arguments and
    # are part of the key, so they have to be derived before the lookup.
    # They are derived on a private copy of the parameter list rather than on
    # the template — see _KeyProbe for why the template must not be written
    # to here. `shape_signature` only reads, so it takes the template.
    probe = _KeyProbe(template)
    infer_param_types(probe, effective_args)
    store_texture_shapes(probe, effective_args)

    disjoint = specialize_disjoint and fields_disjoint(template, effective_args)
    key = kernel_variant_key(probe, kernel, vector_fields, template_args,
                             shape_signature(template, name_to_field),
                             disjoint)

    slot = kernel_cache_slot(backend._cache, kernel)
    variant = slot.get(key)
    if variant is None:
        from tack.lang.ir_optimize import optimize_ir
        from tack.lang.ir_resolve import resolve_ir
        from tack.lang.ir_verify import verify_ir

        ir_func = clone_ir(template)
        resolve_ir(ir_func, name_to_field)
        verify_ir(ir_func, 'resolved')
        infer_param_types(ir_func, effective_args)
        store_texture_shapes(ir_func, effective_args)
        verify_ir(ir_func, 'inferred')
        check_dispatch_types(ir_func, effective_args,
                             supported_dtypes=backend.supported_dtypes,
                             backend_name=backend.label)
        _localize_outer_scalars(ir_func)
        verify_ir(ir_func, 'localized')
        atomic_targets = check_atomic_support(
            ir_func, backend_name=backend.name,
            supported_dtypes=backend.supported_atomic_dtypes,
        )
        check_atomic_alignment(ir_func.name, atomic_targets, effective_args)
        full_groups = (backend.supports_workgroups and
                       check_workgroup_participation(ir_func))
        if full_groups:
            check_workgroup_launch(ir_func.name, _get_loop_range(ir_func, effective_args),
                                   backend_label=backend.label)
        optimize_ir(ir_func)
        verify_ir(ir_func, 'optimized')
        ir_func.disjoint_fields = disjoint
        variant = KernelVariant(ir_func, build(ir_func, effective_args),
                                requires_full_workgroups=full_groups,
                                atomic_targets=atomic_targets)
        slot[key] = variant
    elif variant.requires_full_workgroups:
        check_workgroup_launch(variant.ir.name, _get_loop_range(variant.ir, effective_args),
                               backend_label=backend.label)

    check_atomic_alignment(variant.ir.name, variant.atomic_targets, effective_args)
    return variant, effective_args


def _localize_outer_scalars(ir_func):
    """Give each outer scalar the loop body assigns to a per-iteration local.

    A scalar parameter is one value shared by every iteration, and codegen
    reads it straight from the argument — or, on GPU, from the packed scalar
    buffer, where every read of the name is rewritten to a buffer load. An
    assignment to that name was therefore lost on GPU, and on CPU reached
    only the reads emitted after it. Renaming the name inside the loop body
    to a local seeded from the parameter makes it an ordinary variable on
    every backend, fresh in each iteration.

    A local assigned before the loop has the same problem on CPU: the
    statements before the loop run once per chunk, so an iteration that
    reassigned it passed its value on to the next iteration of the chunk.
    On GPU each thread runs them for its one iteration, which is the
    defined behavior; the same renaming gives it on every backend.

    Needs the `_is_field` annotations, so it runs after type inference.
    """
    from tack.lang.ir_names import fresh_name, ir_names

    used_names = ir_names(ir_func)
    scalars = {p.name for p in ir_func.params
               if not getattr(p, '_is_field', True)}
    for stmt in ir_func.body:
        if not isinstance(stmt, ir.IRParallelFor):
            scalars.update(n.target for n in _walk_ir(stmt) if isinstance(n, ir.IRAssign))
            continue
        bound = [n.target if isinstance(n, ir.IRAssign) else n.var
                 for n in _walk_ir(stmt.body)
                 if isinstance(n, (ir.IRAssign, ir.IRSequentialFor))]
        assigned = scalars.intersection(bound)
        if not assigned:
            continue
        renames = {name: fresh_name(f"__{name}_local__", used_names)
                   for name in sorted(assigned)}
        # The loop bound can share nodes with the body, and it has to keep
        # naming the parameter: dispatch evaluates it against the arguments.
        body = clone_ir(stmt.body)
        for node in _walk_ir(body):
            if isinstance(node, ir.IRName) and node.name in renames:
                node.name = renames[node.name]
            elif isinstance(node, ir.IRAssign) and node.target in renames:
                node.target = renames[node.target]
            elif isinstance(node, ir.IRSequentialFor) and node.var in renames:
                node.var = renames[node.var]
        seeds = [ir.IRAssign(renames[name], ir.IRName(name))
                 for name in sorted(assigned)]
        stmt.body = seeds + body


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


def check_launch_size(what: str, items: int, max_items: int, backend_label: str):
    """Reject a launch larger than one grid of the backend can index.

    Past the limit a driver either refuses the grid with a bare error code,
    or a narrowed count or thread position wraps and the launch silently
    skips or repeats work. `what` names the kernel or operation.
    """
    if items > max_items:
        raise ValueError(
            f"{what}: {items} iterations exceed the {max_items} that one "
            f"{backend_label} launch can index; split the work across "
            f"several launches.")


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

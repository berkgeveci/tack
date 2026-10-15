"""Shared kernel utility functions used by all backends.

These helpers handle template detection, vector array detection, texture detection,
loop range resolution, and scalar packing — common pre-dispatch logic shared
by every backend.

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

import struct
import threading
import weakref

import numpy as np

from tack.lang import ir
from tack.lang.array import Array, Texture3D
from tack.lang.atomic_support import check_atomic_alignment, check_atomic_support
from tack.lang.ir_traversal import clone_ir
from tack.lang.ir_traversal import walk_ir as _walk_ir
from tack.lang.type_inference import check_dispatch_types, infer_param_types
from tack.lang.types import integer_type_for_value
from tack.lang.workgroup_participation import (
    check_workgroup_launch,
    check_workgroup_participation,
)
from tack.lang.workgroup_support import check_workgroup_support
from tack.runtime import debug_bounds, dispatch


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
    # The dispatch fast path holds the variants it resolved; forget them too.
    for per_backend in list(_fast_caches.values()):
        per_backend.pop(kernel, None)


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


def kernel_variant_key(ir_func, kernel, vector_arrays, template_args,
                       shape_sig=(), disjoint=False) -> tuple:
    """Build the cache key distinguishing compiled variants of one kernel.

    The kernel identity is carried by the enclosing per-kernel slot, so this
    only needs to separate specializations: argument types and categories,
    vector widths, texture shapes, template structure/constants, and every
    dimension size the resolve pass bakes into generated code (``shape_sig``,
    from ``shape_signature``).  ``disjoint`` separates code compiled with a
    no-overlap promise on its array pointers from code compiled without one;
    only backends that ask for that specialization ever pass it.

    Leaving the dimension sizes out is a correctness bug, not a missed
    optimization: a kernel that indexes ``a[i, j]`` compiles the row stride
    in as a literal, so reusing that code for a differently shaped array
    reads the wrong addresses and silently returns wrong numbers.
    """
    # One pass over the params: this runs on every dispatch, and a separate
    # walk per property cost a measurable few microseconds.
    param_sig = tuple(
        (p.type_annotation, getattr(p, '_is_array', True),
         getattr(p, '_is_texture', False), getattr(p, '_texture_shape', None))
        for p in ir_func.params)
    vec_sig = tuple(sorted(vector_arrays.items())) if vector_arrays else ()
    tmpl_key = ()
    if template_args:
        # Keep the structural key: stringifying it loses class identity.
        tmpl_key = kernel._make_cache_key(vector_arrays, template_args)
    return (param_sig, vec_sig, tmpl_key, shape_sig, disjoint)


def _static_array_aliases(ir_func) -> dict:
    """Map each alias name to the array name it ultimately refers to.

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
    """Which array dimensions the resolve pass will bake into this IR.

    Returns a tuple of ``(array_name, dim)``, memoized on the IR function —
    it depends on the kernel's source, not on the arguments. Most kernels
    index one-dimensionally and get back an empty tuple, so the per-dispatch
    cost of keying on shapes is nothing at all.
    """
    deps = ir_func.__dict__.get('_shape_deps')
    if deps is not None:
        return deps

    aliases = _static_array_aliases(ir_func)
    found = set()
    for node in _dependent_nodes(ir_func):
        if isinstance(node, ir.IRDimSize):
            found.add((aliases.get(node.array_name, node.array_name), node.dim))
        elif isinstance(node, ir.IRTextureSample):
            # The sampled extent is embedded in the generated code too.
            name = aliases.get(node.array_name, node.array_name)
            found.update((name, d) for d in range(3))
    deps = tuple(sorted(found))
    ir_func._shape_deps = deps
    return deps


def shape_signature(ir_func, name_to_array) -> tuple:
    """The concrete dimension sizes this call would bake into the code."""
    deps = ir_shape_deps(ir_func)
    if not deps:
        return ()
    sig = []
    for name, dim in deps:
        array = name_to_array.get(name)
        if array is None:
            sig.append(None)
            continue
        shape = getattr(array, 'shape_3d', None) \
            or getattr(array, '_logical_shape', None) or array.shape
        sig.append(shape[dim] if dim < len(shape) else None)
    return tuple(sig)


def written_array_params(ir_func):
    """Names of the array parameters this kernel may store to.

    Returns a frozenset, or ``None`` when a store goes through something
    this cannot trace back to a parameter or a local allocation, in which
    case every array has to be assumed written.  Memoized on the IR
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
        elif isinstance(node, (ir.IRArrayStore, ir.IRAtomicOp)):
            stores.append(node.array)

    # A name can be bound to different arrays on different paths, so each
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
    for array in stores:
        name = getattr(array, 'name', None) if isinstance(array, ir.IRName) \
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
    """`written_array_params` as one bool per parameter, memoized."""
    flags = ir_func.__dict__.get('_written_flags')
    if flags is None:
        written = written_array_params(ir_func)
        flags = tuple(written is None or p.name in written
                      for p in ir_func.params)
        ir_func._written_flags = flags
    return flags


def _array_slots(ir_func, effective_args) -> tuple:
    """``(index, written, through_texture)`` of each array argument, for
    `_disjoint_by_slots`: what `arrays_disjoint` works out per call."""
    slots = []
    for i, (is_written, arg) in enumerate(zip(_written_flags(ir_func), effective_args)):
        if isinstance(arg, Array):
            slots.append((i, bool(is_written), False))
        elif isinstance(arg, Texture3D):
            slots.append((i, bool(is_written), True))
    return tuple(slots)


def _disjoint_by_slots(effective_args, array_slots) -> bool:
    """`arrays_disjoint` for a call whose argument kinds are known.

    One walk of the ranges in start order, instead of every written range
    against every other: a dataset view expands to twenty arrays, and
    that was 20 µs a launch. Walking, a range overlaps an earlier one iff
    it starts before the furthest end so far; it overlaps an earlier
    *written* one iff it starts before the furthest written end so far.
    """
    spans = []
    for i, is_written, through_texture in array_slots:
        arg = effective_args[i]
        buf = arg._storage._buffer if through_texture else arg._buffer
        try:
            start, end = buf.span
        except AttributeError:
            return False
        if end > start:
            spans.append((start, end, is_written))
    if len(spans) < 2:
        return True
    spans.sort()
    furthest = furthest_written = 0
    for start, end, is_written in spans:
        if start < furthest_written or (is_written and start < furthest):
            return False
        if end > furthest:
            furthest = end
        if is_written and end > furthest_written:
            furthest_written = end
    return True


def arrays_disjoint(ir_func, effective_args) -> bool:
    """Whether this call's array storage satisfies a no-overlap promise.

    True when no array the kernel stores to shares a byte with any other
    array argument.  Arrays that are only read may overlap each other
    freely -- ``dot(x, x)`` is still disjoint in the sense that matters,
    because nothing read through one pointer can change under the other.

    Needs host addresses, so it is only meaningful for buffers that have a
    ``span``; an array without one makes the answer False.

    This runs on every dispatch, hence the plain loops.
    """
    spans = []
    written = []
    for is_written, arg in zip(_written_flags(ir_func), effective_args):
        if isinstance(arg, Array):
            buf = arg._buffer
        elif isinstance(arg, Texture3D):
            buf = arg._storage._buffer
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
    # handful of arrays at most, so this beats sorting the ranges.
    for w in written:
        start, end = spans[w]
        for i, (other_start, other_end) in enumerate(spans):
            if other_start < end and start < other_end and i != w:
                return False
    return True


def dispatch_name_to_array(ir_func, effective_args) -> dict:
    """Map parameter names to the Array/Texture3D arguments bound to them."""
    from tack.lang.array import Texture3D
    mapping = {}
    for param, arg in zip(ir_func.params, effective_args):
        if isinstance(arg, (Array, Texture3D)):
            mapping[param.name] = arg
    return mapping


def bind_textures(args) -> list:
    """Replace each Texture3D with the storage its compiled kernel binds.

    That is the texture's own snapshot -- a private array, or a hardware
    image -- never the array it was made from, so writes to that array do
    not reach the kernel before ``Texture3D.update()``.
    """
    return [a._storage if isinstance(a, Texture3D) else a for a in args]


def _store_texture_shapes(ir_func, effective_args):
    """Record Texture3D extents on the params, for codegen and the key."""
    from tack.lang.array import Texture3D
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
        "_is_array",
        "_is_texture",
        "_texture_shape",
        "name",
        "type_annotation",
    )

    def __init__(self, name):
        self.name = name
        self.type_annotation = None
        self._is_array = False
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
    whatever the backend needed to cache alongside it. `written_arrays`
    lists ``(index, name)`` for each array argument the kernel may store
    to, so a dispatch can refuse read-only storage without walking IR.

    `dispatch_lock` serializes dispatches of this variant on backends whose
    payload holds per-variant launch state: the GPU scalar pack buffers,
    and Metal's argument buffer. Two threads dispatching one variant would
    otherwise each write their scalars and array bindings into the same
    buffers, and a launch could read the other call's -- measured on CUDA,
    half of the dispatches from four threads computed with another
    thread's scalars. The lock is held across the synchronous launch;
    different variants still dispatch concurrently.
    """

    __slots__ = ("atomic_targets", "dispatch_lock", "has_textures", "ir", "payload",
                 "requires_full_workgroups", "written_arrays")

    def __init__(self, ir_func, payload, *, requires_full_workgroups=False,
                 atomic_targets=(), written_arrays=(), has_textures=False):
        self.ir = ir_func
        self.payload = payload
        self.requires_full_workgroups = requires_full_workgroups
        self.atomic_targets = atomic_targets
        self.written_arrays = written_arrays
        # Whether any argument is a Texture3D, so a dispatch without one
        # skips the two walks that rebind textures.
        self.has_textures = has_textures
        self.dispatch_lock = threading.Lock()


def _written_array_args(template, effective_args) -> tuple:
    """``(index, name)`` of each array argument the kernel may store to."""
    return tuple(
        (index, param.name)
        for index, (param, written, arg) in enumerate(
            zip(template.params, _written_flags(template), effective_args))
        if written and isinstance(arg, Array))


def check_writable_arrays(kernel_name, written_arrays, args):
    """Refuse a read-only array where the kernel may store to it.

    `array_from_ptr` makes read-only arrays by default, and a read-only
    DLPack tensor imports as one; `copyto` and `fill` already honour
    that. A kernel store would otherwise write straight through it.
    Runs on every dispatch, over only the arrays the kernel writes.
    """
    for index, name in written_arrays:
        if not args[index]._writable:
            raise ValueError(
                f"Kernel '{kernel_name}': parameter '{name}' may be written, "
                f"but its array is read-only. Pass writable=True to "
                f"array_from_ptr() if the memory may be modified.")


def resolve_variant(backend, kernel, args, kwargs, build,
                    store_texture_shapes=None,
                    specialize_disjoint=False) -> tuple:
    """Find or build the compiled variant for this call.

    Template arguments are read once per call (``template_scope``): their
    cache key, expansion and vector arrays share one walk of each tree.

    On a cache hit this copies no IR: type inference runs on a stand-in
    parameter list (`_KeyProbe`), and the per-call checks read only what
    the variant recorded. Only on a miss is the pristine template cloned
    (`clone_ir`), with resolve → infer → check → optimize run on the
    clone, which is then handed to `build` for the backend-specific tail
    (annotate, any packing, compile) that returns the payload to cache.

    `store_texture_shapes` overrides how Texture3D extents are recorded on
    the params — Level Zero falls back to software sampling on devices
    without hardware samplers, and that choice changes the generated code,
    so it has to happen before the key is built.

    `specialize_disjoint` asks for a separate variant when this call's array
    storage is provably non-overlapping (see `arrays_disjoint`). The answer
    is part of the key and is recorded on the variant's IR as
    `disjoint_arrays`, for the backend's codegen to act on.

    Returns ``(variant, effective_args)``.
    """
    from tack.lang.template_rewrite import template_scope

    # The fast path reads no template trees, so it runs outside the scope:
    # entering one cost 2-3 µs of a 45 µs warm dispatch. Debug mode takes
    # the slow path every call: its variants take hidden arguments the
    # fast path does not append, and each launch is noted for the check.
    if not kwargs and not dispatch._debug:
        hit = _fast_resolve(backend, kernel, args, specialize_disjoint)
        if hit is not None:
            return hit
    with template_scope():
        return _resolve_variant(backend, kernel, args, kwargs, build, store_texture_shapes,
                                specialize_disjoint)


def _fast_resolve(backend, kernel, args, specialize_disjoint):
    """A call whose signature was seen before reuses what that call resolved
    -- its variants and the plan for expanding its template arguments --
    and runs only the per-call checks. See `_call_signature` for what the
    signature covers, and why a hit implies the same key. None otherwise."""
    sig = _call_signature(args)
    if sig is None:
        return None
    hit = _fast_slot(backend, kernel).get(sig)
    if hit is None:
        return None
    # The entry holds the variants it resolved (dropped with the slot's by
    # `drop_variants`) and, for the disjointness check, which arguments are
    # arrays and which of those the kernel writes: fixed by the signature,
    # so no type test per call.
    plan, variants, array_slots = hit
    effective_args = _expand_with_plan(args, plan) if plan else args
    disjoint = specialize_disjoint and _disjoint_by_slots(effective_args, array_slots)
    variant = variants.get(disjoint)
    if variant is None:
        return None
    _check_call(backend, variant, effective_args)
    return variant, effective_args


def _resolve_variant(backend, kernel, args, kwargs, build, store_texture_shapes,
                     specialize_disjoint):
    if store_texture_shapes is None:
        store_texture_shapes = _store_texture_shapes
    if kwargs:
        raise NotImplementedError("Keyword arguments not supported in kernels")
    debug = dispatch._debug
    if debug:
        debug_bounds.check()        # a launch that came without a check

    # The fast path (`_fast_resolve`) ran before this, outside the template
    # scope; this is the slow path, which records what it resolves for it.
    fast = _fast_slot(backend, kernel)
    sig = _call_signature(args)

    template_args = _detect_template_args(kernel, args)
    effective_args = _expand_template_args(args, template_args)
    vector_arrays = _detect_vector_arrays_from_args(kernel, args, template_args)
    texture_arrays = _detect_texture_arrays(kernel, args, template_args)

    # The pristine IR for this specialization. Never mutated below the
    # parameter list — the passes run on a copy.
    template = kernel.get_ir(
        vector_arrays,
        template_args=template_args if template_args else None,
        texture_arrays=texture_arrays,
        debug=debug,
    ).functions[0]
    if debug:
        effective_args = debug_bounds.hidden_args(backend, template, effective_args)

    check_workgroup_support(
        template, supports_workgroups=backend.supports_workgroups,
        backend_label=backend.label, cache_features=True,
    )

    name_to_array = dispatch_name_to_array(template, effective_args)

    # Parameter types and texture extents come from the actual arguments and
    # are part of the key, so they have to be derived before the lookup.
    # They are derived on a private copy of the parameter list rather than on
    # the template — see _KeyProbe for why the template must not be written
    # to here. `shape_signature` only reads, so it takes the template.
    probe = _KeyProbe(template)
    infer_param_types(probe, effective_args)
    store_texture_shapes(probe, effective_args)

    disjoint = specialize_disjoint and arrays_disjoint(template, effective_args)
    shape_sig = shape_signature(template, name_to_array)
    key = kernel_variant_key(probe, kernel, vector_arrays, template_args, shape_sig, disjoint)

    slot = kernel_cache_slot(backend._cache, kernel)
    variant = slot.get(key)
    if variant is None:
        from tack.runtime import disk_cache
        disk_key = disk_cache.variant_key(backend, template, probe, vector_arrays,
                                          shape_sig, disjoint)
        variant = _stored_variant(backend, disk_key, template, effective_args)
        if variant is not None:
            slot[key] = variant
    if variant is None:
        from tack.lang.ir_optimize import optimize_ir
        from tack.lang.ir_resolve import resolve_ir
        from tack.lang.ir_verify import verify_ir

        ir_func = clone_ir(template)
        resolve_ir(ir_func, name_to_array)
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
        ir_func.disjoint_arrays = disjoint
        variant = KernelVariant(ir_func, build(ir_func, effective_args),
                                requires_full_workgroups=full_groups,
                                atomic_targets=atomic_targets,
                                written_arrays=_written_array_args(
                                    template, effective_args),
                                has_textures=any(isinstance(a, Texture3D)
                                                 for a in effective_args))
        slot[key] = variant
        _store_variant(backend, disk_key, variant)
    elif variant.requires_full_workgroups:
        check_workgroup_launch(variant.ir.name, _get_loop_range(variant.ir, effective_args),
                               backend_label=backend.label)

    if sig is not None and not debug:      # see resolve_variant
        entry = fast.get(sig)
        if entry is None:
            if len(fast) >= _FAST_CAP:
                fast.clear()
            entry = fast[sig] = (_expansion_plan(template_args), {},
                                 _array_slots(template, effective_args))
        entry[1][disjoint] = variant

    check_atomic_alignment(variant.ir.name, variant.atomic_targets, effective_args)
    check_writable_arrays(variant.ir.name, variant.written_arrays, effective_args)
    if debug:
        debug_bounds.launched(effective_args, template)
    return variant, effective_args


def _stored_variant(backend, disk_key, template, effective_args):
    """The variant a previous process compiled for this call, or None.

    The entry holds the IR after the passes and the backend's artifact;
    loading it skips the passes, code generation and, as far as the backend
    can, compilation. The per-call checks below the caller still run. A
    damaged or unloadable entry is ignored, and the variant compiled anew.
    """
    from tack.runtime import disk_cache
    if disk_key is None:
        return None
    entry = disk_cache.load_variant(disk_key)
    if entry is None:
        return None
    import pickle
    try:
        ir_func = pickle.loads(entry["ir"])
        payload = backend.load_variant(ir_func, entry["artifact"], effective_args)
    except Exception:
        disk_cache._count("bad")
        return None
    variant = KernelVariant(ir_func, payload,
                            requires_full_workgroups=entry["full_groups"],
                            atomic_targets=entry["atomic_targets"],
                            written_arrays=_written_array_args(template, effective_args),
                            has_textures=any(isinstance(a, Texture3D) for a in effective_args))
    if variant.requires_full_workgroups:
        check_workgroup_launch(ir_func.name, _get_loop_range(ir_func, effective_args),
                               backend_label=backend.label)
    return variant


def _store_variant(backend, disk_key, variant):
    """Keep a newly compiled variant for later processes, if it can be kept."""
    from tack.runtime import disk_cache
    if disk_key is None:
        return
    import pickle
    try:
        entry = {"ir": pickle.dumps(variant.ir, protocol=5),
                 "artifact": backend.variant_artifact(variant.payload),
                 "full_groups": variant.requires_full_workgroups,
                 "atomic_targets": variant.atomic_targets}
    except Exception:
        disk_cache._count("variant_unpicklable")
        return
    disk_cache.store_variant(disk_key, entry)


def _check_call(backend, variant, effective_args):
    """What every dispatch checks against its actual arguments, hit or miss."""
    if variant.requires_full_workgroups:
        check_workgroup_launch(variant.ir.name, _get_loop_range(variant.ir, effective_args),
                               backend_label=backend.label)
    check_atomic_alignment(variant.ir.name, variant.atomic_targets, effective_args)
    check_writable_arrays(variant.ir.name, variant.written_arrays, effective_args)


# ── The dispatch fast path ──────────────────────────────────────────
#
# Resolving a variant from scratch walks the arguments half a dozen times:
# template detection and expansion, vector and texture detection, the IR
# lookup, type inference on a probe, the shape signature, the variant key.
# Each is a few microseconds; together they were the largest part of a
# dispatch's host time (90 of ~270 µs per launch of a contour's kernels on
# a 2 GHz Xeon, where the kernels themselves ran 0.1-14 ms). A filter is
# dozens of such launches, and on a GPU the host side is most of the call.
#
# The signature below is what the variant key is a function of, read
# straight off the arguments: for an array its dtype, shape and element
# layout; for a scalar its kind and integer width; for a template object
# its class, its class constants' current values, and for each attribute
# the kind, dtype, shape or function that `_template_key` and the
# expansion read -- nested templates included, by path. Equal signatures
# therefore mean an equal key (the full shape is more than the key needs,
# so distinct lengths miss, harmlessly, and find the same variant). What a
# hit must still do per call -- expand the templates, decide disjointness,
# check atomic alignment, writable arrays and the workgroup launch -- it
# does. An entry names its variants by key and holds no class or variant
# itself, so a template class defined per call is collected as before and
# a variant dropped by `drop_variants` is resolved afresh.

_fast_caches = weakref.WeakKeyDictionary()     # backend -> kernel -> {sig: entry}
_FAST_CAP = 256                                # signatures kept per kernel and backend


def _fast_slot(backend, kernel) -> dict:
    try:
        per_backend = backend._fast_slots        # the registry's dict, kept on the backend
    except AttributeError:
        per_backend = _fast_caches.get(backend)
        if per_backend is None:
            per_backend = _fast_caches.setdefault(backend, weakref.WeakKeyDictionary())
        backend._fast_slots = per_backend
    slot = per_backend.get(kernel)
    if slot is None:
        slot = per_backend.setdefault(kernel, {})
    return slot


def _call_signature(args):
    """The call's signature, or None for an argument the fast path does
    not describe (the slow path then reports it)."""
    sig = []
    for arg in args:
        if isinstance(arg, Array):
            try:
                sig.append(arg._sig)
            except AttributeError:
                # Computed once per array: its dtype, shape and element layout
                # are set when it is made and never change (a reshape is a
                # new Array over the same buffer).
                arg._sig = (0, arg.dtype, arg.shape, getattr(arg, '_logical_shape', None),
                            getattr(arg, '_vector_n', None), getattr(arg, '_matrix_shape', None))
                sig.append(arg._sig)
        elif isinstance(arg, Texture3D):
            sig.append((1, arg.dtype, arg.shape_3d))
        elif isinstance(arg, float):
            sig.append(2)
        elif isinstance(arg, int):
            sig.append((3, integer_type_for_value(arg)))
        elif isinstance(arg, np.floating):
            sig.append(2)
        elif isinstance(arg, np.integer):
            sig.append((3, integer_type_for_value(int(arg))))
        elif getattr(type(arg), '_device_class', False) is True:
            tmpl = _template_signature(arg, (), set())
            if tmpl is None:
                return None
            sig.append((4, tuple(tmpl)))
        else:
            return None
    return tuple(sig)


def _scalar_bits(value):
    return struct.pack('!d', value) if isinstance(value, float) else value


# Per template class, how `_template_signature` last read an instance: the
# attribute names in order, each one's kind (constant, array, int, float,
# function, child template) and its value's type. An instance with the same
# names and types is read by kind, one attribute access each, instead of
# the type tests per attribute that cost 25 µs a dataset view on a 2 GHz
# Xeon; one that differs is walked in full and the layout replaced.
_template_layouts = weakref.WeakKeyDictionary()
_LAYOUT_CONSTANT, _LAYOUT_ARRAY, _LAYOUT_INT, _LAYOUT_FLOAT, _LAYOUT_FUNC, _LAYOUT_CHILD, \
    _LAYOUT_SKIP = range(7)


def _template_signature(obj, path, seen):
    """What of a template object the IR and the expansion depend on, for it
    and every template it holds; None on a cycle (the slow path explains)."""
    if id(obj) in seen:
        return None
    seen.add(id(obj))
    cls = type(obj)
    attributes = vars(obj)
    layout = _template_layouts.get(cls)
    if layout is not None and layout[0] != tuple(attributes):
        layout = None                         # another attribute set: walk and relearn
    if layout is not None:
        from tack.lang.template_rewrite import _class_constants
        constants = _class_constants(cls)     # current values: a class constant can change
        _names, kinds, types = layout
        own = []
        children = []
        for (name, value), kind, typ in zip(attributes.items(), kinds, types):
            if type(value) is not typ:
                layout = None
                break
            if kind == _LAYOUT_ARRAY:
                try:
                    own.append((name, *value._sig))
                except AttributeError:
                    own.append((name, 0, value.dtype, value.shape,
                                getattr(value, '_logical_shape', None),
                                getattr(value, '_vector_n', None),
                                getattr(value, '_matrix_shape', None)))
            elif kind == _LAYOUT_CONSTANT:
                own.append((name, typ, _scalar_bits(value)))
            elif kind == _LAYOUT_INT:
                own.append((name, 3, integer_type_for_value(value)))
            elif kind == _LAYOUT_FLOAT:
                own.append((name, 2))
            elif kind == _LAYOUT_FUNC:
                own.append((name, 5, value))
            elif kind == _LAYOUT_CHILD:
                children.append((name, value))
    if layout is None:
        from tack.lang.func import Func
        from tack.lang.template_rewrite import _class_constants
        constants = _class_constants(cls)
        own = []
        children = []
        kinds = []
        for name, value in attributes.items():
            if name.startswith('_'):
                kinds.append(_LAYOUT_SKIP)
            elif name in constants:
                kinds.append(_LAYOUT_CONSTANT)
                own.append((name, type(value), _scalar_bits(value)))
            elif isinstance(value, Array):
                kinds.append(_LAYOUT_ARRAY)
                own.append((name, 0, value.dtype, value.shape,
                            getattr(value, '_logical_shape', None),
                            getattr(value, '_vector_n', None),
                            getattr(value, '_matrix_shape', None)))
            elif isinstance(value, int):
                kinds.append(_LAYOUT_INT)
                own.append((name, 3, integer_type_for_value(value)))
            elif isinstance(value, float):
                kinds.append(_LAYOUT_FLOAT)
                own.append((name, 2))
            elif isinstance(value, Func):
                kinds.append(_LAYOUT_FUNC)
                own.append((name, 5, value))
            elif getattr(type(value), '_device_class', False) is True:
                kinds.append(_LAYOUT_CHILD)
                children.append((name, value))
            else:
                kinds.append(_LAYOUT_SKIP)
        # `isinstance` admitted subclasses above; the layout holds the exact
        # types and is used only for an instance whose types match.
        _template_layouts[cls] = (tuple(attributes), tuple(kinds),
                                  tuple(type(v) for v in attributes.values()))
    from tack.lang.kernel import _class_token
    parts = [(path, _class_token(cls),
              tuple((n, type(v), _scalar_bits(v)) for n, v in constants.items()),
              tuple(own))]
    for name, child in sorted(children):
        below = _template_signature(child, (*path, name), seen)
        if below is None:
            return None
        parts.extend(below)
    seen.discard(id(obj))
    return parts


def _expansion_plan(template_args) -> tuple:
    """How `_expand_template_args` turns these template arguments into
    arguments: for each template, highest index first, the attribute paths
    it appends, arrays then runtime scalars, each sorted by flattened name."""
    from tack.lang.template_rewrite import _classify_own, _prefix, template_tree

    plan = []
    for idx in sorted(template_args, reverse=True):
        _, obj = template_args[idx]
        arrays, scalars = [], []
        for path, node in template_tree(obj):
            _, own_arrays, own_scalars = _classify_own(node)
            for name in own_arrays:
                arrays.append((_prefix(path) + name, (*path, name)))
            for name in own_scalars:
                scalars.append((_prefix(path) + name, (*path, name)))
        arrays.sort()
        scalars.sort()
        plan.append((idx, tuple(p for _, p in arrays) + tuple(p for _, p in scalars)))
    return tuple(plan)


def _expand_with_plan(args, plan) -> tuple:
    templates = {idx for idx, _ in plan}
    new_args = [arg for i, arg in enumerate(args) if i not in templates]
    for idx, paths in plan:
        obj = args[idx]
        for path in paths:
            value = obj
            for name in path:
                value = getattr(value, name)
            new_args.append(value)
    return tuple(new_args)


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

    Needs the `_is_array` annotations, so it runs after type inference.
    """
    from tack.lang.ir_names import fresh_name, ir_names

    used_names = ir_names(ir_func)
    scalars = {p.name for p in ir_func.params
               if not getattr(p, '_is_array', True)}
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
    """Detect which arguments are @tack.device_class template objects.

    Returns dict: param_index -> (param_name, template_object)
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    templates = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if hasattr(arg, '_device_class') and arg._device_class:
            templates[i] = (param_name, arg)
    return templates


def _expand_template_args(args, template_args) -> tuple:
    """Replace template args with their array and runtime scalar attributes.

    Returns new args tuple with template objects removed and their
    array attributes and runtime scalars appended.  These are appended
    in reverse template index order to match the AST rewrite pass
    (which processes templates from highest index to lowest).
    """
    if not template_args:
        return args

    from tack.lang.template_rewrite import classify_template_attrs

    new_args = []
    extra = []
    # Collect template arrays and runtime scalars in reverse index order
    for idx in sorted(template_args.keys(), reverse=True):
        _, obj = template_args[idx]
        _, arrays, runtime_scalars = classify_template_attrs(obj)
        for attr_name in sorted(arrays.keys()):
            extra.append(arrays[attr_name])
        for attr_name in sorted(runtime_scalars.keys()):
            extra.append(runtime_scalars[attr_name])
    # Build non-template args in order
    for i, arg in enumerate(args):
        if i not in template_args:
            new_args.append(arg)
    new_args.extend(extra)
    return tuple(new_args)


def _detect_vector_arrays(kernel, args) -> dict[str, int] | None:
    """Detect which kernel parameters are vector arrays.

    Returns a dict mapping parameter names to component counts,
    or None if no vector arrays are present.
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_arrays = {}
    for param_name, arg in zip(params, args):
        if isinstance(arg, Array) and hasattr(arg, '_vector_n'):
            vector_arrays[param_name] = _element_shape(arg)
    return vector_arrays if vector_arrays else None


def _element_shape(array):
    """What lowering needs to know about a vector or matrix array's elements:
    the component count, or ``(rows, columns)`` for a matrix array."""
    return getattr(array, '_matrix_shape', None) or array._vector_n


def _detect_vector_arrays_from_args(kernel, args, template_args) -> dict[str, int] | None:
    """Detect vector arrays, accounting for template parameters.

    When template args are present, we need to skip them when matching
    parameter names to arguments.
    """
    if not template_args:
        return _detect_vector_arrays(kernel, args)

    from tack.lang.template_rewrite import classify_template_attrs, template_array_param_name

    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_arrays = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if i in template_args:
            # A template's vector array attributes become kernel parameters
            # too, under the rewrite's synthetic names; without this a
            # `self.vel[i, j]` in a template method lowered as a scalar load.
            _, arrays, _ = classify_template_attrs(arg)
            for attr_name, array in arrays.items():
                if hasattr(array, '_vector_n'):
                    vector_arrays[template_array_param_name(param_name, attr_name)] = \
                        _element_shape(array)
            continue
        if isinstance(arg, Array) and hasattr(arg, '_vector_n'):
            vector_arrays[param_name] = _element_shape(arg)
    return vector_arrays if vector_arrays else None


def _detect_texture_arrays(kernel, args, template_args=None) -> dict[str, tuple] | None:
    """Detect which kernel parameters are Texture3D objects."""
    from tack.lang.array import Texture3D
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    texture_arrays = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if template_args and i in template_args:
            continue
        if isinstance(arg, Texture3D):
            texture_arrays[param_name] = arg.shape_3d
    return texture_arrays if texture_arrays else None


def _resolve_range_expr(node: ir.IRNode, name_to_arg: dict) -> int:
    """Resolve a range expression to a concrete integer value."""
    if isinstance(node, ir.IRConstant):
        return int(node.value)

    # x.shape[k] / len(x) in the grid bound. The resolve pass deliberately
    # leaves these alone so the compiled kernel does not depend on the
    # array's length; they are evaluated here instead, per dispatch.
    if isinstance(node, ir.IRDimSize):
        arg = name_to_arg.get(node.array_name)
        if arg is not None:
            shape = getattr(arg, 'shape_3d', None) \
                or getattr(arg, '_logical_shape', None) or arg.shape
            return shape[node.dim]

    # x.shape[0]  →  IRArrayLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))
    if isinstance(node, ir.IRArrayLoad):
        obj = node.array
        if isinstance(obj, ir.IRAttribute) and obj.attr == "shape":
            if isinstance(obj.obj, ir.IRName):
                arg = name_to_arg.get(obj.obj.name)
                if isinstance(arg, Array):
                    idx = _resolve_range_expr(node.index, name_to_arg)
                    return arg.shape[idx]

    # len(x)  →  IRAttribute(IRName("x"), "__len__")
    if isinstance(node, ir.IRAttribute) and node.attr == "__len__":
        if isinstance(node.obj, ir.IRName):
            arg = name_to_arg.get(node.obj.name)
            if isinstance(arg, Array):
                return arg.shape[0]

    # Binary ops on range expressions (e.g., n - 1)
    if isinstance(node, ir.IRBinOp):
        left = _resolve_range_expr(node.left, name_to_arg)
        right = _resolve_range_expr(node.right, name_to_arg)
        ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
               "*": lambda a, b: a * b, "//": lambda a, b: a // b}
        if node.op in ops:
            return ops[node.op](left, right)

    # A negated bound: range(-n, n + 1), ndindex((-n, n + 1), ...)
    if isinstance(node, ir.IRUnaryOp) and node.op in ("-", "+"):
        value = _resolve_range_expr(node.operand, name_to_arg)
        return -value if node.op == "-" else value

    # Plain name reference (e.g., `n` passed as scalar)
    if isinstance(node, ir.IRName):
        arg = name_to_arg.get(node.name)
        if arg is not None:
            return int(arg)

    # The extent of an ndindex (start, end) pair: max(end - start, 0)
    if (isinstance(node, ir.IRCall) and node.func_name in ("min", "max")
            and len(node.args) == 2):
        left = _resolve_range_expr(node.args[0], name_to_arg)
        right = _resolve_range_expr(node.args[1], name_to_arg)
        return max(left, right) if node.func_name == "max" else min(left, right)

    # A typed tack.constant: an integer literal under a cast to its type.
    # The constant was checked to fit that type, so the cast changes nothing.
    if (isinstance(node, ir.IRCast) and isinstance(node.value, ir.IRConstant)
            and isinstance(node.value.value, int)):
        return node.value.value

    raise RuntimeError(f"Cannot resolve loop range expression: {type(node).__name__}")


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments.

    Resolves the loop end expression — supports:
      - IRConstant(N)
      - IRArrayLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))  →  x.shape[0]
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


def _get_loop_extents(ir_func: ir.IRFunction, args: tuple) -> tuple:
    """A multi-dimensional parallel loop's extents for these arguments, slowest first.

    ``()`` for a flat loop. Dispatch passes them to the launch (see
    ``tack.lang.parallel_dims``); a dimension of size zero or less makes the
    whole launch empty, as its product would.
    """
    for stmt in ir_func.body:
        if isinstance(stmt, ir.IRParallelFor):
            if not stmt.dims:
                return ()
            name_to_arg = dict(zip((p.name for p in ir_func.params), args))
            return tuple(_resolve_range_expr(e, name_to_arg) for e in stmt.extents)
    raise RuntimeError("Kernel has no parallel for-loop")


def _launch_plan(ir_func: ir.IRFunction) -> tuple:
    """How to evaluate the parallel loop's end and extents from the
    arguments, found once per IR: a constant, a dimension of one argument,
    a scalar argument, or (anything else) the expression to resolve."""
    plan = ir_func.__dict__.get('_launch_plan')
    if plan is None:
        loop = next((s for s in ir_func.body if isinstance(s, ir.IRParallelFor)), None)
        if loop is None:
            raise RuntimeError("Kernel has no parallel for-loop")
        index = {p.name: i for i, p in enumerate(ir_func.params)}

        def step(node):
            if isinstance(node, ir.IRConstant):
                return (0, int(node.value))
            if isinstance(node, ir.IRDimSize) and node.array_name in index:
                return (1, index[node.array_name], node.dim)
            if isinstance(node, ir.IRName) and node.name in index:
                return (2, index[node.name])
            return (3, node)

        plan = ir_func._launch_plan = (
            step(loop.end), tuple(step(e) for e in loop.extents) if loop.dims else ())
    return plan


def _launch_value(step, ir_func, args) -> int:
    kind = step[0]
    if kind == 0:
        return step[1]
    if kind == 1:
        arg = args[step[1]]
        shape = getattr(arg, 'shape_3d', None) \
            or getattr(arg, '_logical_shape', None) or arg.shape
        return shape[step[2]]
    if kind == 2:
        return int(args[step[1]])
    return _resolve_range_expr(step[1], dict(zip((p.name for p in ir_func.params), args)))


def _get_launch(ir_func: ir.IRFunction, args: tuple) -> tuple:
    """``(items, extents)``: the parallel loop's iteration count and its extents.

    ``items`` is 0 when any extent is, so an empty launch is skipped even
    where the product of two negative sizes would be positive.
    """
    end, extent_steps = _launch_plan(ir_func)
    items = _launch_value(end, ir_func, args)
    extents = tuple(_launch_value(e, ir_func, args) for e in extent_steps)
    if extents and min(extents) <= 0:
        items = 0
    return items, extents


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


def _create_pack_arrays(pack_info, args, backend):
    """Create packed Array objects from scalar pack info.

    Args:
        pack_info: list of (pack_name, dtype, [(orig_name, orig_arg_idx, idx_in_pack)])
        args: the original effective_args tuple
        backend: the active backend (for allocate_array)

    Returns:
        list of Array objects, one per pack group.
    """
    import numpy as np

    from tack.lang.array import Array

    arrays = []
    for pack_name, dtype, entries in pack_info:
        np_dtype = dtype.numpy_dtype
        arr = np.zeros(len(entries), dtype=np_dtype)
        for _, orig_arg_idx, idx_in_pack in entries:
            arr[idx_in_pack] = args[orig_arg_idx]
        buf = backend.allocate_array(dtype, (len(entries),))
        f = Array(dtype, (len(entries),), buf)
        f._buffer.from_numpy(arr)
        arrays.append(f)
    return arrays


def _update_pack_arrays(pack_arrays, pack_info, args):
    """Update existing packed Array objects with new scalar values.

    Reuses the allocated device buffers — only copies new values.
    """
    import numpy as np

    for array, (pack_name, dtype, entries) in zip(pack_arrays, pack_info):
        np_dtype = dtype.numpy_dtype
        arr = np.zeros(len(entries), dtype=np_dtype)
        for _, orig_arg_idx, idx_in_pack in entries:
            arr[idx_in_pack] = args[orig_arg_idx]
        array._buffer.from_numpy(arr)

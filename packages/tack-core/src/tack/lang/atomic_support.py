"""Atomic target domains, checked before optimization or device compilation."""

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

ATOMIC_DTYPES = {
    'cpu': frozenset((i8, u8, i16, u16, i32, u32, i64, u64, f32, f64)),
    'cuda': frozenset((i32, u32, i64, u64, f32, f64)),
    'hip': frozenset((i32, u32, i64, u64, f32, f64)),
    'metal': frozenset((i32, u32, f32)),
    'level_zero': frozenset((i32, u32, f32)),
}


def check_atomic_support(func, *, backend_name, supported_dtypes=None):
    """Validate fresh IR; return atomic parameter indices and byte alignment.

    Targets must resolve uniquely to global field parameters, including
    pointer copies introduced by inlining. Private/shared allocations,
    textures and scalar parameters are separate address spaces, not fields.
    Never trust cached node dtype annotations on mutable direct-codegen IR.
    """
    supported = (ATOMIC_DTYPES.get(backend_name, frozenset())
                 if supported_dtypes is None else supported_dtypes)
    params = {p.name: (index, p) for index, p in enumerate(func.params)}
    sources = {name: {name} for name in params}
    copies, atoms = [], []
    for node in walk_ir(func.body):
        if isinstance(node, ir.IRAssign):
            if isinstance(node.value, ir.IRName):
                copies.append((node.target, node.value.name))
            else:
                sources.setdefault(node.target, set()).add(None)
        elif isinstance(node, (ir.IRLocalAlloc, ir.IRSharedAlloc)):
            sources.setdefault(node.name, set()).add(None)
        elif isinstance(node, (ir.IRParallelFor, ir.IRSequentialFor)):
            sources.setdefault(node.var, set()).add(None)
            for dim in getattr(node, 'dims', None) or ():
                sources.setdefault(dim, set()).add(None)
        elif isinstance(node, ir.IRAtomicOp):
            atoms.append(node)
    if not atoms:
        return ()
    changed = True
    while changed:
        changed = False
        for target, source in copies:
            origins = sources.get(source, set())
            reached = sources.setdefault(target, set())
            if not origins <= reached:
                reached.update(origins)
                changed = True
    targets = {}
    for node in atoms:
        context = f"Kernel '{func.name}': {backend_name} atomic_{node.op}"
        if node.op not in ('add', 'min', 'max'):
            raise TypeError(f'{context} is not supported')
        name = node.field.name if isinstance(node.field, ir.IRName) else None
        origins = sources.get(name, set())
        entry = params.get(next(iter(origins))) if len(origins) == 1 else None
        if (entry is None or not getattr(entry[1], '_is_field', True)
                or getattr(entry[1], '_is_texture', False)):
            raise TypeError(f'{context} requires a global field parameter target')
        index, param = entry
        dtype = param.type_annotation
        if dtype not in supported:
            raise TypeError(f'{context} does not support target dtype {dtype}')
        node.dtype = dtype
        targets[index] = dtype.bits // 8
    return tuple(sorted(targets.items()))


def check_atomic_alignment(kernel_name, targets, args):
    """Check current storage even when a compiled specialization is cached."""
    for index, alignment in targets:
        address = args[index]._buffer.address
        if address % alignment:
            raise ValueError(
                f"Kernel '{kernel_name}': atomic target parameter {index} "
                f'requires {alignment}-byte alignment')

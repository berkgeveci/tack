"""Target capability checks for IR that requires a workgroup execution model."""

from tack.lang import ir
from tack.lang.ir_traversal import walk_ir


def workgroup_features(ir_func):
    """Return required primitives, including those in nested or inlined bodies.

    This is structural discovery, not control-flow or participation analysis:
    even a primitive in an unreachable branch requires target support.
    """
    features = set()
    for node in walk_ir(ir_func.body):
        if isinstance(node, ir.IRSharedAlloc):
            features.add('shared_like' if node.field_name else 'shared')
        elif isinstance(node, ir.IRBarrier):
            features.add('barrier')
        elif isinstance(node, ir.IRThreadId):
            features.add('thread_id')
        elif isinstance(node, ir.IRBlockReduce):
            features.add(f'block_{node.op}')
    return tuple(sorted(features))


def check_workgroup_support(ir_func, *, supports_workgroups, backend_label,
                            cache_features=False):
    """Reject unsupported primitives before compilation or execution.

    Only immutable frontend templates may opt into memoization. Direct
    codegen receives mutable IR and must rediscover requirements each time.
    Supported targets bypass discovery; this check does not prove barrier
    uniformity, full-group participation, or atomic type/scope support.
    """
    if supports_workgroups:
        return
    features = getattr(ir_func, '_workgroup_features', None) if cache_features else None
    if features is None:
        features = workgroup_features(ir_func)
        if cache_features:
            ir_func._workgroup_features = features
    if features:
        raise NotImplementedError(
            f"Kernel '{ir_func.name}': {backend_label} backend does not support "
            f"workgroup execution required by {', '.join(features)}. "
            "Use a backend with supports_workgroups=True; for private scratch "
            "arrays, use local_array or local_array_like."
        )

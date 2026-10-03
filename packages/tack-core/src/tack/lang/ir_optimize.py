"""Conservative copy propagation for inlined kernel arguments.

Tack's former LICM treated an invariant address as an invariant value and
could execute a zero-trip loop's assignments. Its CSE invalidation tracked
field names rather than possibly overlapping storage. Neither pass has the
memory and control-flow analysis required to justify those transformations,
so hoisting and CSE are left to LLVM and the vendor compilers.

Copy propagation keeps assignments and replaces only subsequent uses of a
single-assignment copy whose source is not modified anywhere in the kernel.
Assignment counts are computed once and shared by all nested blocks.
"""

from tack.lang import ir
from tack.lang.ir_traversal import transform_ir, walk_ir


def optimize_ir(ir_func: ir.IRFunction):
    """Canonicalize copies without moving or eliminating computations."""
    _copy_prop_function(ir_func)


def _count_assignments(body: list) -> dict[str, int]:
    """Count how many times each variable is assigned in a statement list (recursive)."""
    counts: dict[str, int] = {}
    for node in walk_ir(body):
        if isinstance(node, ir.IRAssign):
            name = node.target
        elif isinstance(node, (ir.IRLocalAlloc, ir.IRSharedAlloc)):
            name = node.name
        elif isinstance(node, (ir.IRParallelFor, ir.IRSequentialFor)):
            name = node.var
        else:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _copy_prop_function(ir_func: ir.IRFunction):
    """Apply copy propagation to the function body."""
    ir_func.body = _copy_prop_body(ir_func.body)


def _copy_prop_body(body: list, assign_counts: dict[str, int] | None = None) -> list:
    """Propagate copies: when x = y (simple name assignment), replace
    subsequent uses of x with y (unless x or y is reassigned later)."""
    # Counts are kernel-wide, not rebuilt for every subtree. Targets and
    # loop/allocation bindings are not renamed by this pass, so the same
    # summary remains valid after substituting name uses. This deliberately
    # leaves some block-local copies to LLVM/the vendor compiler.
    if assign_counts is None:
        assign_counts = _count_assignments(body)

    # Collect simple copies: x = y where x is assigned exactly once
    # AND y is never assigned in the kernel (normally a parameter).
    # This avoids breaking tuple swaps or following a modified source.
    copies = {}  # target -> source name
    for stmt in body:
        if (isinstance(stmt, ir.IRAssign) and
                isinstance(stmt.value, ir.IRName) and
                assign_counts.get(stmt.target, 0) == 1 and
                assign_counts.get(stmt.value.name, 0) == 0):
            copies[stmt.target] = stmt.value.name

    if not copies:
        # Still recurse into sub-blocks
        return _copy_prop_recurse(body, assign_counts)

    # A copy only describes uses after its assignment. Precomputing the
    # replacement map for the whole block changes earlier reads of a
    # parameter that is later reassigned (or a loop-carried local).
    resolved = {}
    result = []
    for stmt in body:
        result.append(_replace_names(stmt, resolved))
        if isinstance(stmt, ir.IRAssign) and stmt.target in copies:
            source = copies[stmt.target]
            resolved[stmt.target] = resolved.get(source, source)

    return _copy_prop_recurse(result, assign_counts)


def _copy_prop_recurse(body: list, assign_counts: dict[str, int]) -> list:
    """Recurse copy propagation into loops and conditionals."""
    for stmt in body:
        if isinstance(stmt, (ir.IRParallelFor, ir.IRSequentialFor, ir.IRWhile)):
            stmt.body = _copy_prop_body(stmt.body, assign_counts)
        elif isinstance(stmt, ir.IRIf):
            stmt.then_body = _copy_prop_body(stmt.then_body, assign_counts)
            if stmt.else_body:
                stmt.else_body = _copy_prop_body(stmt.else_body, assign_counts)
    return body


def _replace_names(node, mapping: dict):
    """Replace name uses while preserving node annotations and metadata."""
    if not mapping:
        return node

    def replace_name(node):
        if isinstance(node, ir.IRName) and node.name in mapping:
            replacement = ir.IRName(mapping[node.name])
            replacement.dtype = node.dtype
            return replacement
        return node

    # A shared expression can occur before and after a copy assignment.
    # Rewriting the later occurrence must not change the earlier one.
    return transform_ir(node, replace_name, copy_nodes=True)

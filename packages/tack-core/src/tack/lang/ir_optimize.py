"""Conservative copy propagation for inlined kernel arguments.

Tack's former LICM treated an invariant address as an invariant value and
could execute a zero-trip loop's assignments. Its CSE invalidation tracked
field names rather than possibly overlapping storage. Neither pass has the
memory and control-flow analysis required to justify those transformations,
so hoisting and CSE are left to LLVM and the vendor compilers.

Copy propagation keeps assignments and replaces only subsequent uses of a
single-assignment copy whose source is not modified anywhere in the kernel.
Assignment counts are computed once and shared by all nested blocks.

Before it, a local assigned exactly once, to a weak literal expression
(``a = 0.1``, ``c = 1.0 / 3.0``), is replaced by that expression wherever
it is read, so it stays weak there and takes the precision of what it
meets, as the literal written in place would. A device function returning
literals binds its result through such locals; without this, a local
holding only literals was an f32 local, and f64 data computed with its
rounded value.
"""

from tack.lang import ir
from tack.lang.ir_traversal import clone_ir, transform_ir, walk_ir

_ARITHMETIC = {'+', '-', '*', '/', '//', '%', '**'}
_CONSTANT_ONLY = (ir.IRConstant, ir.IRBinOp, ir.IRUnaryOp, ir.IRCompare, ir.IRBoolOp,
                  ir.IRCall, ir.IRIfExp)


def optimize_ir(ir_func: ir.IRFunction):
    """Canonicalize copies without moving or eliminating computations."""
    _inline_weak_literal_locals(ir_func)
    _copy_prop_function(ir_func)


def _literal(node) -> tuple[bool, bool]:
    """Whether ``node`` is a literal expression, and whether it contains a float literal.

    The type annotation's notion (``ir_type_annotate``), less a conditional
    expression whose condition reads a value: moved to where the local is
    read, it could read a different one. A condition built only from
    constants is fine.
    """
    if isinstance(node, ir.IRConstant):
        literal = type(node.value) in (int, float)
        return literal, literal and isinstance(node.value, float)
    if isinstance(node, ir.IRUnaryOp) and node.op in ('+', '-'):
        return _literal(node.operand)
    if isinstance(node, ir.IRBinOp) and node.op in _ARITHMETIC:
        parts = (_literal(node.left), _literal(node.right))
    elif isinstance(node, ir.IRCall):
        parts = tuple(_literal(arg) for arg in node.args)
        if not parts:
            return False, False
    elif isinstance(node, ir.IRIfExp):
        if not all(isinstance(n, _CONSTANT_ONLY) for n in walk_ir(node.condition)):
            return False, False
        parts = (_literal(node.then_value), _literal(node.else_value))
    else:
        return False, False
    return all(p[0] for p in parts), any(p[1] for p in parts)


def _inline_weak_literal_locals(ir_func: ir.IRFunction):
    """Replace each single-assignment local holding a weak literal expression by it.

    Each read gets its own copy, so each meets its own context; the
    assignment goes. Repeated until none is left, because a local copied
    from such a local (a device function's result, bound in turn by the
    caller) becomes one once its source is replaced.
    """
    while True:
        counts = _count_assignments(ir_func.body)
        values = {}
        for node in walk_ir(ir_func.body):
            if isinstance(node, ir.IRAssign) and counts.get(node.target) == 1:
                literal, weak = _literal(node.value)
                if literal and weak:
                    values[node.target] = node.value
        if not values:
            return
        ir_func.body = _replace_locals(ir_func.body, values)


def _replace_locals(body: list, values: dict) -> list:
    """``body`` with each read of a name in ``values`` replaced by its own copy, and the
    names' assignments removed."""
    def substitute(node):
        if isinstance(node, ir.IRName) and node.name in values:
            return clone_ir(values[node.name])
        return node

    def keep(stmts):
        kept = []
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target in values:
                continue
            for attr in ('body', 'then_body', 'else_body'):
                if isinstance(getattr(stmt, attr, None), list):
                    setattr(stmt, attr, keep(getattr(stmt, attr)))
            kept.append(stmt)
        return kept

    return keep(transform_ir(body, substitute, copy_nodes=True))


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
            for dim in getattr(node, 'dims', None) or ():
                counts[dim] = counts.get(dim, 0) + 1
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

    # A simple copy is x = y where x is assigned exactly once and y is
    # never assigned in the kernel (normally a parameter). That avoids
    # breaking tuple swaps or following a modified source.
    #
    # A copy only describes uses after its assignment. Precomputing the
    # replacement map for the whole block changes earlier reads of a
    # parameter that is later reassigned (or a loop-carried local).
    #
    # Each statement is tested after the copies before it are applied, so
    # a chain resolves to its root in one walk: with a = p, the later
    # b = a reads b = p and is itself a copy of p. Nested device functions
    # pass a field down as exactly such a chain, and a link left behind is
    # a local holding a field, which no backend can compile.
    resolved = {}
    result = []
    for stmt in body:
        stmt = _replace_names(stmt, resolved)
        result.append(stmt)
        if (isinstance(stmt, ir.IRAssign) and
                isinstance(stmt.value, ir.IRName) and
                assign_counts.get(stmt.target, 0) == 1 and
                assign_counts.get(stmt.value.name, 0) == 0):
            resolved[stmt.target] = stmt.value.name

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

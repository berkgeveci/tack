"""Conservative copy propagation for inlined kernel arguments.

Tack's former LICM treated an invariant address as an invariant value and
could execute a zero-trip loop's assignments. Its CSE invalidation tracked
field names rather than possibly overlapping storage. Neither pass has the
memory and control-flow analysis required to justify those transformations,
so hoisting and CSE are left to LLVM and the vendor compilers.

Copy propagation keeps assignments and replaces only subsequent uses of a
single-assignment copy whose source is not modified in its block.
"""

from tack.lang import ir


def optimize_ir(ir_func: ir.IRFunction):
    """Canonicalize copies without moving or eliminating computations."""
    _copy_prop_function(ir_func)


def _count_assignments(body: list) -> dict[str, int]:
    """Count how many times each variable is assigned in a statement list (recursive)."""
    counts: dict[str, int] = {}
    for stmt in body:
        if isinstance(stmt, ir.IRAssign):
            counts[stmt.target] = counts.get(stmt.target, 0) + 1
        elif isinstance(stmt, (ir.IRLocalAlloc, ir.IRSharedAlloc)):
            counts[stmt.name] = counts.get(stmt.name, 0) + 1
        elif isinstance(stmt, (ir.IRParallelFor, ir.IRSequentialFor)):
            counts[stmt.var] = counts.get(stmt.var, 0) + 1
            for k, v in _count_assignments(stmt.body).items():
                counts[k] = counts.get(k, 0) + v
        elif isinstance(stmt, ir.IRWhile):
            for k, v in _count_assignments(stmt.body).items():
                counts[k] = counts.get(k, 0) + v
        elif isinstance(stmt, ir.IRIf):
            for k, v in _count_assignments(stmt.then_body).items():
                counts[k] = counts.get(k, 0) + v
            if stmt.else_body:
                for k, v in _count_assignments(stmt.else_body).items():
                    counts[k] = counts.get(k, 0) + v
    return counts


def _copy_prop_function(ir_func: ir.IRFunction):
    """Apply copy propagation to the function body."""
    ir_func.body = _copy_prop_body(ir_func.body)


def _copy_prop_body(body: list) -> list:
    """Propagate copies: when x = y (simple name assignment), replace
    subsequent uses of x with y (unless x or y is reassigned later)."""
    # First, count assignments to find single-assignment variables
    assign_counts = _count_assignments(body)

    # Collect simple copies: x = y where x is assigned exactly once
    # AND y is never assigned in this block (it comes from outside — a
    # parameter or outer scope).  This avoids breaking tuple swaps where
    # the source is reassigned later in the same block.
    copies = {}  # target -> source name
    for stmt in body:
        if (isinstance(stmt, ir.IRAssign) and
                isinstance(stmt.value, ir.IRName) and
                assign_counts.get(stmt.target, 0) == 1 and
                assign_counts.get(stmt.value.name, 0) == 0):
            copies[stmt.target] = stmt.value.name

    if not copies:
        # Still recurse into sub-blocks
        return _copy_prop_recurse(body)

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

    return _copy_prop_recurse(result)


def _copy_prop_recurse(body: list) -> list:
    """Recurse copy propagation into loops and conditionals."""
    for stmt in body:
        if isinstance(stmt, (ir.IRParallelFor, ir.IRSequentialFor, ir.IRWhile)):
            stmt.body = _copy_prop_body(stmt.body)
        elif isinstance(stmt, ir.IRIf):
            stmt.then_body = _copy_prop_body(stmt.then_body)
            if stmt.else_body:
                stmt.else_body = _copy_prop_body(stmt.else_body)
    return body


def _replace_names(node, mapping: dict):
    """Replace variable names in an IR node according to the mapping."""
    if isinstance(node, ir.IRName):
        if node.name in mapping:
            return ir.IRName(mapping[node.name])
        return node

    if isinstance(node, ir.IRAssign):
        return ir.IRAssign(node.target, _replace_names(node.value, mapping))

    if isinstance(node, ir.IRFieldLoad):
        return ir.IRFieldLoad(
            _replace_names(node.field, mapping),
            _replace_names(node.index, mapping),
        )

    if isinstance(node, ir.IRFieldStore):
        return ir.IRFieldStore(
            _replace_names(node.field, mapping),
            _replace_names(node.index, mapping),
            _replace_names(node.value, mapping),
        )

    if isinstance(node, ir.IRAtomicOp):
        return ir.IRAtomicOp(
            node.op,
            _replace_names(node.field, mapping),
            _replace_names(node.index, mapping),
            _replace_names(node.value, mapping),
        )

    if isinstance(node, ir.IRBinOp):
        return ir.IRBinOp(
            node.op,
            _replace_names(node.left, mapping),
            _replace_names(node.right, mapping),
        )

    if isinstance(node, ir.IRUnaryOp):
        return ir.IRUnaryOp(node.op, _replace_names(node.operand, mapping))

    if isinstance(node, ir.IRCompare):
        return ir.IRCompare(
            node.op,
            _replace_names(node.left, mapping),
            _replace_names(node.right, mapping),
        )

    if isinstance(node, ir.IRBoolOp):
        return ir.IRBoolOp(
            node.op, [_replace_names(v, mapping) for v in node.values]
        )

    if isinstance(node, ir.IRCall):
        return ir.IRCall(
            node.func_name,
            [_replace_names(a, mapping) for a in node.args],
        )

    if isinstance(node, ir.IRCast):
        return ir.IRCast(_replace_names(node.value, mapping), node.dtype)

    if isinstance(node, ir.IRIfExp):
        return ir.IRIfExp(
            _replace_names(node.condition, mapping),
            _replace_names(node.then_value, mapping),
            _replace_names(node.else_value, mapping),
        )

    if isinstance(node, ir.IRIf):
        return ir.IRIf(
            _replace_names(node.condition, mapping),
            [_replace_names(s, mapping) for s in node.then_body],
            [_replace_names(s, mapping) for s in node.else_body] if node.else_body else [],
        )

    if isinstance(node, ir.IRParallelFor):
        return ir.IRParallelFor(
            node.var,
            _replace_names(node.start, mapping),
            _replace_names(node.end, mapping),
            [_replace_names(s, mapping) for s in node.body],
        )

    if isinstance(node, ir.IRSequentialFor):
        return ir.IRSequentialFor(
            node.var,
            _replace_names(node.start, mapping),
            _replace_names(node.end, mapping),
            [_replace_names(s, mapping) for s in node.body],
            step=_replace_names(node.step, mapping) if node.step is not None else None,
        )

    if isinstance(node, ir.IRWhile):
        return ir.IRWhile(
            _replace_names(node.condition, mapping),
            [_replace_names(s, mapping) for s in node.body],
        )

    if isinstance(node, ir.IRReturn):
        return ir.IRReturn(_replace_names(node.value, mapping))

    if isinstance(node, ir.IRAttribute):
        return ir.IRAttribute(_replace_names(node.obj, mapping), node.attr)

    # Constants, Break, Continue, DimSize — no names to replace
    return node

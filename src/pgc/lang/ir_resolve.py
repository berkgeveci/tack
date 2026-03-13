"""IR resolution pass — resolves dispatch-time constants in the IR.

Runs after AST transform, before codegen. Replaces:
- IRDimSize(field_name, dim) → IRConstant(shape[dim])
  Uses _logical_shape for vector fields, shape for regular fields.
"""

from pgc.lang import ir


def resolve_ir(ir_func: ir.IRFunction, name_to_field: dict):
    """Resolve dimension sizes in IR using actual field shapes.

    Mutates ir_func.body in place.
    """
    for i, stmt in enumerate(ir_func.body):
        ir_func.body[i] = _resolve(stmt, name_to_field)


def _resolve(node, fields):
    """Recursively resolve dispatch-time constants in an IR node."""
    if node is None:
        return None

    if isinstance(node, ir.IRDimSize):
        field = fields.get(node.field_name)
        if field is None:
            raise RuntimeError(f"Cannot resolve dimension size: unknown field '{node.field_name}'")
        shape = getattr(field, '_logical_shape', None) or field.shape
        return ir.IRConstant(shape[node.dim])

    if isinstance(node, ir.IRBinOp):
        node.left = _resolve(node.left, fields)
        node.right = _resolve(node.right, fields)
        return node

    if isinstance(node, ir.IRUnaryOp):
        node.operand = _resolve(node.operand, fields)
        return node

    if isinstance(node, ir.IRCompare):
        node.left = _resolve(node.left, fields)
        node.right = _resolve(node.right, fields)
        return node

    if isinstance(node, ir.IRBoolOp):
        node.values = [_resolve(v, fields) for v in node.values]
        return node

    if isinstance(node, ir.IRFieldLoad):
        node.field = _resolve(node.field, fields)
        node.index = _resolve(node.index, fields)
        return node

    if isinstance(node, ir.IRFieldStore):
        node.field = _resolve(node.field, fields)
        node.index = _resolve(node.index, fields)
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRAssign):
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRParallelFor):
        node.start = _resolve(node.start, fields)
        node.end = _resolve(node.end, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRSequentialFor):
        node.start = _resolve(node.start, fields)
        node.end = _resolve(node.end, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRWhile):
        node.condition = _resolve(node.condition, fields)
        node.body = [_resolve(s, fields) for s in node.body]
        return node

    if isinstance(node, ir.IRIf):
        node.condition = _resolve(node.condition, fields)
        node.then_body = [_resolve(s, fields) for s in node.then_body]
        node.else_body = [_resolve(s, fields) for s in node.else_body]
        return node

    if isinstance(node, ir.IRIfExp):
        node.condition = _resolve(node.condition, fields)
        node.then_value = _resolve(node.then_value, fields)
        node.else_value = _resolve(node.else_value, fields)
        return node

    if isinstance(node, ir.IRCall):
        node.args = [_resolve(a, fields) for a in node.args]
        return node

    if isinstance(node, ir.IRCast):
        node.value = _resolve(node.value, fields)
        return node

    if isinstance(node, ir.IRReturn):
        if node.value:
            node.value = _resolve(node.value, fields)
        return node

    # Leaf nodes: IRConstant, IRName, IRAttribute, IRBreak, IRContinue
    return node

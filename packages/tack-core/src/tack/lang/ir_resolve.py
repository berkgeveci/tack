"""IR resolution pass — resolves dispatch-time constants in the IR.

Runs after AST transform, before codegen. Replaces:
- IRDimSize(field_name, dim) → IRConstant(shape[dim])
  Uses _logical_shape for vector fields, shape for regular fields.
"""

from tack.lang import ir
from tack.lang.ir_traversal import transform_ir, walk_ir
from tack.lang.types import INTEGER_TYPES


def resolve_ir(ir_func: ir.IRFunction, name_to_field: dict):
    """Resolve dimension sizes in IR using actual field shapes.

    Mutates ir_func.body in place.
    """
    # Build field alias map: inlined @tack.func parameters create assignments
    # like __func_out_x0_0__ = out_x0, where out_x0 is a field. Track these
    # so shared_like/DimSize can resolve the mangled name to the actual field.
    aliases = _collect_field_aliases(ir_func.body, name_to_field)
    extended = {**name_to_field, **aliases}

    for i, stmt in enumerate(ir_func.body):
        if isinstance(stmt, ir.IRParallelFor):
            # Leave the grid bound unresolved. It never reaches generated
            # code — codegen reads the __loop_end__ parameter — so dispatch
            # evaluates it against the actual arguments instead. Folding it
            # to a literal would make the compiled kernel depend on the
            # array's length and force a recompile for every new size.
            stmt.start = _resolve(stmt.start, extended)
            stmt.body = [_resolve(s, extended) for s in stmt.body]
            ir_func.body[i] = stmt
        else:
            ir_func.body[i] = _resolve(stmt, extended)


_FOLDABLE = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "//": lambda a, b: a // b, "%": lambda a, b: a % b, "**": lambda a, b: a ** b,
    "<<": lambda a, b: a << b, ">>": lambda a, b: a >> b,
}


def fold_integer_constant(node):
    """The integer value of an expression made only of integer constants, else None.

    Casts to integer types are transparent, since a constant was checked to
    fit its type where it was declared; anything else (a name, a load, a
    float) leaves the expression as it is.
    """
    if isinstance(node, ir.IRConstant):
        return node.value if isinstance(node.value, int) and not isinstance(node.value, bool) else None
    if isinstance(node, ir.IRCast):
        return fold_integer_constant(node.value) if node.dtype in INTEGER_TYPES else None
    if isinstance(node, ir.IRUnaryOp) and node.op in ("-", "+"):
        inner = fold_integer_constant(node.operand)
        return None if inner is None else (-inner if node.op == "-" else inner)
    if isinstance(node, ir.IRBinOp) and node.op in _FOLDABLE:
        left, right = fold_integer_constant(node.left), fold_integer_constant(node.right)
        if left is None or right is None:
            return None
        if node.op in ("//", "%") and right == 0:
            return None
        if node.op in ("**", "<<", ">>") and right < 0:
            return None
        return _FOLDABLE[node.op](left, right)
    return None


def _collect_field_aliases(stmts, known_fields):
    """Collect inlined field aliases in structural statement order."""
    aliases = {}
    for node in walk_ir(stmts):
        if isinstance(node, ir.IRAssign) and isinstance(node.value, ir.IRName):
            src = node.value.name
            if src in known_fields:
                aliases[node.target] = known_fields[src]
            elif src in aliases:
                aliases[node.target] = aliases[src]
    return aliases


def _resolve(node, fields):
    """Resolve dispatch-time leaves using the shared structural traversal."""
    def resolve_leaf(node):
        if isinstance(node, ir.IRDimSize):
            field = fields.get(node.field_name)
            if field is None:
                raise RuntimeError(f"Cannot resolve dimension size: unknown field '{node.field_name}'")
            shape = getattr(field, '_logical_shape', None) or field.shape
            count = getattr(node, 'index_count', None)
            if count is not None and count != len(shape):
                # field[i, j] linearizes with the sizes of dimensions 1..,
                # so with too few indices it would address some other
                # element without any error.
                line, column, inlined_from = node.index_source
                where = f" at line {line}, column {column}" if line is not None else ""
                if inlined_from is not None:
                    where += f" of device function '{inlined_from}'"
                raise TypeError(
                    f"a field of {len(shape)} dimension{'s' if len(shape) != 1 else ''} "
                    f"{shape} is indexed with {count} indices{where}; give one index per "
                    f"dimension, or a single flat index")
            return ir.IRConstant(shape[node.dim])
        if isinstance(node, (ir.IRSharedAlloc, ir.IRLocalAlloc)):
            if node.dtype is None and node.field_name is not None:
                field = fields.get(node.field_name)
                if field is None:
                    kind = 'shared_like' if isinstance(node, ir.IRSharedAlloc) else 'local_array_like'
                    raise RuntimeError(f"Cannot resolve {kind}: unknown field '{node.field_name}'")
                node.dtype = field.dtype
            # A size written as arithmetic on constants (MAX_HITS * 4, a
            # resolved dimension + 2) is one constant: the GPU generators
            # need a constant expression for an array size, and this
            # pass is where every dimension in it has become a literal.
            folded = fold_integer_constant(node.size)
            if folded is not None:
                node.size = ir.IRConstant(folded)
        if isinstance(node, ir.IRTextureSample):
            field = fields.get(node.field_name)
            if field is not None:
                node.shape = getattr(field, 'shape_3d', None) or field.shape
        return node

    return transform_ir(node, resolve_leaf)

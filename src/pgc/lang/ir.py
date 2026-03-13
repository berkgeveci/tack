"""PGC internal IR — intermediate representation between Python AST and LLVM IR."""


class IRNode:
    """Base class for all IR nodes."""
    pass


class IRModule(IRNode):
    """Top-level module containing kernel functions."""

    def __init__(self):
        self.functions = []


class IRFunction(IRNode):
    """A kernel function."""

    def __init__(self, name: str, params: list, body: list):
        self.name = name
        self.params = params  # list of IRParam
        self.body = body      # list of IRNode statements


class IRParam(IRNode):
    """A function parameter."""

    def __init__(self, name: str, type_annotation=None):
        self.name = name
        self.type_annotation = type_annotation


# --- Loops ---

class IRParallelFor(IRNode):
    """A parallel for-loop over a range (top-level for in range())."""

    def __init__(self, var: str, start, end, body: list):
        self.var = var
        self.start = start
        self.end = end
        self.body = body


class IRSequentialFor(IRNode):
    """A sequential for-loop (nested for in range())."""

    def __init__(self, var: str, start, end, body: list):
        self.var = var
        self.start = start
        self.end = end
        self.body = body


class IRWhile(IRNode):
    """A while-loop."""

    def __init__(self, condition, body: list):
        self.condition = condition
        self.body = body


class IRBreak(IRNode):
    """Break statement."""
    pass


class IRContinue(IRNode):
    """Continue statement."""
    pass


# --- Control flow ---

class IRIf(IRNode):
    """If/else statement."""

    def __init__(self, condition, then_body: list, else_body: list):
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body


class IRIfExp(IRNode):
    """Ternary expression: value_if_true if condition else value_if_false."""

    def __init__(self, condition, then_value, else_value):
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value


# --- Expressions ---

class IRBinOp(IRNode):
    """Binary operation: left op right."""

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class IRUnaryOp(IRNode):
    """Unary operation: op operand."""

    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand


class IRCompare(IRNode):
    """Comparison: left op right (chained comparisons desugared to and-chains)."""

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right


class IRBoolOp(IRNode):
    """Boolean operation: and / or over a list of values."""

    def __init__(self, op: str, values: list):
        self.op = op  # "and" or "or"
        self.values = values


class IRFieldLoad(IRNode):
    """Load from a field: field[index]."""

    def __init__(self, field, index):
        self.field = field
        self.index = index


class IRFieldStore(IRNode):
    """Store to a field: field[index] = value."""

    def __init__(self, field, index, value):
        self.field = field
        self.index = index
        self.value = value


class IRConstant(IRNode):
    """A constant value (int or float literal)."""

    def __init__(self, value, dtype=None):
        self.value = value
        self.dtype = dtype


class IRName(IRNode):
    """A variable reference."""

    def __init__(self, name: str):
        self.name = name


class IRAttribute(IRNode):
    """Attribute access: obj.attr."""

    def __init__(self, obj, attr: str):
        self.obj = obj
        self.attr = attr


class IRCall(IRNode):
    """Function/builtin call: func(args...)."""

    def __init__(self, func_name: str, args: list):
        self.func_name = func_name
        self.args = args


class IRAssign(IRNode):
    """Variable assignment: target = value."""

    def __init__(self, target: str, value):
        self.target = target
        self.value = value


class IRReturn(IRNode):
    """Return statement."""

    def __init__(self, value):
        self.value = value


class IRCast(IRNode):
    """Type cast: cast value to dtype."""

    def __init__(self, value, dtype):
        self.value = value
        self.dtype = dtype


class IRDimSize(IRNode):
    """Query the size of a specific dimension of a field parameter.

    Resolved at dispatch time to an IRConstant when field shapes are known.
    Used for multi-dimensional index linearization.
    """

    def __init__(self, field_name: str, dim: int):
        self.field_name = field_name
        self.dim = dim


# --- IR pretty printer (for debugging) ---

def dump(node, indent=0) -> str:
    """Pretty-print an IR tree for debugging."""
    prefix = "  " * indent
    if isinstance(node, IRModule):
        lines = [f"{prefix}Module:"]
        for fn in node.functions:
            lines.append(dump(fn, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRFunction):
        params = ", ".join(p.name for p in node.params)
        lines = [f"{prefix}Function {node.name}({params}):"]
        for stmt in node.body:
            lines.append(dump(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRParallelFor):
        lines = [f"{prefix}ParallelFor {node.var} in [{dump(node.start)}, {dump(node.end)}):"]
        for stmt in node.body:
            lines.append(dump(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRSequentialFor):
        lines = [f"{prefix}SequentialFor {node.var} in [{dump(node.start)}, {dump(node.end)}):"]
        for stmt in node.body:
            lines.append(dump(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRWhile):
        lines = [f"{prefix}While {dump(node.condition)}:"]
        for stmt in node.body:
            lines.append(dump(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRBreak):
        return f"{prefix}Break"
    if isinstance(node, IRContinue):
        return f"{prefix}Continue"
    if isinstance(node, IRIf):
        lines = [f"{prefix}If {dump(node.condition)}:"]
        for stmt in node.then_body:
            lines.append(dump(stmt, indent + 1))
        if node.else_body:
            lines.append(f"{prefix}Else:")
            for stmt in node.else_body:
                lines.append(dump(stmt, indent + 1))
        return "\n".join(lines)
    if isinstance(node, IRIfExp):
        return f"{prefix}IfExp({dump(node.condition)}, {dump(node.then_value)}, {dump(node.else_value)})"
    if isinstance(node, IRBinOp):
        return f"{prefix}({dump(node.left)} {node.op} {dump(node.right)})"
    if isinstance(node, IRUnaryOp):
        return f"{prefix}({node.op}{dump(node.operand)})"
    if isinstance(node, IRCompare):
        return f"{prefix}({dump(node.left)} {node.op} {dump(node.right)})"
    if isinstance(node, IRBoolOp):
        parts = f" {node.op} ".join(dump(v) for v in node.values)
        return f"{prefix}({parts})"
    if isinstance(node, IRFieldLoad):
        return f"{prefix}{dump(node.field)}[{dump(node.index)}]"
    if isinstance(node, IRFieldStore):
        return f"{prefix}{dump(node.field)}[{dump(node.index)}] = {dump(node.value)}"
    if isinstance(node, IRConstant):
        return f"{prefix}{node.value!r}"
    if isinstance(node, IRName):
        return f"{prefix}{node.name}"
    if isinstance(node, IRAttribute):
        return f"{prefix}{dump(node.obj)}.{node.attr}"
    if isinstance(node, IRCall):
        args = ", ".join(dump(a) for a in node.args)
        return f"{prefix}{node.func_name}({args})"
    if isinstance(node, IRAssign):
        return f"{prefix}{node.target} = {dump(node.value)}"
    if isinstance(node, IRReturn):
        return f"{prefix}Return {dump(node.value)}"
    if isinstance(node, IRCast):
        return f"{prefix}Cast({dump(node.value)}, {node.dtype})"
    if isinstance(node, IRDimSize):
        return f"{prefix}DimSize({node.field_name}, {node.dim})"
    return f"{prefix}<unknown: {type(node).__name__}>"

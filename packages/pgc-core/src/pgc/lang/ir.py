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

    def __init__(self, var: str, start, end, body: list, step=None):
        self.var = var
        self.start = start
        self.end = end
        self.body = body
        self.step = step  # None means step=1


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
        self.dtype = None


# --- Expressions ---

class IRBinOp(IRNode):
    """Binary operation: left op right."""

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.dtype = None


class IRUnaryOp(IRNode):
    """Unary operation: op operand."""

    def __init__(self, op: str, operand):
        self.op = op
        self.operand = operand
        self.dtype = None


class IRCompare(IRNode):
    """Comparison: left op right (chained comparisons desugared to and-chains)."""

    def __init__(self, op: str, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.dtype = None


class IRBoolOp(IRNode):
    """Boolean operation: and / or over a list of values."""

    def __init__(self, op: str, values: list):
        self.op = op  # "and" or "or"
        self.values = values
        self.dtype = None


class IRFieldLoad(IRNode):
    """Load from a field: field[index]."""

    def __init__(self, field, index):
        self.field = field
        self.index = index
        self.dtype = None


class IRFieldStore(IRNode):
    """Store to a field: field[index] = value."""

    def __init__(self, field, index, value):
        self.field = field
        self.index = index
        self.value = value


class IRAtomicOp(IRNode):
    """Atomic operation on a field: atomic_add(field, index, value), etc."""

    def __init__(self, op: str, field, index, value):
        self.op = op  # "add", "min", "max"
        self.field = field
        self.index = index
        self.value = value
        self.dtype = None


class IRConstant(IRNode):
    """A constant value (int or float literal)."""

    def __init__(self, value, dtype=None):
        self.value = value
        self.dtype = dtype


class IRName(IRNode):
    """A variable reference."""

    def __init__(self, name: str):
        self.name = name
        self.dtype = None


class IRAttribute(IRNode):
    """Attribute access: obj.attr."""

    def __init__(self, obj, attr: str):
        self.obj = obj
        self.attr = attr
        self.dtype = None


class IRCall(IRNode):
    """Function/builtin call: func(args...)."""

    def __init__(self, func_name: str, args: list):
        self.func_name = func_name
        self.args = args
        self.dtype = None


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
        self.dtype = dtype  # currently "int"/"float" strings, will become ScalarType


class IRSharedAlloc(IRNode):
    """Allocate shared/threadgroup memory."""

    def __init__(self, name: str, dtype: str, size, field_name: str = None):
        self.name = name
        self.dtype = dtype         # "float", "int", etc. (None for shared_like, resolved later)
        self.size = size           # IRNode expression for number of elements
        self.field_name = field_name  # source field name for shared_like (resolved in ir_resolve)


class IRLocalAlloc(IRNode):
    """Allocate a per-thread local array (private memory on GPU, stack on CPU)."""

    def __init__(self, name: str, dtype: str, size, field_name: str = None):
        self.name = name
        self.dtype = dtype   # "float", "int", etc. (None for local_array_like, resolved later)
        self.size = size     # IRNode expression for number of elements
        self.field_name = field_name  # source field name for local_array_like (resolved in ir_resolve)


class IRBlockReduce(IRNode):
    """Block-level reduction across all threads in a workgroup.

    Emits a shared memory tree reduction pattern:
    shared[tid] = value; barrier; tree reduce; barrier; result = shared[0].
    """

    def __init__(self, op: str, value):
        self.op = op      # "sum", "min", "max"
        self.value = value # IRNode expression
        self.dtype = None


class IRBarrier(IRNode):
    """Threadgroup synchronization barrier."""
    pass


class IRThreadId(IRNode):
    """Thread index within workgroup (threadIdx.x / thread_position_in_threadgroup)."""

    def __init__(self):
        self.dtype = None


class IRPrint(IRNode):
    """Print statement for kernel debugging."""

    def __init__(self, args: list, format_parts: list = None):
        self.args = args            # list of IRNode expressions
        self.format_parts = format_parts  # list of (kind, value): kind is "str" or "expr"


class IRDimSize(IRNode):
    """Query the size of a specific dimension of a field parameter.

    Resolved at dispatch time to an IRConstant when field shapes are known.
    Used for multi-dimensional index linearization.
    """

    def __init__(self, field_name: str, dim: int):
        self.field_name = field_name
        self.dim = dim


class IRTextureSample(IRNode):
    """Sample a 3D texture at normalized coordinates (u, v, w) in [0, 1].

    field_name: name of the texture's underlying field parameter.
    coords: list of 3 IRNode expressions [u, v, w].
    shape: (W, H, D) tuple — filled in by ir_resolve from the field's shape.
    """

    def __init__(self, field_name: str, coords: list, shape: tuple = None):
        self.field_name = field_name
        self.coords = coords  # [u, v, w]
        self.shape = shape    # (W, H, D) — set during resolve
        self.dtype = None


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
        step_str = f" step {dump(node.step)}" if node.step else ""
        lines = [f"{prefix}SequentialFor {node.var} in [{dump(node.start)}, {dump(node.end)}){step_str}:"]
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
    if isinstance(node, IRAtomicOp):
        return f"{prefix}atomic_{node.op}({dump(node.field)}[{dump(node.index)}], {dump(node.value)})"
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
        dtype_str = node.dtype.name if hasattr(node.dtype, 'name') else node.dtype
        return f"{prefix}Cast({dump(node.value)}, {dtype_str})"
    if isinstance(node, IRSharedAlloc):
        return f"{prefix}SharedAlloc {node.name}: {node.dtype}[{dump(node.size)}]"
    if isinstance(node, IRLocalAlloc):
        return f"{prefix}LocalAlloc {node.name}: {node.dtype}[{dump(node.size)}]"
    if isinstance(node, IRBlockReduce):
        return f"{prefix}BlockReduce({node.op}, {dump(node.value)})"
    if isinstance(node, IRBarrier):
        return f"{prefix}Barrier"
    if isinstance(node, IRThreadId):
        return f"{prefix}ThreadId"
    if isinstance(node, IRPrint):
        args = ", ".join(dump(a) for a in node.args)
        return f"{prefix}Print({args})"
    if isinstance(node, IRDimSize):
        return f"{prefix}DimSize({node.field_name}, {node.dim})"
    if isinstance(node, IRTextureSample):
        coords = ", ".join(dump(c) for c in node.coords)
        return f"{prefix}TexSample({node.field_name}, [{coords}], shape={node.shape})"
    return f"{prefix}<unknown: {type(node).__name__}>"

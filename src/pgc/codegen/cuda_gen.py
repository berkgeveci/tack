"""PGC CUDA C code generation — transforms PGC IR to CUDA C source for NVRTC.

Generates an ``extern "C" __global__`` kernel function where:
  - Each Field parameter becomes a typed device pointer (``float*``, etc.)
  - The outermost parallel for-loop maps to the standard CUDA thread index:
        int __idx__ = blockIdx.x * blockDim.x + threadIdx.x;
    with a bounds guard.
  - Sequential for-loops, while-loops, if/else map to standard C control flow.
  - Math builtins map to CUDA device math functions (sqrtf, sinf, etc.).
"""

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64


_C_TYPE_MAP = {
    f32: "float",
    f64: "double",
    i32: "int",
    i64: "long long",
    u32: "unsigned int",
    u64: "unsigned long long",
}

_MATH_FUNCS_F32 = {
    "sqrt": "sqrtf",
    "sin": "sinf",
    "cos": "cosf",
    "tan": "tanf",
    "asin": "asinf",
    "acos": "acosf",
    "atan": "atanf",
    "atan2": "atan2f",
    "exp": "expf",
    "exp2": "exp2f",
    "log": "logf",
    "log2": "log2f",
    "log10": "log10f",
    "floor": "floorf",
    "ceil": "ceilf",
    "fabs": "fabsf",
    "abs": "fabsf",
    "pow": "powf",
}

_MATH_FUNCS_F64 = {
    "sqrt": "sqrt",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "exp2": "exp2",
    "log": "log",
    "log2": "log2",
    "log10": "log10",
    "floor": "floor",
    "ceil": "ceil",
    "fabs": "fabs",
    "abs": "fabs",
    "pow": "pow",
}

_BINOP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "<<": "<<", ">>": ">>", "&": "&", "|": "|", "^": "^",
}

_CMP_MAP = {
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
}


class CUDACodeGen:
    """Generates CUDA C source from a PGC IR function."""

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = ir_func
        self._indent = 0
        self._lines: list[str] = []
        self._param_types: dict[str, ScalarType] = {}
        self._field_params: set[str] = set()
        self._local_vars: dict[str, str] = {}  # name → C type
        self._loop_end_name: str | None = None

    def generate(self) -> str:
        """Generate CUDA C source for the kernel."""
        func = self.ir_func

        # Build parameter info
        for param in func.params:
            if param.type_annotation is None:
                raise TypeError(f"Parameter '{param.name}' has no type. Run type inference first.")
            self._param_types[param.name] = param.type_annotation
            if hasattr(param, '_is_field') and param._is_field:
                self._field_params.add(param.name)
            else:
                self._field_params.add(param.name)

        # Build function signature
        params_c = []
        for param in func.params:
            c_type = _C_TYPE_MAP[param.type_annotation]
            if param.name in self._field_params:
                params_c.append(f"{c_type}* __restrict__ {param.name}")
            else:
                params_c.append(f"{c_type} {param.name}")

        # Add the loop-end parameter (passed as kernel arg for bounds checking)
        params_c.append("long long __n__")

        sig = ", ".join(params_c)
        self._emit(f'extern "C" __global__ void {func.name}({sig}) {{')
        self._indent += 1

        self._emit_body(func.body)

        self._indent -= 1
        self._emit("}")

        return "\n".join(self._lines) + "\n"

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _emit_body(self, stmts: list):
        for stmt in stmts:
            self._emit_stmt(stmt)

    def _emit_stmt(self, node):
        if isinstance(node, ir.IRParallelFor):
            self._emit_parallel_for(node)
        elif isinstance(node, ir.IRSequentialFor):
            self._emit_sequential_for(node)
        elif isinstance(node, ir.IRWhile):
            self._emit_while(node)
        elif isinstance(node, ir.IRIf):
            self._emit_if(node)
        elif isinstance(node, ir.IRFieldStore):
            self._emit_field_store(node)
        elif isinstance(node, ir.IRAssign):
            self._emit_assign(node)
        elif isinstance(node, ir.IRReturn):
            self._emit("return;")
        elif isinstance(node, ir.IRBreak):
            self._emit("break;")
        elif isinstance(node, ir.IRContinue):
            self._emit("continue;")
        elif isinstance(node, ir.IRCall):
            self._emit(f"{self._expr(node)};")
        else:
            raise NotImplementedError(f"CUDA codegen: cannot emit {type(node).__name__}")

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        """Emit the parallel for-loop as CUDA thread index calculation."""
        idx = node.var
        self._emit(f"long long {idx} = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;")
        self._emit(f"if ({idx} >= __n__) return;")
        self._local_vars[idx] = "long long"
        self._emit_body(node.body)

    def _emit_sequential_for(self, node: ir.IRSequentialFor):
        start = self._expr(node.start)
        end = self._expr(node.end)
        var = node.var
        if var not in self._local_vars:
            self._emit(f"for (long long {var} = {start}; {var} < {end}; {var}++) {{")
            self._local_vars[var] = "long long"
        else:
            self._emit(f"for ({var} = {start}; {var} < {end}; {var}++) {{")
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")

    def _emit_while(self, node: ir.IRWhile):
        cond = self._expr(node.condition)
        self._emit(f"while ({cond}) {{")
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")

    def _emit_if(self, node: ir.IRIf):
        cond = self._expr(node.condition)
        self._emit(f"if ({cond}) {{")
        self._indent += 1
        self._emit_body(node.then_body)
        self._indent -= 1
        if node.else_body:
            self._emit("} else {")
            self._indent += 1
            self._emit_body(node.else_body)
            self._indent -= 1
        self._emit("}")

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        self._emit(f"{field}[{index}] = {value};")

    def _emit_assign(self, node: ir.IRAssign):
        value = self._expr(node.value)
        if node.target in self._local_vars:
            self._emit(f"{node.target} = {value};")
        else:
            # Infer a C type from the expression
            c_type = self._infer_c_type(node.value)
            self._emit(f"{c_type} {node.target} = {value};")
            self._local_vars[node.target] = c_type

    def _infer_c_type(self, node) -> str:
        """Best-effort C type inference for local variable declarations."""
        if isinstance(node, ir.IRConstant):
            if isinstance(node.value, float):
                return "float"
            return "long long"
        if isinstance(node, ir.IRFieldLoad):
            field_name = self._get_field_name(node.field)
            if field_name and field_name in self._param_types:
                return _C_TYPE_MAP[self._param_types[field_name]]
        if isinstance(node, ir.IRBinOp):
            # If either operand involves a float, result is float
            lt = self._infer_c_type(node.left)
            rt = self._infer_c_type(node.right)
            if lt == "float" or rt == "float":
                return "float"
            if lt == "double" or rt == "double":
                return "double"
            return lt
        if isinstance(node, ir.IRCall):
            return "float"
        if isinstance(node, ir.IRCast):
            if node.dtype == "int":
                return "long long"
            if node.dtype == "float":
                return "float"
        if isinstance(node, ir.IRIfExp):
            return self._infer_c_type(node.then_value)
        if isinstance(node, ir.IRCompare):
            return "int"
        if isinstance(node, ir.IRUnaryOp):
            return self._infer_c_type(node.operand)
        if isinstance(node, ir.IRName):
            if node.name in self._param_types:
                return _C_TYPE_MAP[self._param_types[node.name]]
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            return "long long"
        return "float"

    def _get_field_name(self, node) -> str | None:
        if isinstance(node, ir.IRName):
            return node.name
        return None

    # --- Expression codegen ---

    def _expr(self, node) -> str:
        if isinstance(node, ir.IRConstant):
            return self._expr_constant(node)
        if isinstance(node, ir.IRName):
            return node.name
        if isinstance(node, ir.IRBinOp):
            return self._expr_binop(node)
        if isinstance(node, ir.IRUnaryOp):
            return self._expr_unaryop(node)
        if isinstance(node, ir.IRCompare):
            return self._expr_compare(node)
        if isinstance(node, ir.IRBoolOp):
            return self._expr_boolop(node)
        if isinstance(node, ir.IRFieldLoad):
            return self._expr_field_load(node)
        if isinstance(node, ir.IRAttribute):
            return self._expr_attribute(node)
        if isinstance(node, ir.IRCall):
            return self._expr_call(node)
        if isinstance(node, ir.IRCast):
            return self._expr_cast(node)
        if isinstance(node, ir.IRIfExp):
            return self._expr_ifexp(node)
        raise NotImplementedError(f"CUDA expr: {type(node).__name__}")

    def _expr_constant(self, node: ir.IRConstant) -> str:
        if isinstance(node.value, float):
            # Use f suffix for float literals
            return f"{node.value!r}f"
        if isinstance(node.value, bool):
            return "1" if node.value else "0"
        return str(node.value)

    def _expr_binop(self, node: ir.IRBinOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        if node.op == "**":
            return f"powf({left}, {right})"
        if node.op == "//":
            return f"(long long)floorf((float)({left}) / (float)({right}))"
        if node.op in _BINOP_MAP:
            return f"({left} {_BINOP_MAP[node.op]} {right})"
        raise NotImplementedError(f"CUDA binop: {node.op}")

    def _expr_unaryop(self, node: ir.IRUnaryOp) -> str:
        operand = self._expr(node.operand)
        if node.op == "-":
            return f"(-{operand})"
        if node.op == "+":
            return operand
        if node.op == "not":
            return f"(!{operand})"
        if node.op == "~":
            return f"(~{operand})"
        raise NotImplementedError(f"CUDA unaryop: {node.op}")

    def _expr_compare(self, node: ir.IRCompare) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        return f"({left} {_CMP_MAP[node.op]} {right})"

    def _expr_boolop(self, node: ir.IRBoolOp) -> str:
        c_op = "&&" if node.op == "and" else "||"
        parts = [self._expr(v) for v in node.values]
        return "(" + f" {c_op} ".join(parts) + ")"

    def _expr_field_load(self, node: ir.IRFieldLoad) -> str:
        field = self._expr(node.field)
        index = self._expr(node.index)
        return f"{field}[{index}]"

    def _expr_attribute(self, node: ir.IRAttribute) -> str:
        # field.shape[0] should have been resolved before codegen
        raise NotImplementedError(
            f"Attribute access '{node.attr}' should be resolved before CUDA codegen."
        )

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]

        # min/max
        if node.func_name == "min" and len(args) == 2:
            return f"fminf({args[0]}, {args[1]})"
        if node.func_name == "max" and len(args) == 2:
            return f"fmaxf({args[0]}, {args[1]})"

        # Standard math builtins (default to f32 versions)
        if node.func_name in _MATH_FUNCS_F32:
            func = _MATH_FUNCS_F32[node.func_name]
            return f"{func}({', '.join(args)})"

        raise NotImplementedError(f"CUDA builtin: {node.func_name}")

    def _expr_cast(self, node: ir.IRCast) -> str:
        val = self._expr(node.value)
        if node.dtype == "int":
            return f"((long long)({val}))"
        if node.dtype == "float":
            return f"((float)({val}))"
        raise NotImplementedError(f"CUDA cast: {node.dtype}")

    def _expr_ifexp(self, node: ir.IRIfExp) -> str:
        cond = self._expr(node.condition)
        then = self._expr(node.then_value)
        else_ = self._expr(node.else_value)
        return f"({cond} ? {then} : {else_})"


def generate_cuda_source(ir_func: ir.IRFunction) -> str:
    """Generate CUDA C source for a single kernel function."""
    codegen = CUDACodeGen(ir_func)
    return codegen.generate()

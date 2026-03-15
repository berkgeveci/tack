"""PGC CUDA C code generation — transforms PGC IR to CUDA C source for NVRTC.

Generates an ``extern "C" __global__`` kernel function where:
  - Each Field parameter becomes a typed device pointer (``float*``, etc.)
  - The outermost parallel for-loop maps to the standard CUDA thread index:
        int __idx__ = blockIdx.x * blockDim.x + threadIdx.x;
    with a bounds guard.
  - Sequential for-loops, while-loops, if/else map to standard C control flow.
  - Math builtins map to CUDA device math functions (sqrtf, sinf, etc.).

All integer locals and loop indices use 32-bit ``int`` to maximise GPU ALU
throughput (64-bit integer math runs at half rate on consumer NVIDIA GPUs).
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

# Default integer type for locals on CUDA — 32-bit for throughput.
_INT = "int"

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
        self._local_vars: dict[str, str] = {}  # name → C type (all known vars)
        self._declared_vars: set[str] = set()  # vars already emitted with declaration
        self._loop_end_name: str | None = None
        self._needs_float_atomic_min = False
        self._needs_float_atomic_max = False

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
            elif not hasattr(param, '_is_field'):
                # Default: treat as field for backwards compatibility
                self._field_params.add(param.name)

        # Build function signature
        params_c = []
        for param in func.params:
            c_type = _C_TYPE_MAP[param.type_annotation]
            if param.name in self._field_params:
                params_c.append(f"{c_type}* __restrict__ {param.name}")
            else:
                params_c.append(f"{c_type} {param.name}")

        # Loop-end parameter — 32-bit is sufficient for CUDA grid sizes
        params_c.append(f"{_INT} __n__")

        sig = ", ".join(params_c)
        self._emit(f'extern "C" __global__ void {func.name}({sig}) {{')
        self._indent += 1

        self._emit_body(func.body)

        self._indent -= 1
        self._emit("}")

        prefix_lines = []
        if self._needs_float_atomic_min:
            prefix_lines.extend([
                "__device__ float atomicMinFloat(float* addr, float val) {",
                "    int* addr_as_int = (int*)addr;",
                "    int old = *addr_as_int, assumed;",
                "    do {",
                "        assumed = old;",
                "        old = atomicCAS(addr_as_int, assumed,",
                "            __float_as_int(fminf(val, __int_as_float(assumed))));",
                "    } while (assumed != old);",
                "    return __int_as_float(old);",
                "}",
                "",
            ])
        if self._needs_float_atomic_max:
            prefix_lines.extend([
                "__device__ float atomicMaxFloat(float* addr, float val) {",
                "    int* addr_as_int = (int*)addr;",
                "    int old = *addr_as_int, assumed;",
                "    do {",
                "        assumed = old;",
                "        old = atomicCAS(addr_as_int, assumed,",
                "            __float_as_int(fmaxf(val, __int_as_float(assumed))));",
                "    } while (assumed != old);",
                "    return __int_as_float(old);",
                "}",
                "",
            ])

        if prefix_lines:
            return "\n".join(prefix_lines) + "\n".join(self._lines) + "\n"
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
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRPrint):
            self._emit_print(node)
        elif isinstance(node, ir.IRSharedAlloc):
            self._emit(f"__shared__ {node.dtype} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRBarrier):
            self._emit("__syncthreads();")
        elif isinstance(node, ir.IRCall):
            self._emit(f"{self._expr(node)};")
        else:
            raise NotImplementedError(f"CUDA codegen: cannot emit {type(node).__name__}")

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        """Emit the parallel for-loop as CUDA thread index calculation."""
        idx = node.var
        self._emit(f"{_INT} {idx} = blockIdx.x * blockDim.x + threadIdx.x;")
        self._emit(f"if ({idx} >= __n__) return;")
        self._local_vars[idx] = _INT
        self._declared_vars.add(idx)
        self._emit_body(node.body)

    def _emit_sequential_for(self, node: ir.IRSequentialFor):
        start = self._expr(node.start)
        end = self._expr(node.end)
        step = self._expr(node.step) if node.step else None
        incr = f"{node.var} += {step}" if step else f"{node.var}++"
        var = node.var
        if var not in self._declared_vars:
            self._emit(f"for ({_INT} {var} = {start}; {var} < {end}; {incr}) {{")
            self._local_vars[var] = _INT
            self._declared_vars.add(var)
        else:
            self._emit(f"for ({var} = {start}; {var} < {end}; {incr}) {{")
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
        # Pre-declare variables assigned in branches so they are visible at
        # the outer scope in C.
        then_new = self._collect_new_assigns(node.then_body)
        else_new = self._collect_new_assigns(node.else_body) if node.else_body else set()
        needs_hoist = then_new | else_new
        for var_name in sorted(needs_hoist):
            if var_name not in self._declared_vars:
                c_type = self._find_assign_type(var_name, node.then_body) or \
                         self._find_assign_type(var_name, node.else_body or []) or "float"
                self._emit(f"{c_type} {var_name};")
                self._local_vars[var_name] = c_type
                self._declared_vars.add(var_name)

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

    def _collect_new_assigns(self, stmts: list) -> set[str]:
        """Collect variable names that would be newly declared in these stmts."""
        result = set()
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target not in self._declared_vars:
                result.add(stmt.target)
            elif isinstance(stmt, ir.IRIf):
                result |= self._collect_new_assigns(stmt.then_body)
                if stmt.else_body:
                    result |= self._collect_new_assigns(stmt.else_body)
        return result

    def _find_assign_type(self, var_name: str, stmts: list) -> str | None:
        """Find the inferred C type for a variable from its assignment in stmts."""
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target == var_name:
                return self._infer_c_type(stmt.value)
            elif isinstance(stmt, ir.IRIf):
                t = self._find_assign_type(var_name, stmt.then_body)
                if t:
                    return t
                if stmt.else_body:
                    t = self._find_assign_type(var_name, stmt.else_body)
                    if t:
                        return t
        return None

    def _emit_print(self, node: ir.IRPrint):
        """Emit a printf call for kernel debugging."""
        fmt_parts = []
        args = []
        if node.format_parts:
            for i, (kind, val) in enumerate(node.format_parts):
                if i > 0:
                    fmt_parts.append(" ")
                if kind == "str":
                    fmt_parts.append(val.replace("%", "%%"))
                else:
                    expr = self._expr(node.args[val])
                    expr_type = self._infer_expr_type(node.args[val])
                    if expr_type in ("float", "double"):
                        fmt_parts.append("%f")
                    else:
                        fmt_parts.append("%d")
                    args.append(expr)
        else:
            for i, arg_node in enumerate(node.args):
                if i > 0:
                    fmt_parts.append(" ")
                expr = self._expr(arg_node)
                expr_type = self._infer_expr_type(arg_node)
                if expr_type in ("float", "double"):
                    fmt_parts.append("%f")
                else:
                    fmt_parts.append("%d")
                args.append(expr)

        fmt_str = "".join(fmt_parts) + "\\n"
        args_str = ", ".join(args)
        if args_str:
            self._emit(f'printf("{fmt_str}", {args_str});')
        else:
            self._emit(f'printf("{fmt_str}");')

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        """Emit a CUDA atomic operation."""
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_INT})({index}))"

        # Determine field type to handle float atomicMin/Max via CAS
        field_name = self._get_field_name(node.field)
        field_type = _C_TYPE_MAP.get(self._param_types.get(field_name)) if field_name else None
        is_float = field_type in ("float", "double")

        if node.op == "min" and is_float:
            self._needs_float_atomic_min = True
            self._emit(f"atomicMinFloat(&{field}[{index}], {value});")
        elif node.op == "max" and is_float:
            self._needs_float_atomic_max = True
            self._emit(f"atomicMaxFloat(&{field}[{index}], {value});")
        else:
            _ATOMIC_FUNCS = {"add": "atomicAdd", "min": "atomicMin", "max": "atomicMax"}
            func = _ATOMIC_FUNCS.get(node.op)
            if func is None:
                raise NotImplementedError(f"CUDA atomic op: {node.op}")
            self._emit(f"{func}(&{field}[{index}], {value});")

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        # Ensure array index is integer
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_INT})({index}))"
        self._emit(f"{field}[{index}] = {value};")

    def _emit_assign(self, node: ir.IRAssign):
        value = self._expr(node.value)
        if node.target in self._declared_vars:
            self._emit(f"{node.target} = {value};")
        else:
            # Infer a C type from the expression
            c_type = self._infer_c_type(node.value)
            self._emit(f"{c_type} {node.target} = {value};")
            self._local_vars[node.target] = c_type
            self._declared_vars.add(node.target)

    def _infer_c_type(self, node) -> str:
        """Best-effort C type inference for local variable declarations."""
        if isinstance(node, ir.IRConstant):
            if isinstance(node.value, float):
                return "float"
            return _INT
        if isinstance(node, ir.IRFieldLoad):
            field_name = self._get_field_name(node.field)
            if field_name and field_name in self._param_types:
                return _C_TYPE_MAP[self._param_types[field_name]]
        if isinstance(node, ir.IRBinOp):
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
                return _INT
            if node.dtype == "float":
                return "float"
        if isinstance(node, ir.IRIfExp):
            return self._infer_c_type(node.then_value)
        if isinstance(node, ir.IRCompare):
            return _INT
        if isinstance(node, ir.IRUnaryOp):
            return self._infer_c_type(node.operand)
        if isinstance(node, ir.IRName):
            if node.name in self._field_params:
                c_type = _C_TYPE_MAP[self._param_types[node.name]]
                return f"{c_type}*"
            if node.name in self._param_types:
                return _C_TYPE_MAP[self._param_types[node.name]]
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            return _INT
        return "float"

    def _infer_expr_type(self, node) -> str:
        """Infer the runtime C type of an expression, considering variable reassignments."""
        if isinstance(node, ir.IRName):
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            if node.name in self._param_types:
                return _C_TYPE_MAP[self._param_types[node.name]]
            return _INT
        if isinstance(node, ir.IRBinOp):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if "float" in (lt, rt):
                return "float"
            if "double" in (lt, rt):
                return "double"
            return lt
        if isinstance(node, ir.IRCall):
            return "float"
        if isinstance(node, ir.IRCast):
            if node.dtype == "int":
                return _INT
            return "float"
        return self._infer_c_type(node)

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
        if isinstance(node, ir.IRThreadId):
            return "threadIdx.x"
        raise NotImplementedError(f"CUDA expr: {type(node).__name__}")

    def _expr_constant(self, node: ir.IRConstant) -> str:
        if isinstance(node.value, float):
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
            # Use true integer division when both operands are integer types,
            # matching LLVM sdiv semantics. Fall back to float floor for floats.
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt not in ("float", "double") and rt not in ("float", "double"):
                return f"({left} / {right})"
            return f"({_INT})floorf((float)({left}) / (float)({right}))"
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
        # Ensure array index is integer — float indices are invalid in CUDA
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_INT})({index}))"
        return f"{field}[{index}]"

    def _expr_attribute(self, node: ir.IRAttribute) -> str:
        raise NotImplementedError(
            f"Attribute access '{node.attr}' should be resolved before CUDA codegen."
        )

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]

        if node.func_name == "min" and len(args) == 2:
            return f"fminf({args[0]}, {args[1]})"
        if node.func_name == "max" and len(args) == 2:
            return f"fmaxf({args[0]}, {args[1]})"

        if node.func_name in _MATH_FUNCS_F32:
            func = _MATH_FUNCS_F32[node.func_name]
            return f"{func}({', '.join(args)})"

        raise NotImplementedError(f"CUDA builtin: {node.func_name}")

    def _expr_cast(self, node: ir.IRCast) -> str:
        val = self._expr(node.value)
        if node.dtype == "int":
            return f"(({_INT})({val}))"
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

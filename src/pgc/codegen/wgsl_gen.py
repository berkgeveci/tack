"""PGC WGSL code generation — transforms PGC IR to WebGPU Shading Language.

Generates a WGSL compute shader where:
  - Each field parameter becomes a storage buffer binding
  - The outermost parallel for-loop maps to global_invocation_id.x
  - Sequential loops, if/else, while map to standard WGSL control flow
  - Math builtins map to WGSL built-in functions

The loop-end value is passed as the first element of a uniform buffer
(pgc_params) to avoid needing a separate binding per scalar.
"""

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64


_WGSL_TYPE_MAP = {
    f32: "f32",
    i32: "i32",
    i64: "i32",  # WGSL has no i64; use i32
    u32: "u32",
    u64: "u32",
}

_WGSL_MATH = {
    "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "atan2": "atan2",
    "exp": "exp", "exp2": "exp2",
    "log": "log", "log2": "log2",
    "floor": "floor", "ceil": "ceil",
    "abs": "abs", "min": "min", "max": "max", "pow": "pow",
    "fabs": "abs",
}

# Default integer type for locals
_INT = "i32"


class WGSLCodeGen:
    """Generates WGSL source from a PGC IR function."""

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = ir_func
        self._indent = 0
        self._lines: list[str] = []
        self._param_types: dict[str, ScalarType] = {}
        self._field_params: set[str] = set()
        self._local_vars: dict[str, str] = {}
        self._declared_vars: set[str] = set()

    def _sanitize(self, name: str) -> str:
        """WGSL forbids identifiers starting with '__'. Replace with 'p_'."""
        if name.startswith("__"):
            return "p_" + name[2:]
        return name

    def generate(self) -> str:
        func = self.ir_func

        # Build parameter info
        for param in func.params:
            if param.type_annotation is None:
                raise TypeError(f"Parameter '{param.name}' has no type.")
            self._param_types[param.name] = param.type_annotation
            if hasattr(param, '_is_field') and param._is_field:
                self._field_params.add(param.name)
            elif not hasattr(param, '_is_field'):
                self._field_params.add(param.name)

        # Emit binding declarations
        binding_idx = 0
        for param in func.params:
            wgsl_type = _WGSL_TYPE_MAP.get(param.type_annotation, "f32")
            sname = self._sanitize(param.name)
            if param.name in self._field_params:
                self._lines.append(
                    f"@group(0) @binding({binding_idx}) "
                    f"var<storage, read_write> {sname}: array<{wgsl_type}>;")
            else:
                # Scalar params are packed into fields by ir_pack_scalars,
                # so this shouldn't happen after packing. But handle as
                # a read-only storage buffer just in case.
                self._lines.append(
                    f"@group(0) @binding({binding_idx}) "
                    f"var<storage, read> {sname}: array<{wgsl_type}>;")
            binding_idx += 1

        # Loop-end uniform
        self._n_binding = binding_idx
        self._lines.append(
            f"@group(0) @binding({binding_idx}) "
            f"var<storage, read> pgc_params: array<u32>;")

        self._lines.append("")
        self._lines.append("@compute @workgroup_size(256)")
        self._lines.append(f"fn {func.name}(@builtin(global_invocation_id) pgc_gid: vec3<u32>) {{")
        self._indent += 1

        self._emit_body(func.body)

        self._indent -= 1
        self._lines.append("}")

        return "\n".join(self._lines) + "\n"

    def _emit(self, text: str):
        self._lines.append("    " * self._indent + text)

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
        elif isinstance(node, ir.IRBreak):
            self._emit("break;")
        elif isinstance(node, ir.IRContinue):
            self._emit("continue;")
        elif isinstance(node, ir.IRReturn):
            self._emit("return;")
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRSharedAlloc):
            # WGSL workgroup vars must be module-scope; emit as comment
            self._emit(f"// shared: var<workgroup> {self._sanitize(node.name)}: array<{node.dtype}, {self._expr(node.size)}>;")
        elif isinstance(node, ir.IRLocalAlloc):
            wgsl_type = "f32" if node.dtype == "float" else "i32"
            self._emit(f"var {self._sanitize(node.name)}: array<{wgsl_type}, {self._expr(node.size)}>;")
        elif isinstance(node, ir.IRBarrier):
            self._emit("workgroupBarrier();")
        elif isinstance(node, ir.IRBlockReduce):
            # Block reduce as expression — handled in _expr
            pass
        elif isinstance(node, ir.IRPrint):
            pass  # no printf in WGSL
        elif isinstance(node, ir.IRCall):
            self._emit(f"{self._expr(node)};")
        else:
            raise NotImplementedError(f"WGSL stmt: {type(node).__name__}")

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        idx = self._sanitize(node.var)
        self._emit(f"let {idx} = {_INT}(pgc_gid.x);")
        self._emit(f"if ({idx} >= {_INT}(pgc_params[0])) {{ return; }}")
        self._local_vars[node.var] = _INT
        self._declared_vars.add(node.var)
        self._emit_body(node.body)

    def _emit_sequential_for(self, node: ir.IRSequentialFor):
        start = self._expr(node.start)
        end = self._expr(node.end)
        step = self._expr(node.step) if node.step else None
        var = self._sanitize(node.var)
        incr = f"{var} += {step}" if step else f"{var}++"
        self._emit(f"for (var {var}: {_INT} = {start}; {var} < {end}; {incr}) {{")
        self._local_vars[node.var] = _INT
        self._declared_vars.add(node.var)
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
        # Pre-declare variables assigned in branches
        then_new = self._collect_new_assigns(node.then_body)
        else_new = self._collect_new_assigns(node.else_body) if node.else_body else set()
        for var_name in sorted(then_new | else_new):
            if var_name not in self._declared_vars:
                c_type = self._find_assign_type(var_name, node.then_body) or \
                         self._find_assign_type(var_name, node.else_body or []) or "f32"
                self._emit(f"var {self._sanitize(var_name)}: {c_type};")
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

    def _collect_new_assigns(self, stmts):
        result = set()
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target not in self._declared_vars:
                result.add(stmt.target)
            elif isinstance(stmt, ir.IRIf):
                result |= self._collect_new_assigns(stmt.then_body)
                if stmt.else_body:
                    result |= self._collect_new_assigns(stmt.else_body)
        return result

    def _find_assign_type(self, var_name, stmts):
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target == var_name:
                if hasattr(stmt, '_resolved_type') and stmt._resolved_type:
                    return _WGSL_TYPE_MAP.get(stmt._resolved_type, "f32")
                return self._infer_type(stmt.value)
        return None

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        field_type = self._get_field_wgsl_type(node.field)
        self._emit(f"{field}[{_INT}({index})] = {field_type}({value});")

    def _emit_assign(self, node: ir.IRAssign):
        value = self._expr(node.value)
        target = self._sanitize(node.target)
        if node.target in self._declared_vars:
            self._emit(f"{target} = {value};")
        else:
            if hasattr(node, '_resolved_type') and node._resolved_type:
                wgsl_type = _WGSL_TYPE_MAP.get(node._resolved_type, "f32")
            else:
                wgsl_type = self._infer_type(node.value)
            self._emit(f"var {target}: {wgsl_type} = {value};")
            self._local_vars[node.target] = wgsl_type
            self._declared_vars.add(node.target)

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        # WGSL atomics require atomic<> type — not directly compatible
        # with storage arrays. For now, emit a non-atomic fallback.
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        if node.op == "add":
            self._emit(f"{field}[{_INT}({index})] = {field}[{_INT}({index})] + {value};")
        elif node.op == "max":
            self._emit(f"{field}[{_INT}({index})] = max({field}[{_INT}({index})], {value});")
        elif node.op == "min":
            self._emit(f"{field}[{_INT}({index})] = min({field}[{_INT}({index})], {value});")

    # --- Expression codegen ---

    def _expr(self, node) -> str:
        if isinstance(node, ir.IRConstant):
            if isinstance(node.value, float):
                return f"{node.value!r}"
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            return str(node.value)
        if isinstance(node, ir.IRName):
            return self._sanitize(node.name)
        if isinstance(node, ir.IRBinOp):
            return self._expr_binop(node)
        if isinstance(node, ir.IRUnaryOp):
            op = node.op
            operand = self._expr(node.operand)
            if op == "not":
                return f"(!({operand}))"
            return f"({op}{operand})"
        if isinstance(node, ir.IRCompare):
            left = self._expr(node.left)
            right = self._expr(node.right)
            return f"({left} {node.op} {right})"
        if isinstance(node, ir.IRBoolOp):
            parts = [self._expr(v) for v in node.values]
            op = " && " if node.op == "and" else " || "
            return f"({op.join(parts)})"
        if isinstance(node, ir.IRFieldLoad):
            return self._expr_field_load(node)
        if isinstance(node, ir.IRCall):
            return self._expr_call(node)
        if isinstance(node, ir.IRCast):
            return self._expr_cast(node)
        if isinstance(node, ir.IRIfExp):
            cond = self._expr(node.condition)
            then = self._expr(node.then_value)
            else_ = self._expr(node.else_value)
            return f"select({else_}, {then}, {cond})"
        if isinstance(node, ir.IRAttribute):
            return self._expr(node.obj)
        if isinstance(node, ir.IRThreadId):
            return "i32(pgc_gid.x) % 256"
        if isinstance(node, ir.IRBlockReduce):
            raise NotImplementedError(
                "pgc.block_sum/max/min not yet supported on WebGPU")
        if isinstance(node, ir.IRTextureSample):
            raise NotImplementedError(
                "texture3d not yet supported on WebGPU")
        raise NotImplementedError(f"WGSL expr: {type(node).__name__}")

    def _expr_binop(self, node: ir.IRBinOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        op = node.op
        if op == "**":
            return f"pow({left}, {right})"
        if op == "//":
            return f"({left} / {right})"
        if op == "%":
            return f"({left} % {right})"
        return f"({left} {op} {right})"

    def _expr_field_load(self, node: ir.IRFieldLoad) -> str:
        field = self._expr(node.field)
        index = self._expr(node.index)
        return f"{field}[{_INT}({index})]"

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]
        func = _WGSL_MATH.get(node.func_name)
        if func:
            return f"{func}({', '.join(args)})"
        raise NotImplementedError(f"WGSL builtin: {node.func_name}")

    def _expr_cast(self, node: ir.IRCast) -> str:
        val = self._expr(node.value)
        if node.dtype == "int":
            return f"{_INT}({val})"
        if node.dtype == "float":
            return f"f32({val})"
        return val

    def _get_field_wgsl_type(self, field_node) -> str:
        if isinstance(field_node, ir.IRName) and field_node.name in self._param_types:
            return _WGSL_TYPE_MAP.get(self._param_types[field_node.name], "f32")
        return "f32"

    def _infer_type(self, node) -> str:
        if isinstance(node, ir.IRConstant):
            return "f32" if isinstance(node.value, float) else _INT
        if isinstance(node, ir.IRFieldLoad):
            if isinstance(node.field, ir.IRName) and node.field.name in self._param_types:
                return _WGSL_TYPE_MAP.get(self._param_types[node.field.name], "f32")
        if isinstance(node, ir.IRBinOp):
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if lt == "f32" or rt == "f32":
                return "f32"
            return lt
        if isinstance(node, ir.IRCall):
            return "f32"
        if isinstance(node, ir.IRCast):
            return _INT if node.dtype == "int" else "f32"
        if isinstance(node, ir.IRCompare):
            return _INT
        if isinstance(node, ir.IRName):
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            if node.name in self._param_types:
                return _WGSL_TYPE_MAP.get(self._param_types[node.name], "f32")
        return "f32"


def generate_wgsl_source(ir_func: ir.IRFunction) -> str:
    codegen = WGSLCodeGen(ir_func)
    return codegen.generate()

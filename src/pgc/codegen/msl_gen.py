"""PGC MSL code generation — transforms PGC IR to Metal Shading Language source.

Generates a ``kernel void`` compute function where:
  - Each Field parameter becomes a ``device`` pointer with ``[[buffer(N)]]``
  - The outermost parallel for-loop maps to ``[[thread_position_in_grid]]``
  - Sequential for-loops, while-loops, if/else map to standard C control flow
  - Math builtins map to Metal stdlib functions (sqrt, sin, etc.)

All integer locals and loop indices use 64-bit ``long`` to support grids
with more than 2^31 elements.  Apple GPUs do not support double precision.
"""

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64


_MSL_TYPE_MAP = {
    f32: "float",
    i32: "int",
    i64: "long",
    u32: "uint",
    u64: "ulong",
}

# Default integer type for locals — 64-bit to support large grids (>2^31 elements).
_INT = "long"

_MATH_FUNCS = {
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
    "fabs": "abs",
    "abs": "abs",
    "pow": "pow",
}

_BINOP_MAP = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "<<": "<<", ">>": ">>", "&": "&", "|": "|", "^": "^",
}

_CMP_MAP = {
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
}


class MSLCodeGen:
    """Generates MSL source from a PGC IR function."""

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = ir_func
        self._indent = 0
        self._lines: list[str] = []
        self._param_types: dict[str, ScalarType] = {}
        self._field_params: set[str] = set()
        self._local_vars: dict[str, str] = {}  # name -> MSL type
        self._declared_vars: set[str] = set()

    def generate(self) -> str:
        """Generate MSL source for the kernel."""
        func = self.ir_func
        self._needs_local_tid = False

        # Pre-scan IR for shared memory / thread_id usage
        self._needs_local_tid = self._scan_for_threadgroup(func.body)

        # Build parameter info
        for param in func.params:
            if param.type_annotation is None:
                raise TypeError(f"Parameter '{param.name}' has no type. Run type inference first.")
            if param.type_annotation is f64:
                raise TypeError("Apple GPUs do not support double precision (f64).")
            self._param_types[param.name] = param.type_annotation
            if hasattr(param, '_is_field') and param._is_field:
                self._field_params.add(param.name)
            elif not hasattr(param, '_is_field'):
                # Default: treat as field for backwards compatibility
                self._field_params.add(param.name)

        # Header
        self._emit("#include <metal_stdlib>")
        self._emit("using namespace metal;")
        self._emit("")

        # Pre-scan for texture samples to generate helper functions.
        # We need to emit the kernel body first to discover texture helpers,
        # then insert them. Instead, pre-scan the IR for IRTextureSample nodes.
        self._texture_helpers = {}
        self._tex_sample_counter = 0
        self._pre_scan_textures(func.body)
        helpers = self._generate_texture_helpers()
        if helpers:
            for line in helpers.strip().split('\n'):
                self._lines.append(line)
            self._emit("")

        # Build function signature
        # Track which params are scalars for dereference in the body
        self._scalar_buffer_params: set[str] = set()
        params_msl = []
        for i, param in enumerate(func.params):
            msl_type = _MSL_TYPE_MAP[param.type_annotation]
            if param.name in self._field_params:
                params_msl.append(f"device {msl_type}* {param.name} [[buffer({i})]]")
            else:
                # Scalar passed as a single-element constant buffer
                params_msl.append(f"constant {msl_type}& {param.name} [[buffer({i})]]")

        params_msl.append("uint __tid__ [[thread_position_in_grid]]")
        if self._needs_local_tid:
            params_msl.append("uint __local_tid__ [[thread_position_in_threadgroup]]")

        sig = ",\n    ".join(params_msl)
        self._emit(f"kernel void {func.name}(")
        self._emit(f"    {sig})")
        self._emit("{")
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
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRPrint):
            self._emit("/* print not supported on Metal */")
        elif isinstance(node, ir.IRSharedAlloc):
            self._emit(f"threadgroup {node.dtype} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRBarrier):
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        elif isinstance(node, ir.IRCall):
            self._emit(f"{self._expr(node)};")
        else:
            raise NotImplementedError(f"MSL codegen: cannot emit {type(node).__name__}")

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        idx = node.var
        self._emit(f"{_INT} {idx} = __tid__;")
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
        # Pre-declare variables assigned in branches for outer scope visibility.
        # Two-pass hoisting to handle forward references (e.g. tuple swap temps
        # referencing variables that are also being hoisted in the same scope).
        then_new = self._collect_new_assigns(node.then_body)
        else_new = self._collect_new_assigns(node.else_body) if node.else_body else set()
        needs_hoist = then_new | else_new
        hoist_types = {}
        for var_name in sorted(needs_hoist):
            if var_name not in self._declared_vars:
                c_type = self._find_assign_type(var_name, node.then_body) or \
                         self._find_assign_type(var_name, node.else_body or []) or "float"
                hoist_types[var_name] = c_type
                self._local_vars[var_name] = c_type
        for var_name in list(hoist_types):
            if hoist_types[var_name] not in ("float", "double"):
                c_type = self._find_assign_type(var_name, node.then_body) or \
                         self._find_assign_type(var_name, node.else_body or []) or hoist_types[var_name]
                hoist_types[var_name] = c_type
                self._local_vars[var_name] = c_type
        for var_name in sorted(hoist_types):
            self._emit(f"{hoist_types[var_name]} {var_name};")
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
        for stmt in stmts:
            if isinstance(stmt, ir.IRAssign) and stmt.target == var_name:
                return self._infer_type(stmt.value)
            elif isinstance(stmt, ir.IRIf):
                t = self._find_assign_type(var_name, stmt.then_body)
                if t:
                    return t
                if stmt.else_body:
                    t = self._find_assign_type(var_name, stmt.else_body)
                    if t:
                        return t
        return None

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        """Emit a Metal atomic operation.

        Metal uses atomic_fetch_* on device atomic pointers.  For float atomics
        (atomic_add), Metal 3.0+ supports atomic_fetch_add_explicit on float.
        For min/max on floats, we use a compare-and-swap loop.
        """
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float",):
            index = f"(({_INT})({index}))"

        # Determine if field is float or int
        field_name = self._get_field_name(node.field)
        is_float = field_name and self._param_types.get(field_name) in (f32,)

        if node.op == "add":
            if is_float:
                # Use atomic_fetch_add_explicit on float (Metal 3.0+)
                self._emit(
                    f"atomic_fetch_add_explicit("
                    f"(volatile device atomic_float*)&{field}[{index}], "
                    f"{value}, memory_order_relaxed);")
            else:
                self._emit(
                    f"atomic_fetch_add_explicit("
                    f"(volatile device atomic_int*)&{field}[{index}], "
                    f"{value}, memory_order_relaxed);")
        elif node.op in ("min", "max"):
            if is_float:
                # Float atomic min/max via compare-and-swap loop
                self._emit("{")
                self._indent += 1
                self._emit(f"float __val__ = {value};")
                self._emit(f"volatile device atomic_uint* __p__ = "
                           f"(volatile device atomic_uint*)&{field}[{index}];")
                self._emit(f"uint __old__ = atomic_load_explicit(__p__, memory_order_relaxed);")
                self._emit(f"while (true) {{")
                self._indent += 1
                self._emit(f"float __old_f__ = as_type<float>(__old__);")
                cmp = "<=" if node.op == "min" else ">="
                self._emit(f"if (__old_f__ {cmp} __val__) break;")
                self._emit(f"uint __new__ = as_type<uint>(__val__);")
                self._emit(f"if (atomic_compare_exchange_weak_explicit(__p__, &__old__, __new__, "
                           f"memory_order_relaxed, memory_order_relaxed)) break;")
                self._indent -= 1
                self._emit("}")
                self._indent -= 1
                self._emit("}")
            else:
                func = "atomic_fetch_min_explicit" if node.op == "min" else "atomic_fetch_max_explicit"
                self._emit(
                    f"{func}("
                    f"(volatile device atomic_int*)&{field}[{index}], "
                    f"{value}, memory_order_relaxed);")
        else:
            raise NotImplementedError(f"MSL atomic op: {node.op}")

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_INT})({index}))"
        self._emit(f"{field}[{index}] = {value};")

    def _emit_assign(self, node: ir.IRAssign):
        value = self._expr(node.value)
        if node.target in self._declared_vars:
            self._emit(f"{node.target} = {value};")
        else:
            c_type = self._infer_type(node.value)
            self._emit(f"{c_type} {node.target} = {value};")
            self._local_vars[node.target] = c_type
            self._declared_vars.add(node.target)

    def _infer_type(self, node) -> str:
        """Best-effort type inference for local variable declarations."""
        if isinstance(node, ir.IRConstant):
            if isinstance(node.value, float):
                return "float"
            return _INT
        if isinstance(node, ir.IRFieldLoad):
            field_name = self._get_field_name(node.field)
            if field_name and field_name in self._param_types:
                return _MSL_TYPE_MAP[self._param_types[field_name]]
        if isinstance(node, ir.IRBinOp):
            lt = self._infer_type(node.left)
            rt = self._infer_type(node.right)
            if lt == "float" or rt == "float":
                return "float"
            return lt
        if isinstance(node, ir.IRCall):
            return "float"
        if isinstance(node, ir.IRCast):
            if node.dtype == "int":
                return _INT
            if node.dtype == "float":
                return "float"
        if isinstance(node, ir.IRIfExp):
            return self._infer_type(node.then_value)
        if isinstance(node, ir.IRCompare):
            return _INT
        if isinstance(node, ir.IRUnaryOp):
            return self._infer_type(node.operand)
        if isinstance(node, ir.IRName):
            if node.name in self._field_params:
                c_type = _MSL_TYPE_MAP[self._param_types[node.name]]
                return f"device {c_type}*"
            if node.name in self._param_types:
                return _MSL_TYPE_MAP[self._param_types[node.name]]
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            return _INT
        return "float"

    def _infer_expr_type(self, node) -> str:
        if isinstance(node, ir.IRName):
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            if node.name in self._param_types:
                return _MSL_TYPE_MAP[self._param_types[node.name]]
            return _INT
        if isinstance(node, ir.IRBinOp):
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if "float" in (lt, rt):
                return "float"
            return lt
        if isinstance(node, ir.IRCall):
            return "float"
        if isinstance(node, ir.IRCast):
            if node.dtype == "int":
                return _INT
            return "float"
        return self._infer_type(node)

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
        if isinstance(node, ir.IRTextureSample):
            return self._expr_texture_sample(node)
        if isinstance(node, ir.IRThreadId):
            self._needs_local_tid = True
            return "__local_tid__"
        raise NotImplementedError(f"MSL expr: {type(node).__name__}")

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
            return f"pow({left}, {right})"
        if node.op == "//":
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt not in ("float",) and rt not in ("float",):
                return f"({left} / {right})"
            return f"(({_INT})floor((float)({left}) / (float)({right})))"
        if node.op in _BINOP_MAP:
            return f"({left} {_BINOP_MAP[node.op]} {right})"
        raise NotImplementedError(f"MSL binop: {node.op}")

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
        raise NotImplementedError(f"MSL unaryop: {node.op}")

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
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float",):
            index = f"(({_INT})({index}))"
        return f"{field}[{index}]"

    def _expr_attribute(self, node: ir.IRAttribute) -> str:
        raise NotImplementedError(
            f"Attribute access '{node.attr}' should be resolved before MSL codegen."
        )

    def _pre_scan_textures(self, stmts):
        """Pre-scan IR to discover all IRTextureSample nodes and register helpers."""
        for stmt in stmts:
            self._pre_scan_textures_node(stmt)

    def _pre_scan_textures_node(self, node):
        if node is None:
            return
        if isinstance(node, ir.IRTextureSample):
            W, H, D = node.shape
            helper = f"__tex3d_linear_{W}_{H}_{D}__"
            self._texture_helpers[helper] = (W, H, D)
            for c in node.coords:
                self._pre_scan_textures_node(c)
            return
        # Recurse into compound nodes
        for attr in ('body', 'then_body', 'else_body'):
            children = getattr(node, attr, None)
            if isinstance(children, list):
                self._pre_scan_textures(children)
        for attr in ('value', 'condition', 'left', 'right', 'operand',
                      'field', 'index', 'then_value', 'else_value'):
            child = getattr(node, attr, None)
            if isinstance(child, ir.IRNode):
                self._pre_scan_textures_node(child)
        if hasattr(node, 'args') and isinstance(node.args, list):
            for a in node.args:
                if isinstance(a, ir.IRNode):
                    self._pre_scan_textures_node(a)
        if hasattr(node, 'values') and isinstance(node.values, list):
            for v in node.values:
                if isinstance(v, ir.IRNode):
                    self._pre_scan_textures_node(v)

    def _expr_texture_sample(self, node: ir.IRTextureSample) -> str:
        """Emit software trilinear interpolation for texture3d.sample()."""
        W, H, D = node.shape
        u = self._expr(node.coords[0])
        v = self._expr(node.coords[1])
        w = self._expr(node.coords[2])
        field = node.field_name

        # Use a unique counter to avoid name collisions in the same scope
        n = getattr(self, '_tex_sample_counter', 0)
        self._tex_sample_counter = n + 1
        p = f"__ts{n}__"

        # Emit inline trilinear as a sequence of local declarations.
        # We use the _pre_stmts pattern — but MSL codegen doesn't have that.
        # Instead, emit a block expression using a lambda or just inline vars.
        # Simplest: emit a helper call. Generate a helper function once.
        helper = f"__tex3d_linear_{W}_{H}_{D}__"
        if not hasattr(self, '_texture_helpers'):
            self._texture_helpers = {}
        if helper not in self._texture_helpers:
            self._texture_helpers[helper] = (W, H, D)
        return f"{helper}({field}, {u}, {v}, {w})"

    def _generate_texture_helpers(self) -> str:
        """Generate MSL helper functions for software texture sampling."""
        if not hasattr(self, '_texture_helpers') or not self._texture_helpers:
            return ""
        lines = []
        for name, (W, H, D) in self._texture_helpers.items():
            lines.append(f"""
inline float {name}(device float* data, float u, float v, float w) {{
    float fx = u * {W - 1}.0f;
    float fy = v * {H - 1}.0f;
    float fz = w * {D - 1}.0f;
    long ix = (long)floor(fx);
    long iy = (long)floor(fy);
    long iz = (long)floor(fz);
    float dx = fx - (float)ix;
    float dy = fy - (float)iy;
    float dz = fz - (float)iz;
    ix = clamp(ix, 0L, {W - 1}L);
    iy = clamp(iy, 0L, {H - 1}L);
    iz = clamp(iz, 0L, {D - 1}L);
    long ix1 = min(ix + 1, {W - 1}L);
    long iy1 = min(iy + 1, {H - 1}L);
    long iz1 = min(iz + 1, {D - 1}L);
    float c000 = data[iz  * {W * H}L + iy  * {W}L + ix ];
    float c100 = data[iz  * {W * H}L + iy  * {W}L + ix1];
    float c010 = data[iz  * {W * H}L + iy1 * {W}L + ix ];
    float c110 = data[iz  * {W * H}L + iy1 * {W}L + ix1];
    float c001 = data[iz1 * {W * H}L + iy  * {W}L + ix ];
    float c101 = data[iz1 * {W * H}L + iy  * {W}L + ix1];
    float c011 = data[iz1 * {W * H}L + iy1 * {W}L + ix ];
    float c111 = data[iz1 * {W * H}L + iy1 * {W}L + ix1];
    float c00 = c000 * (1.0f - dx) + c100 * dx;
    float c10 = c010 * (1.0f - dx) + c110 * dx;
    float c01 = c001 * (1.0f - dx) + c101 * dx;
    float c11 = c011 * (1.0f - dx) + c111 * dx;
    float c0 = c00 * (1.0f - dy) + c10 * dy;
    float c1 = c01 * (1.0f - dy) + c11 * dy;
    return c0 * (1.0f - dz) + c1 * dz;
}}
""")
        return "\n".join(lines)

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]

        if node.func_name == "min" and len(args) == 2:
            # Cast to float to avoid ambiguity between min(int,int) and min(float,float)
            return f"min((float)({args[0]}), (float)({args[1]}))"
        if node.func_name == "max" and len(args) == 2:
            return f"max((float)({args[0]}), (float)({args[1]}))"

        if node.func_name in _MATH_FUNCS:
            func = _MATH_FUNCS[node.func_name]
            return f"{func}({', '.join(args)})"

        raise NotImplementedError(f"MSL builtin: {node.func_name}")

    def _expr_cast(self, node: ir.IRCast) -> str:
        val = self._expr(node.value)
        if node.dtype == "int":
            return f"((int)({val}))"
        if node.dtype == "float":
            return f"((float)({val}))"
        raise NotImplementedError(f"MSL cast: {node.dtype}")

    def _expr_ifexp(self, node: ir.IRIfExp) -> str:
        cond = self._expr(node.condition)
        then = self._expr(node.then_value)
        else_ = self._expr(node.else_value)
        return f"({cond} ? {then} : {else_})"


    def _scan_for_threadgroup(self, stmts: list) -> bool:
        """Check if any statement uses threadgroup features."""
        for stmt in stmts:
            if isinstance(stmt, (ir.IRSharedAlloc, ir.IRBarrier, ir.IRThreadId)):
                return True
            if isinstance(stmt, ir.IRParallelFor):
                if self._scan_for_threadgroup(stmt.body):
                    return True
            elif isinstance(stmt, ir.IRSequentialFor):
                if self._scan_for_threadgroup(stmt.body):
                    return True
            elif isinstance(stmt, ir.IRWhile):
                if self._scan_for_threadgroup(stmt.body):
                    return True
            elif isinstance(stmt, ir.IRIf):
                if self._scan_for_threadgroup(stmt.then_body):
                    return True
                if stmt.else_body and self._scan_for_threadgroup(stmt.else_body):
                    return True
            # Check expressions for IRThreadId
            elif isinstance(stmt, ir.IRAssign):
                if self._expr_contains_thread_id(stmt.value):
                    return True
            elif isinstance(stmt, ir.IRFieldStore):
                if (self._expr_contains_thread_id(stmt.index) or
                        self._expr_contains_thread_id(stmt.value)):
                    return True
        return False

    def _expr_contains_thread_id(self, node) -> bool:
        """Check if an expression contains IRThreadId."""
        if isinstance(node, ir.IRThreadId):
            return True
        if isinstance(node, ir.IRBinOp):
            return (self._expr_contains_thread_id(node.left) or
                    self._expr_contains_thread_id(node.right))
        if isinstance(node, ir.IRUnaryOp):
            return self._expr_contains_thread_id(node.operand)
        if isinstance(node, ir.IRCall):
            return any(self._expr_contains_thread_id(a) for a in node.args)
        if isinstance(node, ir.IRCast):
            return self._expr_contains_thread_id(node.value)
        if isinstance(node, ir.IRFieldLoad):
            return self._expr_contains_thread_id(node.index)
        return False


def generate_msl_source(ir_func: ir.IRFunction) -> str:
    """Generate MSL source for a single kernel function."""
    codegen = MSLCodeGen(ir_func)
    return codegen.generate()

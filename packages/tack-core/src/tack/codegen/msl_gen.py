"""Tack MSL code generation — transforms Tack IR to Metal Shading Language source.

Generates a ``kernel void`` compute function where:
  - Each Field parameter becomes a ``device`` pointer with ``[[buffer(N)]]``
  - The outermost parallel for-loop maps to ``[[thread_position_in_grid]]``
  - Sequential for-loops, while-loops, if/else map to standard C control flow
  - Math builtins map to Metal stdlib functions (sqrt, sin, etc.)

All integer locals and loop indices use 64-bit ``long`` to support grids
with more than 2^31 elements.  Apple GPUs do not support double precision.
"""

from tack.lang import ir
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

_MSL_TYPE_MAP = {
    i8:  "char",
    u8:  "uchar",
    i16: "short",
    u16: "ushort",
    i32: "int",
    u32: "uint",
    i64: "long",
    u64: "ulong",
    f32: "float",
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

# C/MSL reserved words that cannot be used as kernel function names
_MSL_RESERVED = frozenset({
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "return", "short", "signed",
    "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned",
    "void", "volatile", "while", "half", "uint", "uchar", "ushort", "ulong",
})


def _safe_kernel_name(name: str) -> str:
    if name in _MSL_RESERVED:
        return f"_tack_{name}"
    return name


_CMP_MAP = {
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
}


class MSLCodeGen:
    """Generates MSL source from a Tack IR function."""

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

        # Detect texture parameters
        self._texture_params: set[str] = set()
        for param in func.params:
            if getattr(param, '_is_texture', False):
                self._texture_params.add(param.name)

        # Build function signature with separate buffer/texture binding indices
        self._scalar_buffer_params: set[str] = set()
        params_msl = []
        buf_idx = 0
        tex_idx = 0
        for param in func.params:
            msl_type = _MSL_TYPE_MAP[param.type_annotation]
            if param.name in self._texture_params:
                params_msl.append(
                    f"texture3d<float, access::sample> {param.name} [[texture({tex_idx})]]")
                tex_idx += 1
            elif param.name in self._field_params:
                params_msl.append(f"device {msl_type}* {param.name} [[buffer({buf_idx})]]")
                buf_idx += 1
            else:
                params_msl.append(f"constant {msl_type}& {param.name} [[buffer({buf_idx})]]")
                buf_idx += 1
        self._has_textures = tex_idx > 0

        params_msl.append("uint __tid__ [[thread_position_in_grid]]")
        if self._needs_local_tid:
            params_msl.append("uint __local_tid__ [[thread_position_in_threadgroup]]")

        sig = ",\n    ".join(params_msl)
        safe_name = _safe_kernel_name(func.name)
        self._emit(f"kernel void {safe_name}(")
        self._emit(f"    {sig})")
        self._emit("{")
        self._indent += 1

        # Emit sampler for texture sampling
        if self._has_textures:
            self._emit("constexpr sampler __samp__(coord::normalized, "
                       "filter::linear, address::clamp_to_edge);")

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
            msl_type = _MSL_TYPE_MAP[node.dtype]
            self._emit(f"threadgroup {msl_type} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRLocalAlloc):
            msl_type = _MSL_TYPE_MAP[node.dtype]
            self._emit(f"{msl_type} {node.name}[{self._expr(node.size)}];")
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
        # Always declare the loop variable in the for-header to handle
        # re-use of the same variable name in sibling loops (C block scoping).
        self._emit(f"for ({_INT} {var} = {start}; {var} < {end}; {incr}) {{")
        self._local_vars[var] = _INT
        self._declared_vars.add(var)
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
                if hasattr(stmt, '_resolved_type') and stmt._resolved_type is not None:
                    return _MSL_TYPE_MAP.get(stmt._resolved_type, self._infer_type(stmt.value))
                return self._infer_type(stmt.value)
            if isinstance(stmt, ir.IRIf):
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
                self._emit("uint __old__ = atomic_load_explicit(__p__, memory_order_relaxed);")
                self._emit("while (true) {")
                self._indent += 1
                self._emit("float __old_f__ = as_type<float>(__old__);")
                cmp = "<=" if node.op == "min" else ">="
                self._emit(f"if (__old_f__ {cmp} __val__) break;")
                self._emit("uint __new__ = as_type<uint>(__val__);")
                self._emit("if (atomic_compare_exchange_weak_explicit(__p__, &__old__, __new__, "
                           "memory_order_relaxed, memory_order_relaxed)) break;")
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
            if hasattr(node, '_resolved_type') and node._resolved_type is not None:
                c_type = _MSL_TYPE_MAP.get(node._resolved_type, self._infer_type(node.value))
            else:
                c_type = self._infer_type(node.value)
            self._emit(f"{c_type} {node.target} = {value};")
            self._local_vars[node.target] = c_type
            self._declared_vars.add(node.target)

    def _infer_type(self, node) -> str:
        """Get MSL type for an IR expression node, using annotated dtype."""
        # Use type annotation from ir_type_annotate pass
        dtype = getattr(node, 'dtype', None)
        if dtype is not None:
            return _MSL_TYPE_MAP.get(dtype, "float")
        # Check _resolved_cast_type for IRCast nodes
        rct = getattr(node, '_resolved_cast_type', None)
        if rct is not None:
            return _MSL_TYPE_MAP.get(rct, "float")
        # Fallback for unannotated nodes (e.g., field pointer references)
        if isinstance(node, ir.IRName):
            if node.name in self._field_params:
                c_type = _MSL_TYPE_MAP[self._param_types[node.name]]
                return f"device {c_type}*"
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            if node.name in self._param_types:
                return _MSL_TYPE_MAP[self._param_types[node.name]]
            return _INT
        return "float"

    def _infer_expr_type(self, node) -> str:
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
        if isinstance(node, ir.IRBlockReduce):
            return self._expr_block_reduce(node)
        raise NotImplementedError(f"MSL expr: {type(node).__name__}")

    def _expr_block_reduce(self, node: ir.IRBlockReduce) -> str:
        """Emit a threadgroup memory tree reduction."""
        if not hasattr(self, '_block_reduce_counter'):
            self._block_reduce_counter = 0
        idx = self._block_reduce_counter
        self._block_reduce_counter += 1

        smem = f"__breduce_smem_{idx}__"
        tid = f"__breduce_tid_{idx}__"
        result = f"__breduce_result_{idx}__"

        self._needs_local_tid = True
        val_expr = self._expr(node.value)

        op_expr = {
            "sum": lambda a, b: f"({a} + {b})",
            "max": lambda a, b: f"max((float)({a}), (float)({b}))",
            "min": lambda a, b: f"min((float)({a}), (float)({b}))",
        }[node.op]

        self._emit(f"threadgroup float {smem}[256];")
        self._emit(f"int {tid} = __local_tid__;")
        self._emit(f"{smem}[{tid}] = (float)({val_expr});")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._emit("for (int __s = 128; __s > 0; __s >>= 1) {")
        self._indent += 1
        self._emit(f"if ({tid} < __s) {{")
        self._indent += 1
        self._emit(f"{smem}[{tid}] = {op_expr(f'{smem}[{tid}]', f'{smem}[{tid} + __s]')};")
        self._indent -= 1
        self._emit("}")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._indent -= 1
        self._emit("}")
        self._emit(f"float {result} = {smem}[0];")
        self._local_vars[result] = "float"
        self._declared_vars.add(result)
        return result

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
        """Emit hardware texture sampling via Metal texture3d.sample().

        Our API convention: texel centers at i/(N-1), so u=0 → texel 0, u=1 → texel N-1.
        Metal convention:   texel centers at (i+0.5)/N.
        Transform: metal_u = (u * (N-1) + 0.5) / N
        """
        W, H, D = node.shape
        u = self._expr(node.coords[0])
        v = self._expr(node.coords[1])
        w = self._expr(node.coords[2])
        field = node.field_name
        mu = f"(({u}) * {W - 1}.0f + 0.5f) / {W}.0f"
        mv = f"(({v}) * {H - 1}.0f + 0.5f) / {H}.0f"
        mw = f"(({w}) * {D - 1}.0f + 0.5f) / {D}.0f"
        return f"{field}.sample(__samp__, float3({mu}, {mv}, {mw})).x"

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
        if isinstance(node.dtype, ScalarType):
            msl_type = _MSL_TYPE_MAP.get(node.dtype)
            if msl_type is None:
                raise NotImplementedError(f"MSL does not support {node.dtype} (Apple GPUs lack f64)")
            return f"(({msl_type})({val}))"
        # Legacy string fallback
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
            if isinstance(stmt, (ir.IRSharedAlloc, ir.IRBarrier, ir.IRThreadId, ir.IRBlockReduce)):
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
        """Check if an expression contains IRThreadId or IRBlockReduce."""
        if isinstance(node, (ir.IRThreadId, ir.IRBlockReduce)):
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

"""Tack MSL code generation — transforms Tack IR to Metal Shading Language source.

Generates a ``kernel void`` compute function where:
  - Field pointers are members of one argument buffer, so they may alias
  - The outermost parallel for-loop maps to ``[[thread_position_in_grid]]``
  - Sequential for-loops, while-loops, if/else map to standard C control flow
  - Math builtins map to Metal stdlib functions (sqrt, sin, etc.)

All integer locals and loop indices use 64-bit ``long`` to support grids
with more than 2^31 elements.  Apple GPUs do not support double precision.
"""

from tack.codegen.float_division import float_division_expr, float_division_helpers
from tack.codegen.identifiers import kernel_entry_name, rename_gpu_bindings
from tack.codegen.integer_division import integer_division_expr, integer_division_helpers
from tack.codegen.integer_ops import IntegerCodeGen
from tack.codegen.reductions import f32_reduction_helpers
from tack.lang import ir
from tack.lang.atomic_support import check_atomic_support
from tack.lang.ir_traversal import walk_ir
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import WORKGROUP_SIZE, check_workgroup_participation
from tack.lang.workgroup_support import workgroup_features

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
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
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


# The function a kernel's body moves to when it has a loop that stores to a
# field. Generated variables carry the `tack_var_` prefix, so it cannot
# collide with one.
_BODY_FUNCTION = "__tack_body__"


def _stores_inside_sequential_loop(body) -> bool:
    """Whether a sequential loop of the kernel contains a store or an atomic."""
    for node in walk_ir(body):
        if isinstance(node, (ir.IRSequentialFor, ir.IRWhile)):
            if any(isinstance(inner, (ir.IRFieldStore, ir.IRAtomicOp))
                   for inner in walk_ir(node)):
                return True
    return False


class MSLCodeGen:
    """Generates MSL source from a Tack IR function."""

    _integer_type_map = _MSL_TYPE_MAP

    def __init__(self, ir_func: ir.IRFunction):
        self.ir_func = rename_gpu_bindings(ir_func)
        self._indent = 0
        self._lines: list[str] = []
        self._param_types: dict[str, ScalarType] = {}
        self._field_params: set[str] = set()
        self._local_vars: dict[str, str] = {}  # name -> MSL type
        self._declared_vars: set[str] = set()
        self._integer_division_helpers = set()
        self._float_division_helpers = set()
        self._block_extrema = set()
        self._integers = IntegerCodeGen(self._integer_type_map, bitcast=True)
        self._dynamic_range_depth = 0
        self._opaque_integer_add = False

    def generate(self) -> str:
        """Generate MSL source for the kernel."""
        func = self.ir_func
        check_atomic_support(func, backend_name='metal')
        check_workgroup_participation(func)
        self._needs_local_tid = bool(workgroup_features(func))

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
        preamble_end = len(self._lines)

        # Detect texture parameters
        self._texture_params: set[str] = set()
        for param in func.params:
            if getattr(param, '_is_texture', False):
                self._texture_params.add(param.name)

        # Separate device-buffer arguments promise disjoint storage in MSL
        # (section 5.2). Indirect pointers in one argument buffer preserve
        # Tack's overlap contract without specializing on alias relationships.
        buffer_params = [p for p in func.params
                         if p.name in self._field_params
                         and p.name not in self._texture_params]
        if buffer_params:
            self._emit("struct __tack_buffer_args__ {")
            for i, param in enumerate(func.params):
                if param.name in self._field_params and param.name not in self._texture_params:
                    msl_type = _MSL_TYPE_MAP[param.type_annotation]
                    self._emit(f"    device {msl_type}* {param.name} [[id({i})]];")
            self._emit("};")
            self._emit("")

        # Textures keep their separate binding namespace. Unpacked scalars
        # remain constant references; normal dispatch packs them into fields.
        self._scalar_buffer_params: set[str] = set()
        # Each parameter as (declaration, binding attribute, name): the
        # kernel entry declares them with their attributes, and a separate
        # body function (below) takes the same ones without.
        params = []
        buf_idx = 0
        if buffer_params:
            params.append(("constant __tack_buffer_args__& __tack_buffers__",
                           "[[buffer(0)]]", "__tack_buffers__"))
            buf_idx = 1
        tex_idx = 0
        for param in func.params:
            msl_type = _MSL_TYPE_MAP[param.type_annotation]
            if param.name in self._texture_params:
                params.append((f"texture3d<float, access::sample> {param.name}",
                               f"[[texture({tex_idx})]]", param.name))
                tex_idx += 1
            elif param.name in self._field_params:
                continue
            else:
                params.append((f"constant {msl_type}& {param.name}",
                               f"[[buffer({buf_idx})]]", param.name))
                buf_idx += 1
        self._has_textures = tex_idx > 0

        params.append(("uint __tid__", "[[thread_position_in_grid]]", "__tid__"))
        if self._needs_local_tid:
            params.append(("uint __local_tid__", "[[thread_position_in_threadgroup]]",
                           "__local_tid__"))

        # A loop that stores to a field is compiled in a function of its
        # own. In the kernel entry function, Apple's compiler (M1 Max,
        # macOS 26) reads a field element once before such a loop and never
        # again when the element's address does not depend on the thread
        # and the loop also stores to another field of the same type:
        #     for k in range(n): total[0] += x[k]; counter[0] += 1
        # left total at start + x[n - 1]. It began when field pointers
        # became members of one argument buffer. The same loop in a
        # function the entry calls is compiled correctly, provided the
        # function is not inlined; hiding the pointers or the thread index
        # behind opaque calls is not enough, nor is `restrict`. Loop-free
        # kernels were never affected and keep the single function, which
        # is 7-15% faster for the smallest of them.
        self._body_function = _stores_inside_sequential_loop(func.body)
        self._workgroup_arrays: list[tuple[str, str, str]] = []

        sig = ",\n    ".join(f"{decl} {attr}" for decl, attr, _ in params)
        safe_name = kernel_entry_name(func.name)
        if self._body_function:
            body_signature = len(self._lines)
            self._emit("")        # written once the body's workgroup arrays are known
        else:
            self._emit(f"kernel void {safe_name}(")
            self._emit(f"    {sig})")
        self._emit("{")
        self._indent += 1

        for param in buffer_params:
            msl_type = _MSL_TYPE_MAP[param.type_annotation]
            self._emit(f"device {msl_type}* {param.name} = __tack_buffers__.{param.name};")

        # Emit sampler for texture sampling
        if self._has_textures:
            self._emit("constexpr sampler __samp__(coord::normalized, "
                       "filter::linear, address::clamp_to_edge);")

        self._declare_locals_at_kernel_scope(func.body)
        self._emit_body(func.body)

        self._indent -= 1
        self._emit("}")

        if self._body_function:
            # Workgroup arrays can only be declared in the kernel function,
            # so the entry declares them and the body takes pointers.
            body_params = [decl for decl, _, _ in params] + [
                f"threadgroup {msl_type}* {name}" for msl_type, name, _ in self._workgroup_arrays]
            arguments = [name for _, _, name in params] + [
                name for _, name, _ in self._workgroup_arrays]
            self._lines[body_signature] = (
                f"__attribute__((noinline)) static void {_BODY_FUNCTION}("
                f"{', '.join(body_params)})")
            self._emit("")
            self._emit(f"kernel void {safe_name}(")
            self._emit(f"    {sig})")
            self._emit("{")
            for msl_type, name, size in self._workgroup_arrays:
                self._emit(f"    threadgroup {msl_type} {name}[{size}];")
            self._emit(f"    {_BODY_FUNCTION}({', '.join(arguments)});")
            self._emit("}")

        helpers = (
            float_division_helpers(
                self._float_division_helpers, _MSL_TYPE_MAP, 'inline')
            + integer_division_helpers(
                self._integer_division_helpers, _MSL_TYPE_MAP, 'inline')
            + self._integers.definitions('inline')
            + f32_reduction_helpers('metal', self._block_extrema)
        )
        return "\n".join(self._lines[:preamble_end] + helpers
                         + self._lines[preamble_end:]) + "\n"

    def _emit(self, line: str):
        self._lines.append("    " * self._indent + line)

    def _declare_workgroup_array(self, msl_type: str, name: str, size: str):
        """Declare a threadgroup array where MSL allows it: in the kernel function."""
        if self._body_function:
            self._workgroup_arrays.append((msl_type, name, size))
        else:
            self._emit(f"threadgroup {msl_type} {name}[{size}];")

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
            # The kernel body is one iteration of the parallel loop, so
            # continuing that loop means leaving the kernel.
            self._emit("return;" if node.outermost else "continue;")
        elif isinstance(node, ir.IRAtomicOp):
            self._emit_atomic_op(node)
        elif isinstance(node, ir.IRPrint):
            self._emit("/* print not supported on Metal */")
        elif isinstance(node, ir.IRSharedAlloc):
            self._declare_workgroup_array(
                _MSL_TYPE_MAP[node.dtype], node.name, self._expr(node.size))
        elif isinstance(node, ir.IRLocalAlloc):
            msl_type = _MSL_TYPE_MAP[node.dtype]
            self._emit(f"{msl_type} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRBarrier):
            self._emit("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);")
        elif isinstance(node, ir.IRCall):
            self._emit(f"{self._expr(node)};")
        else:
            raise NotImplementedError(f"MSL codegen: cannot emit {type(node).__name__}")

    def _declare_locals_at_kernel_scope(self, body):
        """Declare every typed local once, at kernel scope; see CUDACodeGen."""
        for node in walk_ir(body):
            if not isinstance(node, ir.IRAssign) or node.target in self._declared_vars:
                continue
            resolved = getattr(node, '_resolved_type', None)
            if resolved is None:
                continue
            msl_type = _MSL_TYPE_MAP.get(resolved)
            if msl_type is None:
                continue
            self._emit(f"{msl_type} {node.target};")
            self._local_vars[node.target] = msl_type
            self._declared_vars.add(node.target)

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
        # The header's declaration ends with the loop, so a later plain
        # assignment to the same name must declare it again: restore the
        # bookkeeping on exit rather than leaving the name marked declared.
        outer = (var in self._declared_vars, self._local_vars.get(var))
        self._local_vars[var] = _INT
        self._declared_vars.add(var)
        self._indent += 1
        dynamic = not isinstance(node.end, ir.IRConstant)
        self._dynamic_range_depth += int(dynamic)
        self._emit_body(node.body)
        self._dynamic_range_depth -= int(dynamic)
        self._indent -= 1
        self._emit("}")
        self._leave_loop_scope(var, outer)

    def _leave_loop_scope(self, var, outer):
        """Forget a for-header declaration once its block closes."""
        was_declared, outer_type = outer
        if was_declared:
            self._local_vars[var] = outer_type
        else:
            self._declared_vars.discard(var)
            self._local_vars.pop(var, None)

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

        dtype = node.dtype
        is_float = dtype is f32
        atomic_type = 'atomic_uint' if dtype is u32 else 'atomic_int'
        value = self._integers.convert(value, getattr(node.value, 'dtype', None), dtype)
        value = f"(({_MSL_TYPE_MAP[dtype]})({value}))"

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
                    f"(volatile device {atomic_type}*)&{field}[{index}], "
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
                    f"(volatile device {atomic_type}*)&{field}[{index}], "
                    f"{value}, memory_order_relaxed);")
        else:
            raise NotImplementedError(f"MSL atomic op: {node.op}")

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        value = self._integers.convert(value, getattr(node.value, 'dtype', None), getattr(node, 'dtype', None))
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_INT})({index}))"
        self._emit(f"{field}[{index}] = {value};")

    def _emit_assign(self, node: ir.IRAssign):
        previous = self._opaque_integer_add
        self._opaque_integer_add = self._dynamic_range_depth > 0 and any(
            isinstance(n, ir.IRName) and n.name == node.target for n in walk_ir(node.value))
        value = self._expr(node.value)
        self._opaque_integer_add = previous
        value = self._integers.convert(value, getattr(node.value, 'dtype', None), getattr(node, '_resolved_type', None))
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

        if node.op != 'sum':
            self._block_extrema.add(node.op)

        op_expr = {
            "sum": lambda a, b: f"({a} + {b})",
            "max": lambda a, b: f"tack_reduce_max_f32({a}, {b})",
            "min": lambda a, b: f"tack_reduce_min_f32({a}, {b})",
        }[node.op]

        self._declare_workgroup_array("float", smem, str(WORKGROUP_SIZE))
        self._emit(f"int {tid} = __local_tid__;")
        self._emit(f"{smem}[{tid}] = (float)({val_expr});")
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._emit(f"for (int __s = {WORKGROUP_SIZE // 2}; __s > 0; __s >>= 1) {{")
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
        # Protect the result read against shared-array reuse in loops.
        self._emit("threadgroup_barrier(mem_flags::mem_threadgroup);")
        self._local_vars[result] = "float"
        self._declared_vars.add(result)
        return result

    def _expr_constant(self, node: ir.IRConstant) -> str:
        if isinstance(node.value, float):
            return f"{node.value!r}f"
        if isinstance(node.value, bool):
            return "1" if node.value else "0"
        if isinstance(node.value, int) and -(2**31) < node.value < 2**31:
            return str(node.value)
        if isinstance(node.value, int) and getattr(node, 'dtype', None) is not None:
            ctype = self._integer_type_map[node.dtype]
            # Spell signed minimum using representable tokens.
            literal = f'(-{-(node.value + 1)}LL - 1LL)' if node.value < 0 else f'{node.value}ULL'
            return f'(({ctype})({literal}))'
        return str(node.value)

    def _expr_binop(self, node: ir.IRBinOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        dtype = getattr(node, 'dtype', None)
        left = self._integers.convert(left, getattr(node.left, 'dtype', None), dtype)
        if node.op not in ('<<', '>>', '**'):
            right = self._integers.convert(right, getattr(node.right, 'dtype', None), dtype)
        if node.op == '/' and dtype is f32:
            return f'(((float)({left})) / ((float)({right})))'
        # M1 Max pipeline compilation crashes when wrapping 64-bit additions
        # participate in reduction optimization for runtime-bounded loops.
        # Keep this helper opaque there; other integer operations stay inline.
        noinline = self._opaque_integer_add and node.op == '+' and dtype in (i64, u64)
        fixed = self._integers.operation(node.op, dtype, left, right, noinline=noinline)
        if fixed is not None:
            return fixed
        integer = integer_division_expr(
            node, left, right, _MSL_TYPE_MAP, self._integer_division_helpers)
        if integer is not None:
            return integer
        if node.op == "**":
            return f'pow((float)({left}), (float)({right}))'
        if node.op in ('//', '%') and dtype is None:
            # Support the low-level generator API's legacy unannotated IR.
            operands = (self._infer_expr_type(node.left), self._infer_expr_type(node.right))
            dtype = f64 if 'double' in operands else f32 if 'float' in operands else None
        floating = float_division_expr(
            node, left, right, _MSL_TYPE_MAP, self._float_division_helpers, dtype=dtype)
        if floating is not None:
            return floating
        if node.op == "//":
            # Legacy unannotated integer IR; normal dispatch is fully typed.
            return f"({left} / {right})"
        if node.op in _BINOP_MAP:
            return f"({left} {_BINOP_MAP[node.op]} {right})"
        raise NotImplementedError(f"MSL binop: {node.op}")

    def _expr_unaryop(self, node: ir.IRUnaryOp) -> str:
        operand = self._expr(node.operand)
        if node.op == "+":
            # Before the integer helpers, which know '+' only as addition
            # and would emit a one-argument call to the two-argument add.
            return operand
        op = 'neg' if node.op == '-' else node.op
        fixed = self._integers.operation(op, getattr(node, 'dtype', None), operand)
        if fixed is not None:
            return fixed
        if node.op == "-":
            return f"(-{operand})"
        if node.op == "not":
            return f"(!{operand})"
        if node.op == "~":
            return f"(~{operand})"
        raise NotImplementedError(f"MSL unaryop: {node.op}")

    def _expr_compare(self, node: ir.IRCompare) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        dtype = getattr(node, '_operand_type', None)
        left = self._integers.convert(left, getattr(node.left, 'dtype', None), dtype)
        right = self._integers.convert(right, getattr(node.right, 'dtype', None), dtype)
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
        dtype = getattr(node, 'dtype', None)
        if dtype in (f32, f64):
            ctype = _MSL_TYPE_MAP[dtype]
            args = [f'(({ctype})({arg}))' for arg in args]
        if node.func_name == 'pow':
            fixed = self._integers.operation('**', dtype, *args)
            if fixed is not None:
                return fixed
            return f'pow((float)({args[0]}), (float)({args[1]}))'
        if node.func_name in ('abs', 'min', 'max'):
            converted = [self._integers.convert(a, getattr(n, 'dtype', None), dtype)
                         for a, n in zip(args, node.args)]
            fixed = self._integers.operation(node.func_name, dtype, *converted)
            if fixed is not None:
                return fixed

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
        converted = self._integers.convert(val, getattr(node.value, 'dtype', None), node.dtype)
        if converted != val:
            return converted
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
        dtype = getattr(node, 'dtype', None)
        then = self._integers.convert(then, getattr(node.then_value, 'dtype', None), dtype)
        else_ = self._integers.convert(else_, getattr(node.else_value, 'dtype', None), dtype)
        return f"({cond} ? {then} : {else_})"



def generate_msl_source(ir_func: ir.IRFunction) -> str:
    """Generate MSL source for a single kernel function."""
    codegen = MSLCodeGen(ir_func)
    return codegen.generate()

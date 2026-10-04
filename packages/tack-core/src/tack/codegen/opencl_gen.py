"""Tack OpenCL C code generation — transforms Tack IR to OpenCL C source.

Generates a ``__kernel`` function where:
  - Each Field parameter becomes a ``__global`` typed pointer
  - The outermost parallel for-loop maps to ``get_global_id(0)``
  - Sequential for-loops, while-loops, if/else map to standard C control flow
  - Math builtins map to OpenCL built-in math functions (overloaded, no 'f' suffix)

OpenCL C uses ``long`` for 64-bit integers (``long long`` is not standard OpenCL C).
This module reuses the CUDA codegen with OpenCL-specific overrides.
"""

from tack.codegen.cuda_gen import _BINOP_MAP, CUDACodeGen
from tack.codegen.float_division import float_division_expr, float_division_helpers
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.integer_division import integer_division_expr, integer_division_helpers
from tack.codegen.reductions import f32_reduction_helpers
from tack.lang import ir
from tack.lang.atomic_support import check_atomic_support
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import WORKGROUP_SIZE, check_workgroup_participation

# OpenCL uses 'long' for 64-bit integers (not 'long long')
_OCL_INT = "long"

_OCL_C_TYPE_MAP = {
    i8:  "char",
    u8:  "uchar",
    i16: "short",
    u16: "ushort",
    i32: "int",
    u32: "unsigned int",
    i64: "long",
    u64: "unsigned long",
    f32: "float",
    f64: "double",
}

_OCL_MATH_FUNCS = {
    "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan", "atan2": "atan2",
    "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2", "log10": "log10",
    "floor": "floor", "ceil": "ceil", "fabs": "fabs", "abs": "fabs",
    "pow": "pow",
}


class OpenCLCodeGen(CUDACodeGen):
    """Generates OpenCL C source from a Tack IR function.

    Reuses the CUDA codegen, overriding syntax that differs in OpenCL C:
    kernel qualifier, pointer qualifiers, thread indexing, math functions,
    atomics, shared memory, and barriers.
    """

    _integer_type_map = _OCL_C_TYPE_MAP

    def generate(self) -> str:
        func = self.ir_func
        check_atomic_support(func, backend_name='level_zero')
        check_workgroup_participation(func)
        # Declarations collected by `_declare_local` while the body is emitted,
        # spliced in below at the one scope OpenCL C accepts them in.
        self._hoisted_locals: dict[str, str] = {}

        # Build parameter info
        for param in func.params:
            if param.type_annotation is None:
                raise TypeError(f"Parameter '{param.name}' has no type. Run type inference first.")
            self._param_types[param.name] = param.type_annotation
            if hasattr(param, '_is_field') and param._is_field:
                self._field_params.add(param.name)
            elif not hasattr(param, '_is_field'):
                self._field_params.add(param.name)

        # Detect texture parameters for hardware image3d path
        self._texture_params: set[str] = set()
        for param in func.params:
            if getattr(param, '_is_texture', False):
                self._texture_params.add(param.name)

        # Build function signature with OpenCL qualifiers
        params_c = []
        for param in func.params:
            c_type = _OCL_C_TYPE_MAP[param.type_annotation]
            if param.name in self._texture_params:
                params_c.append(f"__read_only image3d_t {param.name}")
            elif param.name in self._field_params:
                params_c.append(f"__global {c_type}* {param.name}")
            else:
                params_c.append(f"{c_type} {param.name}")
        params_c.append(f"{_OCL_INT} __n__")

        sig = ", ".join(params_c)
        safe_name = kernel_entry_name(func.name)
        self._emit(f"__kernel void {safe_name}({sig}) {{")
        self._indent += 1

        # Emit constant sampler for hardware texture sampling
        if self._texture_params:
            self._emit("const sampler_t __samp__ = CLK_NORMALIZED_COORDS_TRUE "
                       "| CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;")

        hoist_at = len(self._lines)
        self._emit_body(func.body)
        self._lines[hoist_at:hoist_at] = [
            "    " + decl for decl in self._hoisted_locals.values()
        ]
        self._indent -= 1
        self._emit("}")

        # Prepend atomic helpers if needed
        prefix_lines = (
            float_division_helpers(
                self._float_division_helpers, _OCL_C_TYPE_MAP, 'static inline')
            + integer_division_helpers(
                self._integer_division_helpers, _OCL_C_TYPE_MAP, 'static inline')
            + self._integers.definitions('static inline')
            + f32_reduction_helpers('opencl', self._block_extrema)
        )
        if self._needs_float_atomic_min:
            prefix_lines.extend([
                "float atomicMinFloat(volatile __global float* addr, float val) {",
                "    volatile __global atomic_int* addr_as_int = (volatile __global atomic_int*)addr;",
                "    int old = atomic_load_explicit(addr_as_int, memory_order_relaxed, memory_scope_device), assumed;",
                "    do {",
                "        assumed = old;",
                "        int next = as_int(fmin(val, as_float(assumed)));",
                "        if (atomic_compare_exchange_weak_explicit(addr_as_int, &old, next,",
                "            memory_order_relaxed, memory_order_relaxed, memory_scope_device)) break;",
                "    } while (true);",
                "    return as_float(old);",
                "}",
                "",
            ])
        if self._needs_float_atomic_max:
            prefix_lines.extend([
                "float atomicMaxFloat(volatile __global float* addr, float val) {",
                "    volatile __global atomic_int* addr_as_int = (volatile __global atomic_int*)addr;",
                "    int old = atomic_load_explicit(addr_as_int, memory_order_relaxed, memory_scope_device), assumed;",
                "    do {",
                "        assumed = old;",
                "        int next = as_int(fmax(val, as_float(assumed)));",
                "        if (atomic_compare_exchange_weak_explicit(addr_as_int, &old, next,",
                "            memory_order_relaxed, memory_order_relaxed, memory_scope_device)) break;",
                "    } while (true);",
                "    return as_float(old);",
                "}",
                "",
            ])

        # Header pragmas
        header = ""
        body_text = "\n".join(self._lines)
        uses_f64 = "double" in body_text or any(
            p.type_annotation in (f64,) for p in func.params
        )
        if uses_f64:
            header += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"

        # Texture sampling helpers (software trilinear)
        if hasattr(self, '_texture_helpers') and self._texture_helpers:
            for name, (W, H, D) in self._texture_helpers.items():
                prefix_lines.extend([
                    f"inline float {name}(__global float* data, float u, float v, float w) {{",
                    f"    float fx = u * {W - 1}.0f, fy = v * {H - 1}.0f, fz = w * {D - 1}.0f;",
                    f"    {_OCL_INT} ix = ({_OCL_INT})floor(fx), iy = ({_OCL_INT})floor(fy), iz = ({_OCL_INT})floor(fz);",
                    "    float dx = fx - (float)ix, dy = fy - (float)iy, dz = fz - (float)iz;",
                    f"    ix = max(({_OCL_INT})0, min(ix, ({_OCL_INT}){W - 1}));",
                    f"    iy = max(({_OCL_INT})0, min(iy, ({_OCL_INT}){H - 1}));",
                    f"    iz = max(({_OCL_INT})0, min(iz, ({_OCL_INT}){D - 1}));",
                    f"    {_OCL_INT} ix1 = min(ix + 1, ({_OCL_INT}){W - 1});",
                    f"    {_OCL_INT} iy1 = min(iy + 1, ({_OCL_INT}){H - 1});",
                    f"    {_OCL_INT} iz1 = min(iz + 1, ({_OCL_INT}){D - 1});",
                    f"    float c000 = data[iz  * {W * H}L + iy  * {W}L + ix ];",
                    f"    float c100 = data[iz  * {W * H}L + iy  * {W}L + ix1];",
                    f"    float c010 = data[iz  * {W * H}L + iy1 * {W}L + ix ];",
                    f"    float c110 = data[iz  * {W * H}L + iy1 * {W}L + ix1];",
                    f"    float c001 = data[iz1 * {W * H}L + iy  * {W}L + ix ];",
                    f"    float c101 = data[iz1 * {W * H}L + iy  * {W}L + ix1];",
                    f"    float c011 = data[iz1 * {W * H}L + iy1 * {W}L + ix ];",
                    f"    float c111 = data[iz1 * {W * H}L + iy1 * {W}L + ix1];",
                    "    float c00 = c000 * (1.0f - dx) + c100 * dx;",
                    "    float c10 = c010 * (1.0f - dx) + c110 * dx;",
                    "    float c01 = c001 * (1.0f - dx) + c101 * dx;",
                    "    float c11 = c011 * (1.0f - dx) + c111 * dx;",
                    "    float c0 = c00 * (1.0f - dy) + c10 * dy;",
                    "    float c1 = c01 * (1.0f - dy) + c11 * dy;",
                    "    return c0 * (1.0f - dz) + c1 * dz;",
                    "}",
                    "",
                ])

        body = "\n".join(self._lines) + "\n"
        if prefix_lines:
            body = "\n".join(prefix_lines) + body
        if header:
            body = header + "\n" + body
        return body

    # --- Parallel loop: get_global_id(0) ---

    def _emit_parallel_for(self, node: ir.IRParallelFor):
        idx = node.var
        self._emit(f"{_OCL_INT} {idx} = get_global_id(0);")
        self._emit(f"if ({idx} >= __n__) return;")
        self._local_vars[idx] = _OCL_INT
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
        self._emit(f"for ({_OCL_INT} {var} = {start}; {var} < {end}; {incr}) {{")
        self._local_vars[var] = _OCL_INT
        self._declared_vars.add(var)
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")

    # --- Shared memory, barrier, thread ID ---

    def _declare_local(self, name: str, decl: str) -> None:
        """Queue a local-address-space declaration for the kernel's top scope.

        OpenCL C admits `__local` declarations only in the outermost scope of a
        kernel function, so one emitted where it is used -- inside an `if`, or
        inside a loop -- is a compile error. CUDA and Metal both allow a nested
        `__shared__`/`threadgroup`, so the inherited placement is legal there.
        `generate()` splices these in directly after the signature.

        Hoisting is safe because local memory is per-workgroup, statically
        sized and always written before it is read; only the declaration moves,
        never a store or a barrier, so stage six's participation guarantees are
        untouched.
        """
        previous = self._hoisted_locals.get(name)
        if previous is None:
            self._hoisted_locals[name] = decl
        elif previous != decl:
            raise NotImplementedError(
                f"Local allocation '{name}' is declared twice with different "
                f"types or sizes ({previous!r} vs {decl!r}). Hoisting to kernel "
                f"scope cannot keep both; rename one of them."
            )

    def _emit_stmt(self, node):
        if isinstance(node, ir.IRSharedAlloc):
            c_type = _OCL_C_TYPE_MAP[node.dtype]
            self._declare_local(
                node.name, f"__local {c_type} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRLocalAlloc):
            c_type = _OCL_C_TYPE_MAP[node.dtype]
            self._emit(f"{c_type} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRBarrier):
            self._emit("barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);")
        else:
            super()._emit_stmt(node)

    def _ifexp_condition(self, node) -> str:
        """Compare a floating condition against zero before using it in `?:`.

        OpenCL C requires the condition of the conditional operator to be of
        scalar *integer* type; a float is rejected outright ("used type 'float'
        where floating point type is not allowed"). C++ converts it implicitly,
        which is why the inherited CUDA/HIP spelling is correct there and why
        this only ever failed on a device. `if (x)` is unaffected -- OpenCL C
        allows a float there, following C99 -- so only the ternary needs this.
        """
        cond = self._expr(node)
        dtype = getattr(node, 'dtype', None)
        if dtype in (f32, f64):
            return f"({cond}) != {'0.0' if dtype is f64 else '0.0f'}"
        return cond

    def _expr(self, node) -> str:
        if isinstance(node, ir.IRThreadId):
            return "get_local_id(0)"
        if isinstance(node, ir.IRBlockReduce):
            return self._expr_block_reduce_ocl(node)
        return super()._expr(node)

    def _expr_block_reduce_ocl(self, node: ir.IRBlockReduce) -> str:
        """Emit a local memory tree reduction in OpenCL."""
        if not hasattr(self, '_block_reduce_counter'):
            self._block_reduce_counter = 0
        idx = self._block_reduce_counter
        self._block_reduce_counter += 1

        smem = f"__breduce_smem_{idx}__"
        tid = f"__breduce_tid_{idx}__"
        result = f"__breduce_result_{idx}__"

        val_expr = self._expr(node.value)

        if node.op != 'sum':
            self._block_extrema.add(node.op)

        op_expr = {
            "sum": lambda a, b: f"({a} + {b})",
            "max": lambda a, b: f"tack_reduce_max_f32({a}, {b})",
            "min": lambda a, b: f"tack_reduce_min_f32({a}, {b})",
        }[node.op]

        self._declare_local(smem, f"__local float {smem}[{WORKGROUP_SIZE}];")
        self._emit(f"int {tid} = get_local_id(0);")
        self._emit(f"{smem}[{tid}] = (float)({val_expr});")
        self._emit("barrier(CLK_LOCAL_MEM_FENCE);")
        self._emit(f"for (int __s = {WORKGROUP_SIZE // 2}; __s > 0; __s >>= 1) {{")
        self._indent += 1
        self._emit(f"if ({tid} < __s) {{")
        self._indent += 1
        self._emit(f"{smem}[{tid}] = {op_expr(f'{smem}[{tid}]', f'{smem}[{tid} + __s]')};")
        self._indent -= 1
        self._emit("}")
        self._emit("barrier(CLK_LOCAL_MEM_FENCE);")
        self._indent -= 1
        self._emit("}")
        self._emit(f"float {result} = {smem}[0];")
        # Protect the result read against shared-array reuse in loops.
        self._emit("barrier(CLK_LOCAL_MEM_FENCE);")
        self._local_vars[result] = "float"
        self._declared_vars.add(result)
        return result

    # --- Math functions (overloaded, no 'f' suffix) ---

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]
        dtype = getattr(node, 'dtype', None)
        if dtype in (f32, f64):
            ctype = _OCL_C_TYPE_MAP[dtype]
            args = [f'(({ctype})({arg}))' for arg in args]
        if node.func_name == 'pow':
            fixed = self._integers.operation('**', dtype, *args)
            if fixed is not None:
                return fixed
            t = _OCL_C_TYPE_MAP[f64 if dtype is f64 else f32]
            return f'pow(({t})({args[0]}), ({t})({args[1]}))'
        if node.func_name in ('abs', 'min', 'max'):
            converted = [self._integers.convert(a, getattr(n, 'dtype', None), dtype)
                         for a, n in zip(args, node.args)]
            fixed = self._integers.operation(node.func_name, dtype, *converted)
            if fixed is not None:
                return fixed

        if node.func_name == "min" and len(args) == 2:
            return f"fmin({args[0]}, {args[1]})"
        if node.func_name == "max" and len(args) == 2:
            return f"fmax({args[0]}, {args[1]})"

        if node.func_name in _OCL_MATH_FUNCS:
            func = _OCL_MATH_FUNCS[node.func_name]
            call = f"{func}({', '.join(args)})"
            if node.func_name in ('floor', 'ceil') and dtype is f64:
                # Intel's compute runtime returns +0.0 from the double
                # `floor`/`ceil` where IEEE-754 roundToIntegral requires the
                # sign of the operand to be preserved: `ceil(-0.1)` and
                # `floor(-0.0)` both come back +0.0 on an Intel Data Center GPU
                # Max 1100 (intel-opencl-icd 25.05.32567.17), with -cl-std=CL2.0
                # and no relaxed-math option asked for. The f32 overload is
                # correct, and every other f64 operation preserves signed zero,
                # so this is narrowly those two builtins at double width.
                #
                # `floor` and `ceil` never change the sign of their operand --
                # for x > 0 the result is >= 0, for x < 0 it is <= x, and a zero
                # result keeps x's sign -- so restoring it from the operand is
                # exact for every input, including infinities. Where the runtime
                # is already correct this is a no-op, so it needs no device
                # check; drop it once Intel ships the fix.
                return f"copysign({call}, {args[0]})"
            return call

        raise NotImplementedError(f"OpenCL builtin: {node.func_name}")

    def _expr_binop(self, node: ir.IRBinOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        dtype = getattr(node, 'dtype', None)
        left = self._integers.convert(left, getattr(node.left, 'dtype', None), dtype)
        if node.op not in ('<<', '>>', '**'):
            right = self._integers.convert(right, getattr(node.right, 'dtype', None), dtype)
        if node.op == '/' and dtype in (f32, f64):
            t = _OCL_C_TYPE_MAP[dtype]
            return f'((({t})({left})) / (({t})({right})))'
        fixed = self._integers.operation(node.op, dtype, left, right)
        if fixed is not None:
            return fixed
        integer = integer_division_expr(
            node, left, right, _OCL_C_TYPE_MAP, self._integer_division_helpers)
        if integer is not None:
            return integer
        if node.op == "**":
            t = _OCL_C_TYPE_MAP[f64 if dtype is f64 else f32]
            return f'pow(({t})({left}), ({t})({right}))'
        if node.op in ('//', '%') and dtype is None:
            # Support the low-level generator API's legacy unannotated IR.
            operands = (self._infer_expr_type(node.left), self._infer_expr_type(node.right))
            dtype = f64 if 'double' in operands else f32 if 'float' in operands else None
        floating = float_division_expr(
            node, left, right, _OCL_C_TYPE_MAP, self._float_division_helpers, dtype=dtype)
        if floating is not None:
            return floating
        if node.op == "//":
            # Legacy unannotated integer IR; normal dispatch is fully typed.
            return f"({left} / {right})"
        if node.op in _BINOP_MAP:
            return f"({left} {_BINOP_MAP[node.op]} {right})"
        raise NotImplementedError(f"OpenCL binop: {node.op}")

    # --- Atomics (OpenCL syntax) ---

    def _emit_atomic_op(self, node: ir.IRAtomicOp):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_OCL_INT})({index}))"

        dtype = node.dtype
        value = self._integers.convert(value, getattr(node.value, 'dtype', None), dtype)
        value = f"(({_OCL_C_TYPE_MAP[dtype]})({value}))"
        if node.op == "min" and dtype is f32:
            self._needs_float_atomic_min = True
            self._emit(f"atomicMinFloat(&{field}[{index}], {value});")
        elif node.op == "max" and dtype is f32:
            self._needs_float_atomic_max = True
            self._emit(f"atomicMaxFloat(&{field}[{index}], {value});")
        elif dtype is f32:
            # Evaluate the contributed value once, outside retries.
            self._emit("{")
            self._indent += 1
            self._emit(f"float __val = {value};")
            self._emit(f"volatile __global atomic_uint* __addr = (volatile __global atomic_uint*)&{field}[{index}];")
            self._emit("uint __old = atomic_load_explicit(__addr, memory_order_relaxed, memory_scope_device);")
            self._emit("while (true) {")
            self._indent += 1
            self._emit("uint __next = as_uint(as_float(__old) + __val);")
            self._emit("if (atomic_compare_exchange_weak_explicit(__addr, &__old, __next, "
                       "memory_order_relaxed, memory_order_relaxed, memory_scope_device)) break;")
            self._indent -= 1
            self._emit("}")
            self._indent -= 1
            self._emit("}")
        else:
            atomic_type = 'atomic_uint' if dtype is u32 else 'atomic_int'
            self._emit(f"atomic_fetch_{node.op}_explicit("
                       f"(volatile __global {atomic_type}*)&{field}[{index}], {value}, "
                       "memory_order_relaxed, memory_scope_device);")

    # --- Field store/load (use 'long' for index casts) ---

    def _emit_field_store(self, node: ir.IRFieldStore):
        field = self._expr(node.field)
        index = self._expr(node.index)
        value = self._expr(node.value)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_OCL_INT})({index}))"
        self._emit(f"{field}[{index}] = {value};")

    def _expr_field_load(self, node: ir.IRFieldLoad) -> str:
        field = self._expr(node.field)
        index = self._expr(node.index)
        idx_type = self._infer_expr_type(node.index)
        if idx_type in ("float", "double"):
            index = f"(({_OCL_INT})({index}))"
        return f"{field}[{index}]"

    # --- Texture sampling: hardware image3d or software trilinear fallback ---

    def _expr_texture_sample(self, node: ir.IRTextureSample) -> str:
        W, H, D = node.shape
        u = self._expr(node.coords[0])
        v = self._expr(node.coords[1])
        w = self._expr(node.coords[2])

        if hasattr(self, '_texture_params') and node.field_name in self._texture_params:
            # Hardware path: read_imagef with coordinate transform.
            # Tack convention: texel centers at i/(N-1), u=0 → texel 0, u=1 → texel N-1.
            # OpenCL normalized+linear: texel centers at (i+0.5)/N.
            # Transform: ocl_u = (u * (N-1) + 0.5) / N
            ou = f"(({u}) * {W - 1}.0f + 0.5f) / {W}.0f"
            ov = f"(({v}) * {H - 1}.0f + 0.5f) / {H}.0f"
            ow = f"(({w}) * {D - 1}.0f + 0.5f) / {D}.0f"
            return (f"read_imagef({node.field_name}, __samp__, "
                    f"(float4)({ou}, {ov}, {ow}, 0.0f)).x")

        # Software trilinear fallback
        helper = f"__tex3d_linear_{W}_{H}_{D}__"
        if not hasattr(self, '_texture_helpers'):
            self._texture_helpers = {}
        self._texture_helpers[helper] = (W, H, D)
        return f"{helper}({node.field_name}, {u}, {v}, {w})"

    def _expr_cast(self, node) -> str:
        val = self._expr(node.value)
        converted = self._integers.convert(val, getattr(node.value, 'dtype', None), node.dtype)
        if converted != val:
            return converted
        if isinstance(node.dtype, ScalarType):
            c_type = _OCL_C_TYPE_MAP[node.dtype]
            return f"(({c_type})({val}))"
        if node.dtype == "int":
            return f"((int)({val}))"
        if node.dtype == "float":
            return f"((float)({val}))"
        raise NotImplementedError(f"OpenCL cast: {node.dtype}")

    # --- Type inference ---

    def _resolved_type_to_c(self, scalar_type) -> str:
        return _OCL_C_TYPE_MAP.get(scalar_type, "float")

    def _infer_c_type(self, node) -> str:
        """Get OpenCL C type for an IR expression node, using annotated dtype."""
        dtype = getattr(node, 'dtype', None)
        if dtype is not None:
            return self._resolved_type_to_c(dtype)
        rct = getattr(node, '_resolved_cast_type', None)
        if rct is not None:
            return self._resolved_type_to_c(rct)
        # Fallback for unannotated nodes (e.g., field pointer references)
        if isinstance(node, ir.IRName):
            if node.name in self._field_params:
                c_type = _OCL_C_TYPE_MAP[self._param_types[node.name]]
                return f"__global {c_type}*"
            if node.name in self._local_vars:
                return self._local_vars[node.name]
            if node.name in self._param_types:
                return _OCL_C_TYPE_MAP[self._param_types[node.name]]
            return _OCL_INT
        return "float"

    def _infer_expr_type(self, node) -> str:
        return self._infer_c_type(node)


def generate_opencl_source(ir_func: ir.IRFunction) -> str:
    """Generate OpenCL C source for a single kernel function."""
    codegen = OpenCLCodeGen(ir_func)
    return codegen.generate()

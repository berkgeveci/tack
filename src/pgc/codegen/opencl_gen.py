"""PGC OpenCL C code generation — transforms PGC IR to OpenCL C source.

Generates a ``__kernel`` function where:
  - Each Field parameter becomes a ``__global`` typed pointer
  - The outermost parallel for-loop maps to ``get_global_id(0)``
  - Sequential for-loops, while-loops, if/else map to standard C control flow
  - Math builtins map to OpenCL built-in math functions (overloaded, no 'f' suffix)

OpenCL C uses ``long`` for 64-bit integers (``long long`` is not standard OpenCL C).
This module reuses the CUDA codegen with OpenCL-specific overrides.
"""

from pgc.lang import ir
from pgc.codegen.cuda_gen import CUDACodeGen, _BINOP_MAP, _CMP_MAP
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64

# OpenCL uses 'long' for 64-bit integers (not 'long long')
_OCL_INT = "long"

_OCL_C_TYPE_MAP = {
    f32: "float",
    f64: "double",
    i32: "int",
    i64: "long",
    u32: "unsigned int",
    u64: "unsigned long",
}

_OCL_MATH_FUNCS = {
    "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan", "atan2": "atan2",
    "exp": "exp", "exp2": "exp2", "log": "log", "log2": "log2", "log10": "log10",
    "floor": "floor", "ceil": "ceil", "fabs": "fabs", "abs": "fabs",
    "pow": "pow",
}


class OpenCLCodeGen(CUDACodeGen):
    """Generates OpenCL C source from a PGC IR function.

    Reuses the CUDA codegen, overriding syntax that differs in OpenCL C:
    kernel qualifier, pointer qualifiers, thread indexing, math functions,
    atomics, shared memory, and barriers.
    """

    def generate(self) -> str:
        func = self.ir_func

        # Build parameter info
        for param in func.params:
            if param.type_annotation is None:
                raise TypeError(f"Parameter '{param.name}' has no type. Run type inference first.")
            self._param_types[param.name] = param.type_annotation
            if hasattr(param, '_is_field') and param._is_field:
                self._field_params.add(param.name)
            elif not hasattr(param, '_is_field'):
                self._field_params.add(param.name)

        # Build function signature with OpenCL qualifiers
        params_c = []
        for param in func.params:
            c_type = _OCL_C_TYPE_MAP[param.type_annotation]
            if param.name in self._field_params:
                params_c.append(f"__global {c_type}* restrict {param.name}")
            else:
                params_c.append(f"{c_type} {param.name}")
        params_c.append(f"{_OCL_INT} __n__")

        sig = ", ".join(params_c)
        self._emit(f"__kernel void {func.name}({sig}) {{")
        self._indent += 1
        self._emit_body(func.body)
        self._indent -= 1
        self._emit("}")

        # Prepend atomic helpers if needed
        prefix_lines = []
        if self._needs_float_atomic_min:
            prefix_lines.extend([
                "float atomicMinFloat(volatile __global float* addr, float val) {",
                "    volatile __global int* addr_as_int = (volatile __global int*)addr;",
                "    int old = *addr_as_int, assumed;",
                "    do {",
                "        assumed = old;",
                "        old = atomic_cmpxchg(addr_as_int, assumed,",
                "            as_int(fmin(val, as_float(assumed))));",
                "    } while (assumed != old);",
                "    return as_float(old);",
                "}",
                "",
            ])
        if self._needs_float_atomic_max:
            prefix_lines.extend([
                "float atomicMaxFloat(volatile __global float* addr, float val) {",
                "    volatile __global int* addr_as_int = (volatile __global int*)addr;",
                "    int old = *addr_as_int, assumed;",
                "    do {",
                "        assumed = old;",
                "        old = atomic_cmpxchg(addr_as_int, assumed,",
                "            as_int(fmax(val, as_float(assumed))));",
                "    } while (assumed != old);",
                "    return as_float(old);",
                "}",
                "",
            ])

        # Header pragmas
        header = ""
        uses_f64 = any(p.type_annotation in (f64,) for p in func.params)
        if uses_f64:
            header += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"

        # Texture sampling helpers (software trilinear)
        if hasattr(self, '_texture_helpers') and self._texture_helpers:
            for name, (W, H, D) in self._texture_helpers.items():
                prefix_lines.extend([
                    f"inline float {name}(__global float* data, float u, float v, float w) {{",
                    f"    float fx = u * {W - 1}.0f, fy = v * {H - 1}.0f, fz = w * {D - 1}.0f;",
                    f"    {_OCL_INT} ix = ({_OCL_INT})floor(fx), iy = ({_OCL_INT})floor(fy), iz = ({_OCL_INT})floor(fz);",
                    f"    float dx = fx - (float)ix, dy = fy - (float)iy, dz = fz - (float)iz;",
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
                    f"    float c00 = c000 * (1.0f - dx) + c100 * dx;",
                    f"    float c10 = c010 * (1.0f - dx) + c110 * dx;",
                    f"    float c01 = c001 * (1.0f - dx) + c101 * dx;",
                    f"    float c11 = c011 * (1.0f - dx) + c111 * dx;",
                    f"    float c0 = c00 * (1.0f - dy) + c10 * dy;",
                    f"    float c1 = c01 * (1.0f - dy) + c11 * dy;",
                    f"    return c0 * (1.0f - dz) + c1 * dz;",
                    f"}}",
                    f"",
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
        if var not in self._declared_vars:
            self._emit(f"for ({_OCL_INT} {var} = {start}; {var} < {end}; {incr}) {{")
            self._local_vars[var] = _OCL_INT
            self._declared_vars.add(var)
        else:
            self._emit(f"for ({var} = {start}; {var} < {end}; {incr}) {{")
        self._indent += 1
        self._emit_body(node.body)
        self._indent -= 1
        self._emit("}")

    # --- Shared memory, barrier, thread ID ---

    def _emit_stmt(self, node):
        if isinstance(node, ir.IRSharedAlloc):
            self._emit(f"__local {node.dtype} {node.name}[{self._expr(node.size)}];")
        elif isinstance(node, ir.IRBarrier):
            self._emit("barrier(CLK_LOCAL_MEM_FENCE);")
        else:
            super()._emit_stmt(node)

    def _expr(self, node) -> str:
        if isinstance(node, ir.IRThreadId):
            return "get_local_id(0)"
        return super()._expr(node)

    # --- Math functions (overloaded, no 'f' suffix) ---

    def _expr_call(self, node: ir.IRCall) -> str:
        args = [self._expr(a) for a in node.args]

        if node.func_name == "min" and len(args) == 2:
            return f"fmin({args[0]}, {args[1]})"
        if node.func_name == "max" and len(args) == 2:
            return f"fmax({args[0]}, {args[1]})"

        if node.func_name in _OCL_MATH_FUNCS:
            func = _OCL_MATH_FUNCS[node.func_name]
            return f"{func}({', '.join(args)})"

        raise NotImplementedError(f"OpenCL builtin: {node.func_name}")

    def _expr_binop(self, node: ir.IRBinOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        if node.op == "**":
            return f"pow({left}, {right})"
        if node.op == "//":
            lt = self._infer_expr_type(node.left)
            rt = self._infer_expr_type(node.right)
            if lt not in ("float", "double") and rt not in ("float", "double"):
                return f"({left} / {right})"
            return f"({_OCL_INT})floor((float)({left}) / (float)({right}))"
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

        field_name = self._get_field_name(node.field)
        field_type = _OCL_C_TYPE_MAP.get(self._param_types.get(field_name)) if field_name else None
        is_float = field_type in ("float", "double")

        if node.op == "min" and is_float:
            self._needs_float_atomic_min = True
            self._emit(f"atomicMinFloat(&{field}[{index}], {value});")
        elif node.op == "max" and is_float:
            self._needs_float_atomic_max = True
            self._emit(f"atomicMaxFloat(&{field}[{index}], {value});")
        elif node.op == "add" and is_float:
            # CAS-based float atomic add
            self._emit(f"{{ volatile __global int* __addr = (volatile __global int*)&{field}[{index}];")
            self._emit(f"  int __old = *__addr, __assumed;")
            self._emit(f"  do {{ __assumed = __old;")
            self._emit(f"    __old = atomic_cmpxchg(__addr, __assumed, as_int(as_float(__assumed) + {value}));")
            self._emit(f"  }} while (__assumed != __old); }}")
        else:
            _ATOMIC_FUNCS = {"add": "atomic_add", "min": "atomic_min", "max": "atomic_max"}
            func = _ATOMIC_FUNCS.get(node.op)
            if func is None:
                raise NotImplementedError(f"OpenCL atomic op: {node.op}")
            self._emit(f"{func}(&{field}[{index}], {value});")

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

    # --- Texture sampling: software trilinear with OpenCL qualifiers ---

    def _expr_texture_sample(self, node: ir.IRTextureSample) -> str:
        W, H, D = node.shape
        u = self._expr(node.coords[0])
        v = self._expr(node.coords[1])
        w = self._expr(node.coords[2])
        helper = f"__tex3d_linear_{W}_{H}_{D}__"
        if not hasattr(self, '_texture_helpers'):
            self._texture_helpers = {}
        self._texture_helpers[helper] = (W, H, D)
        return f"{helper}({node.field_name}, {u}, {v}, {w})"

    # --- Type inference: map 'long long' → 'long' ---

    def _resolved_type_to_c(self, scalar_type) -> str:
        return _OCL_C_TYPE_MAP.get(scalar_type, "float")

    def _infer_c_type(self, node) -> str:
        result = super()._infer_c_type(node)
        return result.replace("long long", "long").replace("unsigned long long", "unsigned long")

    def _infer_expr_type(self, node) -> str:
        result = super()._infer_expr_type(node)
        return result.replace("long long", "long").replace("unsigned long long", "unsigned long")


def generate_opencl_source(ir_func: ir.IRFunction) -> str:
    """Generate OpenCL C source for a single kernel function."""
    codegen = OpenCLCodeGen(ir_func)
    return codegen.generate()

"""PGC CPU backend — JIT compiles kernels via llvmlite and runs with thread pool.

The kernel function signature is:
    void kernel(field0_ptr, field1_ptr, ..., i64 loop_start, i64 loop_end)

Fields are passed as ctypes pointers to their underlying numpy data.
The loop range is split across threads for parallel execution.
"""

import ctypes
import ctypes.util
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llvmlite import binding as llvm

from pgc.lang import ir
from pgc.lang.field import Field, NumpyBuffer
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.llvm_gen import generate_llvm_ir


# ── NUMA interleave support ──────────────────────────────────────────
# If libnuma is available, we wrap numpy allocations with MPOL_INTERLEAVE
# so pages are spread across all NUMA nodes.  This avoids the pathological
# case where all memory lands on one node and remote threads pay 2x latency.

_numa_available = False
_numa_node_mask = 0
_numa_max_node = 0
_libc = None
_SYS_SET_MEMPOLICY = 238  # x86_64 syscall number
_MPOL_DEFAULT = 0
_MPOL_INTERLEAVE = 5


def _init_numa():
    global _numa_available, _numa_node_mask, _numa_max_node, _libc
    try:
        numa = ctypes.CDLL(ctypes.util.find_library("numa"))
        if numa.numa_available() == -1:
            return
        max_node = numa.numa_max_node()
        # Only include nodes that actually have memory
        nodes_with_mem = 0
        mask = 0
        for n in range(max_node + 1):
            try:
                with open(f"/sys/devices/system/node/node{n}/meminfo") as f:
                    for line in f:
                        if "MemTotal" in line:
                            kb = int(line.split()[-2])
                            if kb > 0:
                                mask |= (1 << n)
                                nodes_with_mem += 1
                            break
            except (OSError, ValueError):
                continue
        if nodes_with_mem < 2:
            return  # single memory node, interleave won't help
        _numa_node_mask = mask
        _numa_max_node = max_node
        _libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        _numa_available = True
    except (OSError, AttributeError):
        pass

_init_numa()


class _NumaInterleave:
    """Context manager: set MPOL_INTERLEAVE for allocations, restore on exit."""

    def __enter__(self):
        if not _numa_available:
            return self
        mask = (ctypes.c_ulong * 1)(_numa_node_mask)
        _libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_INTERLEAVE, mask,
                      _numa_max_node + 2)
        return self

    def __exit__(self, *exc):
        if not _numa_available:
            return
        _libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_DEFAULT, None, 0)

# Initialize LLVM native target (required before JIT compilation)
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


# ctypes type for each PGC scalar type (used for pointer casting)
_CTYPES_MAP = {
    f32: ctypes.c_float,
    f64: ctypes.c_double,
    i32: ctypes.c_int32,
    i64: ctypes.c_int64,
    u32: ctypes.c_uint32,
    u64: ctypes.c_uint64,
}


def _detect_template_args(kernel, args) -> dict[int, tuple[str, object]]:
    """Detect which arguments are @pgc.data_oriented template objects.

    Returns dict: param_index -> (param_name, template_object)
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    templates = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if hasattr(arg, '_data_oriented') and arg._data_oriented:
            templates[i] = (param_name, arg)
    return templates


def _expand_template_args(args, template_args) -> tuple:
    """Replace template args with their field attributes.

    Returns new args tuple with template objects removed and their
    field attributes appended.  Fields are appended in reverse template
    index order to match the AST rewrite pass (which processes templates
    from highest index to lowest).
    """
    if not template_args:
        return args

    from pgc.lang.template_rewrite import classify_template_attrs

    new_args = []
    extra_fields = []
    # Collect template fields in reverse index order (matching rewrite order)
    for idx in sorted(template_args.keys(), reverse=True):
        _, obj = template_args[idx]
        _, fields = classify_template_attrs(obj)
        for attr_name in sorted(fields.keys()):
            extra_fields.append(fields[attr_name])
    # Build non-template args in order
    for i, arg in enumerate(args):
        if i not in template_args:
            new_args.append(arg)
    new_args.extend(extra_fields)
    return tuple(new_args)


def _detect_vector_fields_from_args(kernel, args, template_args) -> dict[str, int] | None:
    """Detect vector fields, accounting for template parameters.

    When template args are present, we need to skip them when matching
    parameter names to arguments.
    """
    if not template_args:
        return _detect_vector_fields(kernel, args)

    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_fields = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if i in template_args:
            continue
        if isinstance(arg, Field) and hasattr(arg, '_vector_n'):
            vector_fields[param_name] = arg._vector_n
    return vector_fields if vector_fields else None


def _detect_vector_fields(kernel, args) -> dict[str, int] | None:
    """Detect which kernel parameters are vector fields.

    Returns a dict mapping parameter names to component counts,
    or None if no vector fields are present.
    """
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    vector_fields = {}
    for param_name, arg in zip(params, args):
        if isinstance(arg, Field) and hasattr(arg, '_vector_n'):
            vector_fields[param_name] = arg._vector_n
    return vector_fields if vector_fields else None


def _detect_texture_fields(kernel, args, template_args=None) -> dict[str, tuple] | None:
    """Detect which kernel parameters are Texture3D objects."""
    from pgc.lang.field import Texture3D
    funcdef = kernel._funcdef
    params = [a.arg for a in funcdef.args.args]
    texture_fields = {}
    for i, (param_name, arg) in enumerate(zip(params, args)):
        if template_args and i in template_args:
            continue
        if isinstance(arg, Texture3D):
            texture_fields[param_name] = arg.shape_3d
    return texture_fields if texture_fields else None


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments.

    Resolves the loop end expression — supports:
      - IRConstant(N)
      - IRFieldLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))  →  x.shape[0]
      - IRAttribute(IRName("x"), "__len__")  →  len(x)
    """
    # Find the top-level parallel for
    parallel_for = None
    for stmt in ir_func.body:
        if isinstance(stmt, ir.IRParallelFor):
            parallel_for = stmt
            break

    if parallel_for is None:
        raise RuntimeError("Kernel has no parallel for-loop")

    # Build name → arg mapping
    name_to_arg = {}
    for param, arg in zip(ir_func.params, args):
        name_to_arg[param.name] = arg

    return _resolve_range_expr(parallel_for.end, name_to_arg)


def _resolve_range_expr(node: ir.IRNode, name_to_arg: dict) -> int:
    """Resolve a range expression to a concrete integer value."""
    if isinstance(node, ir.IRConstant):
        return int(node.value)

    # x.shape[0]  →  IRFieldLoad(IRAttribute(IRName("x"), "shape"), IRConstant(0))
    if isinstance(node, ir.IRFieldLoad):
        obj = node.field
        if isinstance(obj, ir.IRAttribute) and obj.attr == "shape":
            if isinstance(obj.obj, ir.IRName):
                arg = name_to_arg.get(obj.obj.name)
                if isinstance(arg, Field):
                    idx = _resolve_range_expr(node.index, name_to_arg)
                    return arg.shape[idx]

    # len(x)  →  IRAttribute(IRName("x"), "__len__")
    if isinstance(node, ir.IRAttribute) and node.attr == "__len__":
        if isinstance(node.obj, ir.IRName):
            arg = name_to_arg.get(node.obj.name)
            if isinstance(arg, Field):
                return arg.shape[0]

    # Binary ops on range expressions (e.g., n - 1)
    if isinstance(node, ir.IRBinOp):
        left = _resolve_range_expr(node.left, name_to_arg)
        right = _resolve_range_expr(node.right, name_to_arg)
        ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
               "*": lambda a, b: a * b, "//": lambda a, b: a // b}
        if node.op in ops:
            return ops[node.op](left, right)

    # Plain name reference (e.g., `n` passed as scalar)
    if isinstance(node, ir.IRName):
        arg = name_to_arg.get(node.name)
        if arg is not None:
            return int(arg)

    raise RuntimeError(f"Cannot resolve loop range expression: {type(node).__name__}")


class CompiledKernel:
    """A JIT-compiled kernel ready for execution."""

    def __init__(self, engine, func_ptr, param_types, param_is_field, func_name):
        self._engine = engine  # prevent GC
        self._func_ptr = func_ptr
        self._param_types = param_types
        self._param_is_field = param_is_field  # list[bool]
        self._func_name = func_name

        # Build the ctypes function type
        ctypes_params = []
        for ptype, is_field in zip(param_types, param_is_field):
            ct = _CTYPES_MAP[ptype]
            if is_field:
                ctypes_params.append(ctypes.POINTER(ct))
            else:
                ctypes_params.append(ct)
        ctypes_params.extend([ctypes.c_int64, ctypes.c_int64])

        self._cfunc_type = ctypes.CFUNCTYPE(None, *ctypes_params)
        self._cfunc = self._cfunc_type(func_ptr)

    def __call__(self, kernel_args: list, loop_start: int, loop_end: int):
        """Call the compiled kernel with field pointers/scalars and loop range."""
        ctypes_args = []
        for arg, ptype, is_field in zip(kernel_args, self._param_types, self._param_is_field):
            ct = _CTYPES_MAP[ptype]
            if is_field:
                ptr = arg._buffer._data.ctypes.data_as(ctypes.POINTER(ct))
                ctypes_args.append(ptr)
            else:
                ctypes_args.append(ct(arg))
        ctypes_args.extend([ctypes.c_int64(loop_start), ctypes.c_int64(loop_end)])
        self._cfunc(*ctypes_args)


def _create_target_machine():
    """Create a target machine for the host CPU with full feature support."""
    target = llvm.Target.from_default_triple()
    cpu = llvm.get_host_cpu_name()
    features = llvm.get_host_cpu_features().flatten()
    return target.create_target_machine(cpu=cpu, features=features, opt=3)


def _optimize_module(mod, target_machine):
    """Run LLVM optimization passes including loop vectorization."""
    from llvmlite.binding.newpassmanagers import create_pipeline_tuning_options

    pto = create_pipeline_tuning_options(3)  # O3
    pto.loop_vectorization = True
    pto.slp_vectorization = True
    pto.loop_unrolling = True
    pto.loop_interleaving = True

    pb = llvm.create_pass_builder(target_machine, pto)
    pm = pb.getModulePassManager()
    pm.run(mod, pb)


def _compile_kernel(ir_func: ir.IRFunction) -> CompiledKernel:
    """JIT-compile a PGC IR function to native code via llvmlite."""
    # Generate LLVM IR
    llvm_module = generate_llvm_ir(ir_func)
    llvm_ir_str = str(llvm_module)

    # Parse and verify
    mod = llvm.parse_assembly(llvm_ir_str)
    mod.verify()

    # Run optimization passes (loop vectorization, unrolling, etc.)
    # Uses its own target machine instance since MCJIT takes ownership of one
    tm_opt = _create_target_machine()
    _optimize_module(mod, tm_opt)

    # Create execution engine with a fresh target machine
    tm_jit = _create_target_machine()
    engine = llvm.create_mcjit_compiler(mod, tm_jit)

    # Get function pointer
    func_ptr = engine.get_function_address(ir_func.name)
    if func_ptr == 0:
        raise RuntimeError(f"Failed to JIT compile kernel '{ir_func.name}'")

    param_types = [p.type_annotation for p in ir_func.params]
    param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
    return CompiledKernel(engine, func_ptr, param_types, param_is_field, ir_func.name)


def _physical_core_count() -> int:
    """Return the number of physical CPU cores (not hyperthreads)."""
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            threads_per_core = len(f.read().strip().split(","))
        total = os.cpu_count() or 1
        return max(1, total // threads_per_core)
    except (OSError, ValueError):
        return os.cpu_count() or 1


class CPUBackend:
    """CPU backend — JIT compiles kernels and runs them with thread parallelism."""

    def __init__(self, num_threads: int | None = None):
        self.num_threads = num_threads or _physical_core_count()
        self._cache: dict[str, CompiledKernel] = {}
        self._pool: ThreadPoolExecutor | None = None

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> NumpyBuffer:
        if _numa_available:
            # Use np.empty (no page faults) under interleave policy,
            # so the first write spreads pages across NUMA nodes.
            with _NumaInterleave():
                buf = NumpyBuffer.__new__(NumpyBuffer)
                buf._data = np.empty(shape, dtype=dtype.numpy_dtype)
                # Force page faults under interleave policy by touching all pages
                buf._data.fill(0)
            return buf
        return NumpyBuffer(dtype.numpy_dtype, shape)

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the CPU.

        1. Detect template and vector fields, get specialized IR
        2. Resolve dimension sizes from actual field shapes
        3. Run type inference from actual arguments
        4. JIT-compile (or use cached version)
        5. Determine loop range
        6. Split range across threads and execute
        """
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        # Detect template arguments and expand them
        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        # Detect vector and texture fields from effective arguments
        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
        texture_fields = _detect_texture_fields(kernel, args, template_args)

        # Get IR (re-transforms if vector/template/texture fields present)
        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
            texture_fields=texture_fields,
        )
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes and texture shapes using actual field shapes.
        # For Texture3D args, register the Texture3D itself so ir_resolve can
        # embed shape_3d into IRTextureSample nodes.
        from pgc.lang.field import Texture3D
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference (Texture3D is handled — sets _is_texture flag)
        infer_param_types(ir_func, effective_args)

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Cache key: kernel name + argument type signature + template info
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            self._cache[cache_key] = _compile_kernel(ir_func)

        compiled = self._cache[cache_key]

        # Build kernel args list — unwrap Texture3D to underlying Field for dispatch
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]

        # Determine loop range (use kernel_args which has Fields, not Texture3D)
        loop_end = _get_loop_range(ir_func, kernel_args)

        # Parallel execution: split range across threads
        if self.num_threads <= 1 or loop_end <= 1024:
            compiled(kernel_args, 0, loop_end)
        else:
            self._parallel_execute(compiled, kernel_args, 0, loop_end)

    def _get_pool(self) -> ThreadPoolExecutor:
        """Return the persistent thread pool, creating it on first use."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.num_threads)
        return self._pool

    def _parallel_execute(self, compiled: CompiledKernel,
                          field_args: list[Field],
                          start: int, end: int):
        """Split the loop range across threads."""
        total = end - start
        chunk = (total + self.num_threads - 1) // self.num_threads

        pool = self._get_pool()
        futures = []
        for t in range(self.num_threads):
            t_start = start + t * chunk
            t_end = min(t_start + chunk, end)
            if t_start >= end:
                break
            futures.append(
                pool.submit(compiled, field_args, t_start, t_end)
            )
        for f in futures:
            f.result()  # propagate exceptions

"""Tack CPU backend — JIT compiles kernels via llvmlite and runs with thread pool.

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

from tack.lang import ir
from tack.lang.field import Field, NumpyBuffer
from tack.lang.types import ScalarType, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64
from tack.lang.type_inference import infer_param_types, check_dispatch_types

_CPU_SUPPORTED_DTYPES = {i8, u8, i16, u16, i32, u32, i64, u64, f32, f64}
from tack.codegen.llvm_gen import generate_llvm_ir


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
    except (OSError, AttributeError, TypeError):
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


# ctypes type for each Tack scalar type (used for pointer casting)
_CTYPES_MAP = {
    i8:  ctypes.c_int8,
    u8:  ctypes.c_uint8,
    i16: ctypes.c_int16,
    u16: ctypes.c_uint16,
    i32: ctypes.c_int32,
    u32: ctypes.c_uint32,
    i64: ctypes.c_int64,
    u64: ctypes.c_uint64,
    f32: ctypes.c_float,
    f64: ctypes.c_double,
}


# Re-export shared utilities so existing `from tack.runtime.cpu import ...` works
from tack.runtime.kernel_utils import (  # noqa: F401
    _detect_template_args,
    _expand_template_args,
    _detect_vector_fields,
    _detect_vector_fields_from_args,
    _detect_texture_fields,
    _create_pack_fields,
    _update_pack_fields,
    _get_loop_range,
    _resolve_range_expr,
)


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
    """JIT-compile a Tack IR function to native code via llvmlite."""
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

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> NumpyBuffer:
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

    def memory_space(self, ptr) -> str:
        """CPU backend: all pointers are host memory."""
        return "cpu"

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing pointer as a NumpyBuffer without copying.

        Args:
            ptr: integer memory address or numpy array.
        """
        import ctypes
        buf = NumpyBuffer.__new__(NumpyBuffer)
        if isinstance(ptr, np.ndarray):
            # Wrap existing numpy array (shares memory, no copy)
            buf._data = ptr.view(dtype.numpy_dtype).reshape(shape)
        else:
            # Wrap raw C pointer as numpy array
            ct = np.ctypeslib.as_array(
                (ctypes.c_char * (int(np.prod(shape)) * np.dtype(dtype.numpy_dtype).itemsize))
                .from_address(int(ptr)))
            buf._data = np.frombuffer(ct, dtype=dtype.numpy_dtype).reshape(shape)
        return buf

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
        from tack.lang.field import Texture3D
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from tack.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference and dispatch-time type checking
        infer_param_types(ir_func, effective_args)
        check_dispatch_types(ir_func, effective_args,
                             supported_dtypes=_CPU_SUPPORTED_DTYPES,
                             backend_name="CPU")

        # Optimization passes (LICM, CSE)
        from tack.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Type annotation (sets dtype on all expression nodes — needed by
        # LLVM codegen for signed/unsigned distinction on casts)
        from tack.lang.ir_type_annotate import annotate_types
        annotate_types(ir_func)

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

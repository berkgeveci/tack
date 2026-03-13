"""PGC CPU backend — JIT compiles kernels via llvmlite and runs with thread pool.

The kernel function signature is:
    void kernel(field0_ptr, field1_ptr, ..., i64 loop_start, i64 loop_end)

Fields are passed as ctypes pointers to their underlying numpy data.
The loop range is split across threads for parallel execution.
"""

import ctypes
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llvmlite import binding as llvm

from pgc.lang import ir
from pgc.lang.field import Field
from pgc.lang.types import f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.llvm_gen import generate_llvm_ir

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

    def __init__(self, engine, func_ptr, param_types, func_name):
        self._engine = engine  # prevent GC
        self._func_ptr = func_ptr
        self._param_types = param_types
        self._func_name = func_name

        # Build the ctypes function type
        # params: one pointer per field, then i64 loop_start, i64 loop_end
        ctypes_params = []
        for ptype in param_types:
            ct = _CTYPES_MAP[ptype]
            ctypes_params.append(ctypes.POINTER(ct))
        ctypes_params.extend([ctypes.c_int64, ctypes.c_int64])

        self._cfunc_type = ctypes.CFUNCTYPE(None, *ctypes_params)
        self._cfunc = self._cfunc_type(func_ptr)

    def __call__(self, field_args: list[Field], loop_start: int, loop_end: int):
        """Call the compiled kernel with field pointers and loop range."""
        ctypes_args = []
        for field, ptype in zip(field_args, self._param_types):
            ct = _CTYPES_MAP[ptype]
            ptr = field.data.ctypes.data_as(ctypes.POINTER(ct))
            ctypes_args.append(ptr)
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
    return CompiledKernel(engine, func_ptr, param_types, ir_func.name)


class CPUBackend:
    """CPU backend — JIT compiles kernels and runs them with thread parallelism."""

    def __init__(self, num_threads: int | None = None):
        self.num_threads = num_threads or os.cpu_count() or 1
        self._cache: dict[str, CompiledKernel] = {}

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the CPU.

        1. Run type inference from actual arguments
        2. JIT-compile (or use cached version)
        3. Determine loop range
        4. Split range across threads and execute
        """
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        ir_module = kernel._ir
        ir_func = ir_module.functions[0]

        # Type inference
        infer_param_types(ir_func, args)

        # Cache key: kernel name + argument type signature
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}"

        if cache_key not in self._cache:
            self._cache[cache_key] = _compile_kernel(ir_func)

        compiled = self._cache[cache_key]

        # Extract field arguments
        field_args = []
        for arg in args:
            if isinstance(arg, Field):
                field_args.append(arg)
            else:
                raise NotImplementedError(
                    "Scalar kernel arguments not yet supported in JIT mode. "
                    "Use constants in the kernel body instead."
                )

        # Determine loop range
        loop_end = _get_loop_range(ir_func, args)

        # Parallel execution: split range across threads
        # ThreadPoolExecutor overhead is ~0.1ms per dispatch, so only
        # parallelize when the workload is large enough to amortize it.
        # TODO: use C-level pthreads for lower dispatch overhead.
        if self.num_threads <= 1 or loop_end <= 4_000_000:
            # Single-threaded for small workloads
            compiled(field_args, 0, loop_end)
        else:
            self._parallel_execute(compiled, field_args, 0, loop_end)

    def _parallel_execute(self, compiled: CompiledKernel,
                          field_args: list[Field],
                          start: int, end: int):
        """Split the loop range across threads."""
        total = end - start
        chunk = (total + self.num_threads - 1) // self.num_threads

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for t in range(self.num_threads):
                t_start = start + t * chunk
                t_end = min(t_start + chunk, end)
                if t_start >= end:
                    break
                futures.append(
                    executor.submit(compiled, field_args, t_start, t_end)
                )
            for f in futures:
                f.result()  # propagate exceptions

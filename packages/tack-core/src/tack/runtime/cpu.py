"""Tack CPU backend — JIT compiles kernels via llvmlite and runs with thread pool.

The kernel function signature is:
    void kernel(field0_ptr, field1_ptr, ..., i64 loop_start, i64 loop_end)

Fields are passed as ctypes pointers to their underlying numpy data.
The loop range is split across threads for parallel execution.

Threading decision
------------------
A fan-out costs ~200 µs of pure Python thread wakeup on a 10-core machine,
so it only pays when the serial run would take meaningfully longer than
that.  Both sides of the comparison are *measured* rather than assumed:
the backend calibrates the fan-out cost once, and every compiled kernel
carries a running estimate of its serial nanoseconds per element.

A fixed element-count threshold cannot do this job, because the crossover
moves by three orders of magnitude with the kernel's arithmetic intensity.
Measured here (10 physical cores, f32):

    out[i] = x[i]*2 + 1          memory bound     crossover ~4,000,000
    out[i] = sqrt(..) + sin(..)  ~10 flop/elem    crossover   ~130,000
    20-iteration inner loop      ~120 flop/elem   crossover     ~4,000

The old constant of 1024 sat below all three, so every mid-size dispatch
of a cheap kernel paid ~200 µs to save ~20 µs — a 3-10x pessimization.
"""

import ctypes
import ctypes.util
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llvmlite import binding as llvm

from tack.lang import ir
from tack.lang.field import Field, NumpyBuffer
from tack.lang.types import ScalarType, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64

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


from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    new_kernel_cache,
    resolve_variant,
)

# Re-export shared utilities so existing `from tack.runtime.cpu import ...` works.
# Backends must NOT rely on this — importing this module pulls in llvmlite, which
# is a CPU-only dependency.  Import from tack.runtime.kernel_utils instead.
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


# Thread only on a real win: parallel takes overhead + T_serial/P, so the
# break-even is already P/(P-1); the rest is margin against a mis-estimate.
_PARALLEL_BREAK_EVEN = 2.0

# Weight of the newest sample in the per-kernel cost estimate. Low enough
# to ride out ordinary jitter, high enough to track a kernel whose cost
# depends on its data.
_COST_SMOOTHING = 0.25

# Most a single sample may claim, as a multiple of the running estimate.
# Smoothing alone does not contain an outlier: a dispatch preempted mid-run
# can time a thousand times the kernel's real cost, and a quarter of that is
# still enough to flip the decision. A kernel whose cost genuinely rises
# still converges within a few dispatches.
_MAX_SAMPLE_RATIO = 8.0

# Smallest slice worth timing when refreshing the estimate mid-fan-out.
# Below this the whole range is timed instead: a range that small cannot
# repay a fan-out anyway, so there is nothing to protect by splitting it.
_RECHECK_PROBE_MIN = 4096

# Parallel dispatches to allow between refreshes of the cost estimate.
# Doubles up to the cap, so a wrong decision to thread is caught after a
# single dispatch, while a kernel that really wants threads settles into
# one serial run per cap.
_RECHECK_CAP = 1024

# Below this, the first dispatch of an unmeasured kernel just runs serially
# rather than splitting off a timing probe — a range this small cannot
# repay a fan-out even for the most expensive kernel body.
_PROBE_MIN_RANGE = 16384

_CALIBRATION_REPS = 5

# Stand-in fan-out cost until the real one is measured. Pessimistic on
# purpose: being too high only delays the first fan-out by one dispatch,
# whereas being too low fans out work that would have been faster serial.
_DEFAULT_FAN_OUT_NS = 200_000.0

# "Do not thread this" — larger than any realistic loop range.
_NEVER = 1 << 62


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

        # Measured serial nanoseconds per element, updated on every serial
        # dispatch. 0.0 means "not measured yet".
        self.ns_per_elem = 0.0
        # Range size at which a fan-out starts paying for itself, derived
        # from ns_per_elem. Precomputed so the dispatch hot path is a single
        # integer compare. _NEVER until the kernel has been timed once.
        self.parallel_min_elems = _NEVER

        # Only serial runs measure, so the decision to thread rests on an
        # estimate that threading itself stops refreshing. Left alone that
        # is a one-way door: one mistimed sample turns threading on, and
        # nothing afterwards can discover it was wrong. These schedule an
        # occasional serial run to re-measure.
        self.parallel_since_measure = 0
        self.recheck_after = 1

    def recheck_due(self) -> bool:
        """Whether this parallel dispatch should re-measure instead.

        Backs off geometrically: the first wrong fan-out is caught
        immediately, while a kernel that genuinely wants threads converges
        to one serial run per `_RECHECK_CAP`.
        """
        self.parallel_since_measure += 1
        if self.parallel_since_measure < self.recheck_after:
            return False
        self.parallel_since_measure = 0
        self.recheck_after = min(self.recheck_after * 2, _RECHECK_CAP)
        return True

    def bind(self, kernel_args: list) -> tuple:
        """Marshal the field pointers and scalars once for a dispatch.

        Only loop_start/loop_end differ between the chunks of one dispatch,
        so doing this per chunk re-derived every pointer under the GIL —
        about 3.5 µs a chunk, against a 4.6 µs serial dispatch. Pass the
        result to `call_range` for each chunk.
        """
        prefix = []
        for arg, ptype, is_field in zip(kernel_args, self._param_types,
                                        self._param_is_field):
            ct = _CTYPES_MAP[ptype]
            if is_field:
                prefix.append(arg._buffer._data.ctypes.data_as(ctypes.POINTER(ct)))
            else:
                prefix.append(ct(arg))
        return tuple(prefix)

    def call_range(self, prefix: tuple, loop_start: int, loop_end: int):
        """Run one chunk. Fresh c_int64s — chunks run concurrently."""
        self._cfunc(*prefix, ctypes.c_int64(loop_start), ctypes.c_int64(loop_end))

    def __call__(self, kernel_args: list, loop_start: int, loop_end: int):
        """Call the compiled kernel with field pointers/scalars and loop range."""
        self.call_range(self.bind(kernel_args), loop_start, loop_end)


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


class CPUBackend(Backend):
    """CPU backend — JIT compiles kernels and runs them with thread parallelism."""

    name = "cpu"
    display_name = "CPU"
    supported_dtypes = _CPU_SUPPORTED_DTYPES
    # Reductions go through numpy on the host — the data is already there.
    supports_device_reductions = False


    def __init__(self, num_threads: int | None = None):
        if num_threads is None:
            env = os.environ.get("TACK_CPU_THREADS")
            num_threads = int(env) if env else _physical_core_count()
        self.num_threads = max(1, num_threads)
        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledKernel}
        self._pool: ThreadPoolExecutor | None = None
        # Cost of a fan-out on this machine, measured on first use.
        self._fan_out_ns: float | None = None

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

        The IR passes and the JIT run only when this argument shape/type
        combination is new; see `resolve_variant`. A repeat dispatch resolves
        the loop range, marshals the arguments, and runs.
        """
        from tack.lang.field import Texture3D

        variant, effective_args = resolve_variant(
            self, kernel, args, kwargs,
            build=self._build_variant,
        )

        # Unwrap Texture3D to the underlying Field for dispatch
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(variant.ir, kernel_args)

        self._dispatch(variant.payload, kernel_args, loop_end)

    @staticmethod
    def _build_variant(ir_func, effective_args):
        """Annotate types and JIT-compile. Runs once per variant."""
        from tack.lang.ir_type_annotate import annotate_types
        annotate_types(ir_func)
        return _compile_kernel(ir_func)

    # ── Dispatch: serial or fan-out ──────────────────────────────────

    def _dispatch(self, compiled: CompiledKernel, kernel_args: list,
                  loop_end: int):
        """Run the loop range, threading it only when that is faster."""
        prefix = compiled.bind(kernel_args)

        if loop_end >= compiled.parallel_min_elems:
            if self._fan_out_ns is None:
                # First range big enough to want threads: measure what they
                # actually cost here, then re-check against the real number.
                self._calibrate_fan_out(compiled, prefix)
                compiled.parallel_min_elems = self._min_elems(compiled.ns_per_elem)
                if loop_end < compiled.parallel_min_elems:
                    self._run_serial(compiled, prefix, 0, loop_end)
                    return
            if compiled.recheck_due():
                # Refresh the estimate on a slice, then fan out the rest.
                # The estimate that chose threading cannot be checked
                # against itself — a wildly high one predicts an even worse
                # serial run, so fanning out looks like a win however slow
                # it actually is — so the only way to find out is to time
                # one. Taking a slice rather than the whole range keeps that
                # from costing the dispatch its parallelism; small ranges,
                # where a fan-out is the expensive option anyway, run whole.
                probe_end = min(loop_end,
                                max(_RECHECK_PROBE_MIN, loop_end // 64))
                self._run_serial(compiled, prefix, 0, probe_end)
                if loop_end > probe_end:
                    self._parallel_execute(compiled, prefix, probe_end,
                                           loop_end)
                return
            self._parallel_execute(compiled, prefix, 0, loop_end)
            return

        if compiled.ns_per_elem > 0.0 or loop_end < _PROBE_MIN_RANGE \
                or self.num_threads <= 1:
            # Already measured and too small, or too small to be worth
            # splitting off a probe.
            self._run_serial(compiled, prefix, 0, loop_end)
            return

        # First sight of this kernel at a range where threading might pay.
        # Time a small prefix, then decide about the rest — so a one-shot
        # large dispatch is not stuck running serially for want of a sample.
        probe_end = max(_PROBE_MIN_RANGE // 16, loop_end // 64)
        self._run_serial(compiled, prefix, 0, probe_end)
        if loop_end - probe_end >= compiled.parallel_min_elems:
            self._parallel_execute(compiled, prefix, probe_end, loop_end)
        else:
            self._run_serial(compiled, prefix, probe_end, loop_end)

    def _run_serial(self, compiled: CompiledKernel, prefix: tuple,
                    start: int, end: int):
        """Run a range on this thread, refreshing the cost estimate."""
        if end <= start:
            return
        t0 = time.perf_counter_ns()
        compiled.call_range(prefix, start, end)
        elapsed = time.perf_counter_ns() - t0

        sample = elapsed / (end - start)
        prev = compiled.ns_per_elem

        # The response is deliberately asymmetric, because the two errors
        # are not: an estimate that is too high fans out ranges that cannot
        # repay it and costs an order of magnitude, while one that is too
        # low only leaves some parallelism unclaimed.
        if prev <= 0.0:
            ns_per_elem = sample
        elif sample > prev * _MAX_SAMPLE_RATIO:
            # Almost always a descheduled run rather than a real change.
            # Bound what it may claim; a genuine rise still arrives over a
            # few dispatches.
            ns_per_elem = prev + _COST_SMOOTHING * (
                prev * _MAX_SAMPLE_RATIO - prev)
        elif sample * _MAX_SAMPLE_RATIO < prev:
            # The estimate sits far above what the kernel now measures.
            # Believe the measurement outright: easing toward it would leave
            # threading on for thousands of dispatches on the way down.
            ns_per_elem = sample
        else:
            ns_per_elem = prev + _COST_SMOOTHING * (sample - prev)
        compiled.ns_per_elem = ns_per_elem
        compiled.parallel_min_elems = self._min_elems(ns_per_elem)

    def _min_elems(self, ns_per_elem: float) -> int:
        """Smallest range worth fanning out, for a kernel of this cost.

        Threading takes overhead + T_serial/P, so it wins once T_serial
        passes overhead·P/(P-1); the break-even factor covers that term and
        leaves margin, so we thread on a real win rather than a predicted
        tie. Until the fan-out has been measured this uses a deliberately
        pessimistic default, which only delays the first fan-out.
        """
        if ns_per_elem <= 0.0 or self.num_threads <= 1:
            return _NEVER
        fan_out = self._fan_out_ns
        if fan_out is None:
            fan_out = _DEFAULT_FAN_OUT_NS
        return max(self.num_threads,
                   int(fan_out * _PARALLEL_BREAK_EVEN / ns_per_elem))

    def _get_pool(self) -> ThreadPoolExecutor:
        """Return the persistent thread pool, creating it on first use."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.num_threads)
        return self._pool

    def _calibrate_fan_out(self, compiled: CompiledKernel, prefix: tuple):
        """Measure what a parallel dispatch costs before any work happens.

        Fans out *empty* ranges through the real dispatch path — submit,
        wake, ctypes call, join, with the GIL contention a real dispatch
        sees. An empty range runs no loop iterations, so the probe writes
        nothing and is safe on any kernel, including ones that accumulate
        with atomics.

        Measured against a real 1024-element fan-out on this machine: probe
        223 µs, actual 235 µs. A pure-Python no-op probe reads 122 µs, which
        would have understated the cost by nearly half.
        """
        pool = self._get_pool()
        run = compiled.call_range
        best = None
        for _ in range(_CALIBRATION_REPS):
            t0 = time.perf_counter_ns()
            futures = [pool.submit(run, prefix, 0, 0)
                       for _ in range(self.num_threads)]
            for f in futures:
                f.result()
            elapsed = time.perf_counter_ns() - t0
            best = elapsed if best is None else min(best, elapsed)
        self._fan_out_ns = float(best)

    def _parallel_execute(self, compiled: CompiledKernel, prefix: tuple,
                          start: int, end: int):
        """Split the loop range across threads."""
        total = end - start
        chunk = (total + self.num_threads - 1) // self.num_threads

        pool = self._get_pool()
        run = compiled.call_range
        futures = []
        for t in range(self.num_threads):
            t_start = start + t * chunk
            t_end = min(t_start + chunk, end)
            if t_start >= end:
                break
            futures.append(pool.submit(run, prefix, t_start, t_end))
        for f in futures:
            f.result()  # propagate exceptions

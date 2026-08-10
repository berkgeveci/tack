"""Tack CPU backend — JIT compiles kernels via llvmlite and runs with thread pool.

The kernel function signature is:
    void kernel(field0_ptr, field1_ptr, ..., i64 loop_start, i64 loop_end)

Fields are passed as ctypes pointers to their underlying numpy data.
The loop range is split across threads for parallel execution.

Threading decision
------------------
Waking a pool of Python threads costs on the order of a hundred
microseconds, so a fan-out only pays when the serial run would take
meaningfully longer than that.  Both sides of that comparison are
*measured*, never assumed: the backend times its own fan-out once, and
every compiled kernel carries a running estimate of its serial nanoseconds
per element, less the fixed cost of the call around it.

A fixed element-count threshold cannot do this job, because the crossover
moves by three orders of magnitude with the kernel's arithmetic intensity.
Measured on a 10-core Apple silicon machine, f32:

    out[i] = x[i]*2 + 1          memory bound     crossover ~4,000,000
    out[i] = sqrt(..) + sin(..)  ~10 flop/elem    crossover   ~130,000
    20-iteration inner loop      ~120 flop/elem   crossover     ~4,000

The original constant of 1024 sat below all three, so every mid-size
dispatch of a cheap kernel paid ~200 µs to save ~20 µs.

Those numbers are one machine's.  Anything here expressed in nanoseconds or
elements is derived from a measurement taken on the machine in hand rather
than written down, because a threshold in elements silently encodes the
thread cost and timer resolution of wherever it was chosen.  What is
written down are ratios -- how much margin to demand, how much of a sample
to trust -- which carry across machines in a way element counts do not.

None of this has been validated off Apple silicon.  CI exercises the logic
on Linux, but the tests supply their own timings so they do not measure
tuning quality.
"""

import ctypes
import ctypes.util
import itertools
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from llvmlite import binding as llvm

from tack.lang import ir
from tack.lang.field import NumpyBuffer
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

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
# Syscall numbers are per-architecture.  These are x86-64's; arm64 numbers the
# same calls differently -- 238 is migrate_pages there -- so _init_numa declines
# on any other machine rather than issuing a syscall it cannot name.
_SYS_SET_MEMPOLICY = 238
_SYS_GET_MEMPOLICY = 239
_MPOL_DEFAULT = 0
# From linux/mempolicy.h.  Note 5 is MPOL_PREFERRED_MANY, a non-binding hint
# that leaves pages on the faulting node -- i.e. does nothing this code wants.
_MPOL_INTERLEAVE = 3


def _mempolicy_interleaves() -> bool:
    """Ask the kernel to interleave, then ask it what it actually stored.

    Neither wrong constant fails loudly.  A bad syscall number or an
    unsupported mode returns -1, which nothing here would notice, and a mode
    that exists but means something else installs cleanly and then never
    interleaves.  Both were true of this code before, and both are invisible
    without reading the policy back, so it is read back.
    """
    mask = (ctypes.c_ulong * 1)(_numa_node_mask)
    if _libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_INTERLEAVE, mask,
                     _numa_max_node + 2) != 0:
        return False
    mode = ctypes.c_int(-1)
    rc = _libc.syscall(_SYS_GET_MEMPOLICY, ctypes.byref(mode), None,
                       ctypes.c_ulong(0), None, 0)
    _libc.syscall(_SYS_SET_MEMPOLICY, _MPOL_DEFAULT, None, 0)
    return rc == 0 and mode.value == _MPOL_INTERLEAVE


def _init_numa():
    global _numa_available, _numa_node_mask, _numa_max_node, _libc
    if platform.machine() != "x86_64":
        return  # the syscall numbers above are x86-64's
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
        if not _mempolicy_interleaves():
            _libc = None
            _numa_node_mask = 0
            return
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

# Re-export shared utilities so existing `from tack.runtime.cpu import ...` works.
# Backends must NOT rely on this — importing this module pulls in llvmlite, which
# is a CPU-only dependency.  Import from tack.runtime.kernel_utils instead.
from tack.runtime.kernel_utils import (  # noqa: F401
    _create_pack_fields,
    _detect_template_args,
    _detect_texture_fields,
    _detect_vector_fields,
    _detect_vector_fields_from_args,
    _expand_template_args,
    _get_loop_range,
    _resolve_range_expr,
    _update_pack_fields,
    new_kernel_cache,
    resolve_variant,
)

# Thread only on a real win: parallel takes overhead + T_serial/P, so the
# break-even is already P/(P-1); the rest is margin against a mis-estimate.
_PARALLEL_BREAK_EVEN = 2.0

# --- policy v2, off by default (TACK_CPU_POLICY=v2) ------------------------
#
# Two measured defects, which want fixing together because each is the only
# thing currently masking the other.
#
# The probe takes the minimum of back-to-back fan-outs, which is the hot end
# of a distribution a real dispatch rarely meets: workers that have been idle
# cost 2.45x that on a 2-socket Xeon and 3.81x on an M1 Max. But the *shape*
# of the descent differs -- the Xeon has finished falling by 10 ms, the M1 Max
# is barely started at 10 and still going at 50 -- so no single "idle floor"
# constant travels. Calibrating at several gaps and interpolating on the
# actual idleness at dispatch does travel, and costs one timestamp.
#
# And the threshold assumes the parallel speedup of the work is the thread
# count. It is not: P_eff is 1.5-9.2 depending on the kernel, so the margin
# the formula really applies is `M * (1 - 1/P_eff)`, which goes to zero as
# P_eff goes to 1. Deriving the threshold from `r_s - r_p` instead removes
# the assumption -- and reduces to the old formula exactly when r_p is 0,
# which is what it is until a parallel dispatch has been observed.
_FAN_OUT_GAPS_MS = (0.0, 10.0, 50.0)
_FAN_OUT_GAP_REPS = 3

# Shortest worker chunk whose timing is worth believing. `perf_counter_ns`
# resolves to tens of nanoseconds, so a microsecond of work is a couple of
# hundred ticks and the quantisation is nothing.
#
# This replaced a guard of a very different kind, and the difference is the
# whole point. `r_p` used to be learned as a residual --
# `(elapsed - fan_out_estimate) / elems` -- which charges every error in the
# fan-out estimate to `r_p`, amplified by `1/elems`. Its guard therefore had
# to demand that the work be a large multiple of the fan-out, and on yavin
# no such multiple existed in the range that matters: at 1.3x it admitted
# noise (r_p biased 1.3-13.6x high, and a P_eff of 0.13 for the
# bandwidth-bound kernel, which says fanning out makes work slower per
# element), and at 3x and above no dispatch in the scoring grid qualified at
# all, so `r_p` stayed 0.0 and v2 reduced silently to v1.
#
# That is not a badly chosen constant; near the crossover work and fan-out
# are the same size *by definition*, so no ratio can separate them. Timing
# the workers instead removes the subtraction, and with it the need to
# compare against the fan-out at all -- so what is left to guard is only
# whether the clock can see the interval.
_RP_MIN_WORKER_NS = 1_000.0

# v2's margin, and lower than v1's 2.0 deliberately. Under v1 the margin
# was absorbing *systematic* error -- an understated probe and, on a
# 2-socket box, an overstated serial rate -- which is why 2.0 was still
# not enough for a bandwidth-bound kernel and too much for every other.
# v2 removes both, leaving sampling noise: the floor repeats within
# 10-15% across runs and r_p smooths similarly, so 1.5 covers what is
# left about three times over.
#
# It is also close to what v1 already applied where it behaved. The
# effective margin is `M * (1 - 1/P_eff)`, so v1's 2.0 came to 1.65 for a
# ~10 flop/elem kernel and 1.77 for a heavy one; 1.5 leaves those roughly
# where they ship. What moves is the bandwidth-bound case, whose
# effective margin was 0.67-1.09 -- around and below break-even, which is
# the entire defect.
_V2_MARGIN = 1.5

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

# How much longer a timed slice should take than the fixed per-call cost.
# The rate comes from subtracting that cost, so the remainder has to be the
# bulk of the sample or the rate inherits the fixed cost's own error. Ten
# times over puts that under a few per cent, and lets the slice scale with
# the kernel: two elements of a costly one, thousands of a cheap one.
_SAMPLE_OVERHEAD_RATIO = 10.0

# Parallel dispatches to allow between refreshes of the cost estimate.
# Doubles up to the cap, so a wrong decision to thread is caught after a
# single dispatch, while a kernel that really wants threads settles into
# one serial run per cap.
_RECHECK_CAP = 1024

# Per-element cost assumed when deciding whether an *unmeasured* kernel is
# worth probing. It cannot be measured yet — that is what the probe is for —
# so the threshold answers "could a kernel this expensive repay a fan-out
# over this range?". Kernels dearer than this exist; guessing low costs them
# one serial dispatch before the estimate exists, which is the cheapest way
# to be wrong here. The range it implies scales with the measured fan-out,
# so a machine with cheaper threads starts probing sooner.
_PROBE_REFERENCE_NS_PER_ELEM = 24.0

_CALIBRATION_REPS = 5

# Repetitions when timing a kernel's fixed per-call cost. It runs once per
# compiled kernel, on empty ranges.
_OVERHEAD_REPS = 5

# Floor for the per-element estimate, so a kernel whose body disappears into
# the noise still reads as measured rather than as never measured.
_MIN_NS_PER_ELEM = 1e-3

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
        # The same rate for a fanned-out run, updated on every parallel
        # dispatch big enough to measure one (policy v2). 0.0 means "not
        # measured yet", which makes the v2 threshold reduce to the v1 one
        # rather than needing a special case. P_eff is ns_per_elem over it.
        self.ns_per_elem_parallel = 0.0
        # What one call_range costs before touching a single element --
        # ctypes marshalling, mostly. Timed once, on an empty range.
        self.call_overhead_ns = 0.0
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


def _linux_core_count() -> int | None:
    """Physical cores from sysfs, discounting SMT siblings."""
    try:
        with open("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list") as f:
            threads_per_core = len(f.read().strip().split(","))
        total = os.cpu_count() or 1
        return max(1, total // threads_per_core)
    except (OSError, ValueError):
        return None


def _macos_core_count() -> int | None:
    """Performance cores, or physical cores where there is one kind.

    Apple silicon mixes performance and efficiency cores, and the fan-out
    splits a range into equal chunks. An efficiency core takes several times
    as long over the same chunk, and every thread waits for the slowest, so
    counting the efficiency cores in makes the whole dispatch run at their
    pace. Measured on an 8+2 machine, eight threads beat ten.
    """
    try:
        libc = ctypes.CDLL("libc.dylib")
    except OSError:
        return None
    for key in (b"hw.perflevel0.physicalcpu", b"hw.physicalcpu"):
        value = ctypes.c_int(0)
        length = ctypes.c_size_t(ctypes.sizeof(value))
        rc = libc.sysctlbyname(key, ctypes.byref(value), ctypes.byref(length),
                               None, ctypes.c_size_t(0))
        if rc == 0 and value.value > 0:
            return value.value
    return None


def _windows_core_count() -> int | None:
    """Physical cores via GetLogicalProcessorInformationEx.

    os.cpu_count() reports logical processors, so on any SMT machine it is
    double what we want. The records are variable length, but each carries
    its own size, so counting the processor-core ones needs no unpacking of
    the union that follows.

    Untested — no Windows machine to hand. It fails to None rather than
    guessing, and the caller falls back to the logical count.
    """
    try:
        import ctypes
        from ctypes import wintypes

        RelationProcessorCore = 0
        kernel32 = ctypes.windll.kernel32
        size = wintypes.DWORD(0)
        kernel32.GetLogicalProcessorInformationEx(
            RelationProcessorCore, None, ctypes.byref(size))
        if size.value == 0:
            return None
        buf = (ctypes.c_byte * size.value)()
        if not kernel32.GetLogicalProcessorInformationEx(
                RelationProcessorCore, buf, ctypes.byref(size)):
            return None

        raw = bytes(buf)
        offset = 0
        cores = 0
        while offset + 8 <= size.value:
            relationship = int.from_bytes(raw[offset:offset + 4], "little")
            record_size = int.from_bytes(raw[offset + 4:offset + 8], "little")
            if record_size == 0:
                break
            if relationship == RelationProcessorCore:
                cores += 1
            offset += record_size
        return cores or None
    except Exception:
        return None


def _physical_core_count() -> int:
    """Number of cores worth running one compute thread on each.

    Not the logical processor count: hyperthreads share an execution unit,
    so a second thread on one buys little for compute-bound work while
    costing a full fan-out slot. os.cpu_count() counts them, which is why
    each platform is asked properly first.
    """
    for probe in (_linux_core_count, _macos_core_count, _windows_core_count):
        count = probe()
        if count:
            return count
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
        # policy v2: the fan-out cost as a function of how long the workers
        # have been idle -- [(gap_ns, cost_ns)], ascending -- plus the clock
        # reading that says which point of it this dispatch is at.
        self.policy = os.environ.get("TACK_CPU_POLICY", "v1")
        # The margin is insurance against a mis-estimate, so how much is
        # wanted depends on how good the estimates are -- and under v1 the
        # 2.0 written here was never the margin applied: P6 discounts it to
        # `2.0 * (1 - 1/P_eff)`, which for a bandwidth-bound kernel is under
        # 1.0. Correcting that makes 2.0 mean 2.0 for the first time, so the
        # number itself wants re-choosing rather than inheriting.
        default_margin = (_V2_MARGIN if self.policy == "v2"
                          else _PARALLEL_BREAK_EVEN)
        self.margin = float(os.environ.get("TACK_CPU_MARGIN", default_margin))
        self._fan_out_curve: list[tuple[float, float]] = []
        self._last_dispatch_ns = 0

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
                compiled.parallel_min_elems = self._min_elems(compiled.ns_per_elem,
                                                    compiled.ns_per_elem_parallel)
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
                                self._sample_elems(compiled, loop_end))
                self._run_serial(compiled, prefix, 0, probe_end)
                if loop_end > probe_end:
                    self._parallel_execute(compiled, prefix, probe_end,
                                           loop_end)
                return
            self._parallel_execute(compiled, prefix, 0, loop_end)
            return

        probe_min_range = self._probe_min_range()
        if compiled.ns_per_elem > 0.0 or loop_end < probe_min_range \
                or self.num_threads <= 1:
            # Already measured and too small, or too small to be worth
            # splitting off a probe.
            self._run_serial(compiled, prefix, 0, loop_end)
            return

        # First sight of this kernel at a range where threading might pay.
        # Time a small prefix, then decide about the rest — so a one-shot
        # large dispatch is not stuck running serially for want of a sample.
        probe_end = max(probe_min_range // 16, loop_end // 64)
        self._run_serial(compiled, prefix, 0, probe_end)
        if loop_end - probe_end >= compiled.parallel_min_elems:
            self._parallel_execute(compiled, prefix, probe_end, loop_end)
        else:
            self._run_serial(compiled, prefix, probe_end, loop_end)

    def _probe_min_range(self) -> int:
        """Smallest range worth splitting a timing probe off.

        Derived from the machine's own fan-out cost rather than fixed: where
        waking threads is cheap a shorter range can repay it, so measuring
        should start sooner. A hardcoded element count silently encodes the
        thread cost of whichever machine it was chosen on.
        """
        fan_out = self._fan_out_ns
        if fan_out is None:
            fan_out = _DEFAULT_FAN_OUT_NS
        return max(self.num_threads,
                   int(fan_out * _PARALLEL_BREAK_EVEN
                       / _PROBE_REFERENCE_NS_PER_ELEM))

    def _sample_elems(self, compiled: CompiledKernel, loop_end: int) -> int:
        """Elements to time so the sample outweighs the fixed call cost.

        Targets a duration rather than a count, since what makes a sample
        trustworthy is that the per-element work dominates the fixed cost
        subtracted from it — and how many elements that takes is entirely
        the kernel's business.

        The two guards matter more than the target. A re-measurement exists
        because the estimate may be wrong, so sizing the slice purely from
        that estimate is circular: a wildly high one asks for a handful of
        elements, whose timing is then almost all fixed cost, which keeps
        the estimate wrong. Measured directly, that took a corrupted
        estimate three elements at a time and left it 640x high after four
        dispatches.

        So the slice is never a smaller share than 1/64 of the range, and a
        range too small to want threading at all is simply timed whole.
        """
        if loop_end <= self._probe_min_range():
            return loop_end

        rate = max(compiled.ns_per_elem, _MIN_NS_PER_ELEM)
        target_ns = _SAMPLE_OVERHEAD_RATIO * max(compiled.call_overhead_ns, 1.0)
        target = int(target_ns / rate)
        return max(1, min(loop_end, max(target, loop_end // 64)))

    def _measure_call_overhead(self, compiled: CompiledKernel, prefix: tuple):
        """Time an empty call_range — the part that is not per-element.

        Empty ranges run no iterations, so this writes nothing and is safe
        on any kernel, including ones that accumulate with atomics.
        """
        best = None
        for _ in range(_OVERHEAD_REPS):
            t0 = time.perf_counter_ns()
            compiled.call_range(prefix, 0, 0)
            dt = time.perf_counter_ns() - t0
            best = dt if best is None else min(best, dt)
        compiled.call_overhead_ns = float(best)

    def _run_serial(self, compiled: CompiledKernel, prefix: tuple,
                    start: int, end: int):
        """Run a range on this thread, refreshing the cost estimate."""
        if end <= start:
            return
        if compiled.call_overhead_ns <= 0.0:
            self._measure_call_overhead(compiled, prefix)

        t0 = time.perf_counter_ns()
        compiled.call_range(prefix, start, end)
        elapsed = time.perf_counter_ns() - t0

        # Charge only the per-element part. The rest is a fixed cost every
        # chunk pays, so threading does not divide it down -- and folding it
        # into a per-element rate is what made a cheap kernel read 2.6x more
        # expensive than it is, and fan out where serial was faster.
        work_ns = elapsed - compiled.call_overhead_ns
        sample = max(work_ns / (end - start), _MIN_NS_PER_ELEM)
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
        compiled.parallel_min_elems = self._min_elems(
            ns_per_elem, compiled.ns_per_elem_parallel)

    def _min_elems(self, ns_per_elem: float,
                   ns_per_elem_parallel: float = 0.0) -> int:
        """Smallest range worth fanning out, for a kernel of this cost.

        The two paths cost `n·r_s` and `fan_out + n·r_p`, so they cross at
        `fan_out / (r_s - r_p)` and the threshold is that with a margin --
        a real win rather than a predicted tie. Until the fan-out has been
        measured this uses a deliberately pessimistic default, which only
        delays the first fan-out.

        The shipped formula drops `r_p`, i.e. assumes a fanned-out run
        spreads the work across every thread and so costs nothing per
        element next to the serial rate. It does not: the measured speedup
        of the *work* is 1.5-9.2 depending on how much of the kernel is
        memory rather than arithmetic. Dropping the term inflates the
        denominator, and the margin actually applied comes out as
        `M·(1 - 1/P_eff)` -- which for a bandwidth-bound kernel with
        P_eff ≈ 1.5 is 0.67, i.e. below break-even, so the threshold lands
        *under* the crossover and the fan-out loses. Keeping the term needs
        no special case for "not measured yet": r_p is 0 then, and the
        expression reduces to exactly the old one.
        """
        if ns_per_elem <= 0.0 or self.num_threads <= 1:
            return _NEVER
        fan_out = self._fan_out_estimate()
        rate = ns_per_elem
        if self.policy == "v2":
            rate = ns_per_elem - min(ns_per_elem_parallel, ns_per_elem * 0.9)
        return max(self.num_threads,
                   int(fan_out * self.margin / rate))

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
        if self.policy == "v2":
            self._calibrate_fan_out_curve(compiled, prefix)

    def _calibrate_fan_out_curve(self, compiled: CompiledKernel,
                                 prefix: tuple):
        """Measure the fan-out cost at several degrees of worker idleness.

        The probe above answers "what does a fan-out cost right after
        another one", which is the cheapest it ever is. A dispatch that
        follows a pause pays more, because the cores have descended into
        deeper idle states -- and how much more, and how soon, is a fact
        about the machine: finished by 10 ms on a 2-socket Xeon, still
        falling at 50 ms on an M1 Max. Sampling the curve rather than
        picking a constant is what lets one mechanism fit both.

        Costs the sum of the gaps once, on the first fan-out.
        """
        pool = self._get_pool()
        run = compiled.call_range
        curve = []
        for gap_ms in _FAN_OUT_GAPS_MS:
            times = []
            for _ in range(_FAN_OUT_GAP_REPS):
                if gap_ms:
                    time.sleep(gap_ms / 1000.0)
                t0 = time.perf_counter_ns()
                futures = [pool.submit(run, prefix, 0, 0)
                           for _ in range(self.num_threads)]
                for f in futures:
                    f.result()
                times.append(time.perf_counter_ns() - t0)
            times.sort()
            # Median, not minimum: the minimum is what made the shipped
            # probe read the hot end of its own distribution.
            curve.append((gap_ms * 1e6, float(times[len(times) // 2])))
        # Monotone by construction -- a longer pause cannot wake threads
        # faster, and a dip is sampling noise that would otherwise make
        # the interpolation non-monotone.
        for i in range(1, len(curve)):
            if curve[i][1] < curve[i - 1][1]:
                curve[i] = (curve[i][0], curve[i - 1][1])
        self._fan_out_curve = curve

    def _fan_out_estimate(self) -> float:
        """What a fan-out costs *for this dispatch*, given its idleness.

        Piecewise-linear on the measured curve, clamped at both ends. With
        no curve (policy v1, or before calibration) this is the single
        probe value, so callers need no branch -- and the clock is not read
        at all, which keeps the shipped path exactly as it was.
        """
        curve = self._fan_out_curve
        if not curve:
            return (self._fan_out_ns if self._fan_out_ns is not None
                    else _DEFAULT_FAN_OUT_NS)
        gap = time.perf_counter_ns() - self._last_dispatch_ns
        if gap <= curve[0][0]:
            return curve[0][1]
        for (g0, c0), (g1, c1) in itertools.pairwise(curve):
            if gap <= g1:
                span = g1 - g0
                return c0 + (c1 - c0) * ((gap - g0) / span) if span else c1
        return curve[-1][1]

    def _record_parallel_cost(self, compiled: CompiledKernel,
                              worker_rates: list, workers: int):
        """Learn `r_p` from what the workers themselves reported.

        `worker_rates` holds each worker's own ns-per-element for the
        chunk it ran. Those timestamps are taken *inside* the worker,
        around `call_range` and nothing else, so they contain no fan-out
        to subtract and no wakeup to be confused by.

        Two conversions, and both are easy to get backwards.

        A worker's own rate is roughly the *serial* rate: the speedup of
        a fan-out comes from running `P` chunks at once, not from any
        element becoming cheaper. So the aggregate rate the crossover
        model wants is the worker rate divided by the number of workers
        that actually ran -- which is not always `num_threads`, since a
        range shorter than `num_threads` chunks starts fewer.

        And the statistic is the **median** across workers, not the max.
        The dispatch does wait for the slowest, so the max is what the
        model literally describes -- but on a machine with any background
        load the max over eight samples is whichever worker was
        descheduled, which measures the scheduler rather than the kernel.
        Probed on a loaded yavin, max read 10-134x the fitted rate where
        the median read 0.5-6.3x.
        """
        rates = sorted(worker_rates)
        if not rates or workers <= 0:
            return
        median_rate = rates[len(rates) // 2]
        sample = max(median_rate / workers, _MIN_NS_PER_ELEM)
        prev = compiled.ns_per_elem_parallel
        compiled.ns_per_elem_parallel = (
            sample if prev <= 0.0
            else prev + _COST_SMOOTHING * (sample - prev))
        compiled.parallel_min_elems = self._min_elems(
            compiled.ns_per_elem, compiled.ns_per_elem_parallel)

    def _parallel_execute(self, compiled: CompiledKernel, prefix: tuple,
                          start: int, end: int):
        """Split the loop range across threads."""
        total = end - start
        chunk = (total + self.num_threads - 1) // self.num_threads

        pool = self._get_pool()
        run = compiled.call_range
        measure = self.policy == "v2"

        if not measure:
            # v1's path, untouched: no clock read, no per-worker wrapper.
            futures = []
            for t in range(self.num_threads):
                t_start = start + t * chunk
                t_end = min(t_start + chunk, end)
                if t_start >= end:
                    break
                futures.append(pool.submit(run, prefix, t_start, t_end))
            for f in futures:
                f.result()  # propagate exceptions
            return

        # One slot per worker, written only by that worker, so the rates
        # need no lock -- which matters because a lock here would be held
        # inside the measured region.
        rates: list = [0.0] * self.num_threads

        def timed(slot: int, t_start: int, t_end: int):
            t = time.perf_counter_ns()
            run(prefix, t_start, t_end)
            elapsed = time.perf_counter_ns() - t
            elems = t_end - t_start
            if elems > 0 and elapsed >= _RP_MIN_WORKER_NS:
                rates[slot] = elapsed / elems

        futures = []
        workers = 0
        for t in range(self.num_threads):
            t_start = start + t * chunk
            t_end = min(t_start + chunk, end)
            if t_start >= end:
                break
            workers += 1
            futures.append(pool.submit(timed, t, t_start, t_end))
        for f in futures:
            f.result()  # propagate exceptions

        self._record_parallel_cost(compiled, [r for r in rates if r > 0.0],
                                   workers)
        self._last_dispatch_ns = time.perf_counter_ns()

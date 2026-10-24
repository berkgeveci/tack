# Vendor workarounds

Tack generates source for five toolchains and calls four vendor runtimes
through bindings, and some of what it emits is shaped around a defect in one
of them rather than by the language contract. This page lists each such
workaround with the vendor and version it was found on, what goes wrong
without it, where it lives, and the test that would notice if it were
removed. The point is to know when one can go: a workaround guards a version
range, and a driver fix makes it dead weight that still costs something
(a function call, a slower path, a missed optimization).

To retire one, run its test on the newest toolchain with the workaround
switched off. If the test passes there and on the oldest version Tack still
supports, remove it and record the versions here.

Design choices that look like workarounds but are not are listed at the
end: they follow from the contract, and no vendor fix would remove them.

## Metal (Apple)

| Workaround | Found on | Without it | Where | Test |
|---|---|---|---|---|
| A kernel whose sequential loop stores to an array (or runs an atomic) has its body in a `noinline` function, `__tack_body__`, that the entry calls; workgroup arrays are declared in the entry and passed as pointers. | M1 Max, Apple's GPU back end at pipeline creation (2026) | Wrong results: the compiler read an array element once before the loop when its address did not depend on the thread and the loop stored to another array of the same type (`total[0] += x[k]; counter[0] += 1`). The front end's LLVM IR is correct; alias metadata is not the cause, and `restrict`, opaque pointers and compile options do not avoid it. | `msl_gen._stores_inside_sequential_loop`, `_BODY_FUNCTION` | `test_array_updates_in_loops.py` |
| i64/u64 additions that update an accumulator inside a runtime-bound loop go through separate `noinline` helpers. | M1 Max compiler | The Metal compiler crashed. | `msl_gen` (`_opaque_integer_add`), `codegen/integer_ops.py` | none dedicated; i64 accumulator loops on Metal would crash the compiler again |
| A typed or integer constant array read at a runtime index is one constant table (`IRTableLoad`), not a select chain. | Metal's compiler service | Select chains inside a storing loop crashed the compiler service. | `codegen/tables.py`, `ast_transform` (`constant_table`) | `test_constant_tables.py` |
| Every `new...` selector Tack calls is registered as returning an owned object (`_OWNED_RESULTS`) before any is called. | pyobjc-framework-Metal 12.1 | Its metadata omits `already_retained`, so PyObjC retained each buffer, texture, library and pipeline once more and none was ever freed; a benchmark's temporaries reached 150 GB. | `runtime/metal.py` | `test_arrays_own_their_buffers_alone` in `test_metal.py` |

## CUDA (NVIDIA)

| Workaround | Found on | Without it | Where | Test |
|---|---|---|---|---|
| i8/i16 negation and `abs` go through an inline-PTX `sub.u32` (`__tack_ptx_neg_u32__`). | H100, driver 595.58 (CUDA 13.2): ptxas 12.0-13.2 and the PTX JIT of drivers before 13.3, on sm_80, sm_89 and sm_90 alike | PTX `neg.s16`/`abs.s16` were widened without re-extending, so the wrapped minimum came back as its magnitude. | `codegen/integer_ops.py` (`_negates_in_ptx`), `_ptx_negation` in `cuda_gen.py` | `test_narrow_negation_avoids_ptx_neg_s16` (`test_cuda.py`) checks the PTX, since drivers from 13.3 pass the widening test either way |

| A floating product that is not an operand of a `+` or `-` is emitted as `__fmul_rn`/`__dmul_rn`; products inside a sum stay plain. | NVRTC 13 (`--fmad=true`), RTX 4060 Ti, 2026-10-23 | NVRTC fuses a multiply into an add in a later statement: `p = a * b; out = p + c` became one fma, `p` never rounded (100k of 100k values), against the contract's one rounding per named value. Found while the locator's point-side and cell-side searches disagreed at shared vertices on CUDA (that disagreement was fixed by a margin on the locator's holds test; this one is the contract). NVRTC has no `-ffp-contract=on`; `--fmad=false` would forbid the fusion the contract permits inside an expression. | `cuda_gen.py` (`_rounded_products`, `_fusable`) | `test_f32/f64_a_named_product_is_rounded_before_a_later_sum` (`test_float_semantics.py`) |
| Exportable memory uses `CU_MEM_HANDLE_TYPE_WIN32_KMT` on Windows, and a successful allocation is trusted over the device attribute. | GeForce RTX 5060; cuda-python 13.3.1 | `cuMemCreate` refused the NT handle type despite the attribute claiming support, and cuda-python segfaulted when asked for it with security attributes. | `cuda_backend._EXPORT_HANDLE_TYPES`, `_create_backing` | none dedicated (Windows only) |

## HIP (AMD)

| Workaround | Found on | Without it | Where | Test |
|---|---|---|---|---|
| `hiprtcDestroyProgram` is never called (a small leak, bounded by kernel caching). | hip-python 7.1; still crashing on hip-python 7.2.2 with ROCm 7.0.2 (MI300X, 2026-08-11) | SIGSEGV on the first call, after a successful compile. The documented convention (`hiprtcDestroyProgram(prog)`) is the one that crashes; a pointer to the program is refused as a type error. | `runtime/hip_backend.py` | none: removing it crashes the process |
| hipRTC compiles with `-ffp-contract=on` (`_HIPRTC_OPTIONS`). | hipRTC's default `fast` contraction | A product was fused into one read of a variable while another read of it stayed rounded, so `a - a != 0.0` was true for finite `a` (the finiteness test in `_crossing`). | `runtime/hip_backend.py` | none dedicated; the dataset contour tests on HIP failed without it (`_crossing` in `tack.dataset.algorithms._common`) |
| Waits go through ctypes on the runtime library (`_runtime`, `_StreamMarker`), never hip-python's `hipStreamSynchronize`. | hip-python 7.x wrappers; ROCm 7.1 | hip-python holds the GIL while the runtime waits (another thread ran 6 times during a 357 ms wait, 353 through ctypes), and ROCm 7.1's stream synchronize holds the stream, so another thread's launch waited the whole 350 ms. | `runtime/hip_backend.py` | `test_hip_async.py`, `test_asyncio.py` |
| `HIPTextureImage` writes array handles and fixed-size arrays through ctypes, located from neighbouring views. | hip-python 7.x struct wrappers | `make_hipExtent` and friends are gone, `hipCreateChannelDesc` returns `(err, desc)`, nested members are read-only views, and `res.array.array`, `hipMemcpy3DParms.dstArray` and `hipTextureDesc.addressMode` are not exposed at all. | `runtime/hip_backend.py` | `test_hip.py` (texture tests; MI210 with point filtering) |
| Textures fall back to software sampling where the device has no image hardware, and on gfx90a. | MI300 (gfx940-942) has no texture hardware; gfx90a (MI200) reports image support but refuses a linearly filtered 3D texture (`hipErrorNotSupported`) | hipRTC refuses `tex3D` ("The image/texture API not supported on the device"), or texture creation fails. | `HIPBackend.texture_in_hardware` (asks `hipDeviceAttributeImageSupport` and tries a 2×2×2 texture at init) | `test_texture_snapshot.py` |
| A launch is capped below 2^32 work-items. | AMD's dispatch packet | The packet counts work-items in 32 bits; a larger grid would wrap. | `HIPBackend._query_max_launch` | `test_launch_limits.py` |
| A GPU-aware MPI is handed `hipMalloc` memory, never the stream-ordered pool's: `tack.communication` copies an array into a buffer from `allocate_mpi_array` before a send and receives into one (`HIPBackend.mpi_direct`). | MVAPICH-Plus 4.1 for ROCm 7.1.1, its embedded UCX with `libuct_rocm`; two MI210s (illyad, 2026-10-24) | HSA reports pool memory (`hipMallocAsync`) as `HSA_EXT_POINTER_TYPE_HSA_VMEM`, which this UCX takes for host memory: a send asserts (`rndv_rtr.c:121 rkey_size == packed_rkey_size`) or, eagerly, segfaults reading the device address. `hipMalloc` memory (`HSA`) goes through at 44 GB/s a rank; the copy costs a device-to-device pass. | `runtime/hip_backend.py`, `communication/_world.py` (`_direct`, `_empty_for_mpi`) | `test_hip.py::test_memory_for_a_gpu_aware_mpi`; `test_communication.py`'s `gpu-aware-not-direct` MPI run (the CPU standing in) |

**A known defect left in place.** ROCm 7.0.2 and 7.1.1's device compiler
(AMD clang 20, in hipRTC's comgr) miscompiles one generated integer
expression at `-O1` and above (a 64-bit signed `-253 < -1` inside the full
expression evaluates false). It depends on the surrounding expression, so
Tack does not try to avoid it; `test_integer_expression_differential.py`
marks that case as a strict expected failure when `hipRuntimeGetVersion()`
reports ROCm 7.0 or 7.1, and a fixed toolchain reports XPASS, when the mark
goes. Details in [Backend implementations](backend-implementations.md#hip).

hip-python 10 is a compatibility package over `rocm.bindings` that moved
several names (`LONG_VERSION`, `hip.chip`, output structs, string fields);
`hip_backend.py` supports both. That is an API change rather than a defect,
and goes when 7.x support does.

## Level Zero and IGC (Intel)

| Workaround | Found on | Without it | Where | Test |
|---|---|---|---|---|
| Signed negation below 32 bits and `abs` below 64 are `noinline` helpers. | IGC 2.7.11 (compute runtime 25.05, Max 1100) | The wrapped minimum was widened as its magnitude: i16 `-(-32768)` came back as 32768. | `codegen/integer_ops.py`, `opencl_gen.py` | `test_opencl_gen.py` (`test_widenable_negation_and_abs_are_noinline`), `test_integer_semantics.py` (`test_wrapped_negation_and_abs_survive_widening`) on Level Zero |
| The global index is built from `get_group_id(0)`, widened, times the group size, not from `get_global_id(0)`. | Intel's OpenCL builtins (compute runtime 25.05) | `get_global_id(0)` wraps at 2^32 despite returning `size_t`, so item 2^32 + k ran as item k. | `opencl_gen.py`, `codegen/reductions.py` | `test_opencl_gen.py` (the generated index) |
| Each queued launch, fill and copy is followed by a barrier on an asynchronous immediate list, rather than using an in-order list. | Driver 25.05 | In-order lists (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`) cost about 100 µs of device time per kernel; a barrier costs about 5. | `level_zero_backend._ordered` | `test_level_zero_async.py` |
| Copies to the device up to 16 MiB and from it up to 64 KiB go through pooled USM host memory (`_HostStaging`). | Driver 25.05 | Appending a copy from pageable memory did not return until the queue drained (to the device up to 4 MiB, from it up to 1 KiB), so a launch's scalar upload waited for every kernel before it. | `level_zero_backend.py` | `test_level_zero_async.py` |
| `f64` `floor` and `ceil` are wrapped as `copysign(floor(x), x)`. | Max 1100, intel-opencl-icd 25.05.32567.17 | The double overloads returned `+0.0` for `ceil(-0.1)` and `floor(-0.0)`, where IEEE 754 requires `-0.0`; the f32 overloads are correct. Exact for every input, since neither function changes its operand's sign. | `OpenCLCodeGen._expr_call` | `test_float_semantics.py` on Level Zero |
| A runtime vector or matrix component index is compared as an i32: the selector is cast once where it is captured. | IGC 2.7.11 (compute runtime 25.05, Max 1100) | A chain of 64-bit equality tests on a selector that a conditional expression defined (`rows[f][j + 1 if j + 1 < size else 0]` in a sequential loop) selected the first component for an even selector and the last for an odd one, so the polyhedra face kernels read a triangle's padded -1 under debug mode's bounds checks (TA-48). `-ze-opt-disable` avoided it, as did a 32-bit selector, a `switch` or plain index arithmetic; the generated OpenCL C is correct and the CPU agrees. | `lowering/vectors.py` (`_narrow_selector`) | `test_new_features.py` (`test_runtime_component_from_a_conditional_in_a_loop`) on Level Zero |
| A stored native binary the driver refuses is rebuilt from the stored SPIR-V. | Drivers that refuse another build's native binary | The disk cache entry would be unusable. Counted as `native_refused` in the cache stats. | `level_zero_backend.load_variant` | `test_disk_cache.py` (`test_a_refused_native_binary_falls_back_to_spirv`) |

## CPU (LLVM)

| Workaround | Found on | Without it | Where | Test |
|---|---|---|---|---|
| The native assembly parser is initialized before the thread pool is built. | llvmlite on Apple silicon | The pool program uses inline asm (`yield`); without the parser LLVM aborts the process instead of raising. | `runtime/cpu_pool.py` (`_target_machine`) | `test_cpu_pool.py` |

## Not workarounds

These shape generated code or runtime behavior for reasons in the language
contract or in Tack's own design; no vendor fix would remove them.

- Metal array arguments are members of one argument buffer: separate
  device-buffer arguments promise disjoint storage in MSL, and arrays may
  alias (`docs/design/memory-and-aliasing.md`).
- CUDA/HIP widen `blockIdx.x` before multiplying by `blockDim.x`, because
  the built-ins are 32-bit; this is C's arithmetic, not a defect.
- Precise floating options (no `--use_fast_math`, `fastMathEnabled` off) are
  the floating-point contract.
- Level Zero refuses arrays from another context because Intel's driver
  resolves pointers from any context, so Tack's check is the only one
  (`test_level_zero_context.py`).
- Integer helpers for floor division, remainder, wrapping and power exist on
  every GPU backend because C leaves those cases undefined or different.
- Two CUDA spellings that OpenCL C forbids are rewritten for Level Zero: a
  floating condition in `?:` (`_ifexp_condition` compares it against zero)
  and `__local` declarations in nested scopes (`_declare_local` hoists them
  to the kernel's outermost scope). These are the dialect's rules.

## Details

The longer history of the narrowest ones, kept from the backend notes.

**HIP: `hiprtcDestroyProgram` is never called.** In hip-python it
segfaults on the first call after a successful compile. Recorded against
7.1 and confirmed again on 2026-08-11 with hip-python 7.2.2 on ROCm 7.0.2
(MI300X). The documented calling convention is the one that crashes. Tack
skips the call in `_compile_code_object`; each compile leaks one program
object, which the variant cache keeps rare.

**Metal: an opaque 64-bit add in one loop shape.** On an M1 Max, pipeline
compilation crashed when a wrapping 64-bit addition took part in reduction
optimization inside a loop with a runtime bound. `MSLCodeGen._emit_assign`
sets `_opaque_integer_add` while emitting an assignment whose value reads
its own target, inside a sequential loop whose end is not a constant;
`_expr_binop` then emits that `i64`/`u64` `+` through a helper declared
`__attribute__((noinline))` (`IntegerCodeGen` in `codegen/integer_ops.py`).
Every other integer operation stays inline.

**Level Zero: opaque signed negation and `abs`.** On an Intel Data Center
GPU Max 1100 (IGC 2.7.11), a wrapped negation or `abs` of a signed minimum
was correct at its own width but lost its sign when widened: i16
`-(-32768)` stored to an i32 array read `32768`, and `abs` of the i8, i16
and i32 minimum widened as its magnitude. The SPIR-V from `ocloc` was
correct, `-ze-opt-disable` hid the error, and no inlined form tried —
`as_short`, a select on the minimum, complement-plus-one, a sign mask —
survived the driver's folding; LLVM canonicalizes them all to the same
shape. `OpenCLCodeGen` sets `_opaque_negation`, and `IntegerCodeGen`
emits `neg` below 32 bits and `abs` below 64 bits as
`__attribute__((noinline))` helpers. i32 negation, i64 `abs` and every
other operation stay inline. `test_wrapped_negation_and_abs_survive_widening`
covers the widened results on each backend.

**CUDA: signed negation and `abs` below 32 bits as inline PTX.** On an
H100 with driver 595.58 (CUDA 13.2), the same symptom: i16 `-(-32768)` and
`abs(-32768)` widened as `32768`. Tack's C and NVRTC's PTX are correct
(`neg.s16`, then `cvt.s64.s16`); the driver's JIT is not. ptxas 12.0
through 13.2 sign-extend the input, negate in 32 bits and widen that
without re-extending from bit 15; ptxas 13.3 does not, and the architecture
does not matter (sm_80, sm_89 and sm_90 alike). Only `neg.s16` and
`abs.s16` are affected. NVVM narrows every C spelling of the negation to
those instructions, so `CUDACodeGen` sets `_ptx_negation` and
`IntegerCodeGen` negates i8 and i16 (for `neg` and `abs`) with
`__tack_ptx_neg_u32__`, an inline-PTX `sub.u32` on the 32-bit carrier,
then narrows as before. A noinline helper also gave the right answer, but
only because its caller re-extended the returned `short`. HIP and OpenCL
inherit `CUDACodeGen` and turn the option off. `test_narrow_negation_avoids_ptx_neg_s16`
checks the PTX, since drivers from 13.3 pass the widening test either way.

**Level Zero: `f64` `floor` and `ceil` lose the sign of zero.** On an Intel
Data Center GPU Max 1100 (intel-opencl-icd 25.05.32567.17), the double
overloads returned `+0.0` for `ceil(-0.1)` and `floor(-0.0)`, where IEEE 754
requires `-0.0`. The `f32` overloads are correct. `OpenCLCodeGen._expr_call`
wraps those two builtins at double width as `copysign(floor(x), x)`. Neither
function ever changes the sign of its operand, so restoring it is exact for
every input, and a no-op where the runtime is already correct.

**OpenCL C dialect rules.** Two inherited CUDA spellings are illegal OpenCL
C and failed only on the device: a floating-point condition in `?:`
(`_ifexp_condition` compares it against zero), and `__local` declarations
in nested scopes (`_declare_local` hoists them to the kernel's outermost
scope, which is legal because local memory is per-workgroup and statically
sized).

**CUDA exports avoid Win32 NT handles.** `_EXPORT_HANDLE_TYPES` uses
`CU_MEM_HANDLE_TYPE_WIN32_KMT` on Windows. On a GeForce RTX 5060 the NT
handle type was refused by `cuMemCreate` despite the device attribute
claiming support, and cuda-python 13.3.1 segfaulted when asked for it with
security attributes; `_create_backing` therefore trusts a successful
allocation, not the attribute.

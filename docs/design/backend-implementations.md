# Backend Implementations

Every backend turns the same typed, optimized Tack IR into native code and
runs it. What differs is the toolchain, how a device and context are
chosen, how arguments reach the kernel, and which hardware features exist.
This page walks through each backend in the order a kernel meets it —
initialization, compilation, launch — and closes with the workarounds each
one carries and why.

The shared front half — `resolve_variant()` in
`packages/tack-core/src/tack/runtime/kernel_utils.py`, which turns a call
into a cached compiled variant — is described in [Compilation
Pipeline](compilation-pipeline.md) and [Specialization and
Caching](specialization-and-caching.md). Each backend's `execute()` calls it
with a `build` callback, and that callback is where this page begins.

## The Backend contract

All five backends subclass `Backend` (`runtime/backend.py`); the four GPU
backends do so through `GPUBackend` (`runtime/gpu_backend.py`), which turns a
call into a queued launch the same way for all of them and leaves each the
hooks where they differ: how a kernel is compiled and launched, which lock a
launch holds, how its packed scalars are owned, and whether it must complete
before returning. Their code generators likewise share `CFamilyCodeGen`
(`codegen/c_family.py`). It declares
three required methods — `allocate_array()`, `wrap_ptr()` and `execute()` —
and a set of capability attributes that callers *read* instead of probing
with `hasattr`. Optional methods have defaults that suit a backend whose
launches complete before `execute()` returns: `synchronize()` waits for
every launch queued so far and is a no-op by default (`tack.sync()` calls
it; see [Launches are queued](#launches-are-queued)); `fill_array()`
returns false, leaving `Array.fill` to the buffer (the CPU fills large
arrays through its own dispatch); and `cache_target()` returns `None`,
which keeps a backend's compiled variants off the disk cache, while
`variant_artifact()` and `load_variant()` store and rebuild them (see
[Specialization and Caching](specialization-and-caching.md#the-disk-cache)).
`DeviceBuffer` likewise has `synchronize()`, a wait for the launches
queued on the buffer's device, and `share()`, which hands the storage
outside Tack: it waits now, and makes every later launch that uses the
buffer complete before returning. Anything derivable is derived: `supports_f64` is a property
over `supported_dtypes`, and `supported_atomic_dtypes` is a property that
intersects the per-target table `ATOMIC_DTYPES` (`lang/atomic_support.py`)
with `supported_dtypes`, so neither can disagree with the array types.

| Attribute | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| `name` | `cpu` | `metal` | `cuda` | `hip` | `level_zero` |
| `display_name` | CPU | Metal | CUDA | HIP | Level Zero |
| `supported_dtypes` | all ten | all but `f64` | all ten | all ten | nine, plus `f64` if the device reports it |
| `supports_f64` (derived) | yes | no | yes | yes | per device |
| `supports_device_reductions` | no | yes | yes | yes | yes |
| `supports_workgroups` | no | yes | yes | yes | yes |
| `supported_atomic_dtypes` (derived) | all ten | `i32`, `u32`, `f32` | `i32`, `u32`, `i64`, `u64`, `f32`, `f64` | as CUDA | `i32`, `u32`, `f32` |
| `device_memory_spaces` | — | — | `cuda`, `cuda_pinned`, `cuda_managed` | `hip`, `hip_pinned`, `hip_managed` | `level_zero` |
| `device_options` | `num_threads` | — | — | — | `external_context` |
| `dlpack_refusal_note` | — | set | — | — | — |

"All ten" is `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`,
`f64`. `supported_dtypes` is a `frozenset` on every backend. `device_options` lists the keyword arguments `tack.device()` forwards to
the backend's constructor; `tack.device()` (`runtime/dispatch.py`) raises
`ValueError` for any other keyword before constructing anything, so a
misspelled option cannot be silently dropped. The user-facing summary of
these capabilities is in [Backends](../reference/backends.md); the
normative statements are in [Backend
Capabilities](../contracts/backend-capabilities.md).

## Comparison

| | CPU | Metal | CUDA | HIP | Level Zero |
|---|---|---|---|---|---|
| Codegen | `llvm_gen.py` → LLVM IR | `msl_gen.py` → MSL | `cuda_gen.py` → CUDA C | `hip_gen.py` → HIP C | `opencl_gen.py` → OpenCL C |
| Compiler | llvmlite, O3 pipeline, MCJIT | `newLibraryWithSource` | NVRTC → PTX | hipRTC → code object | `libocloc` → SPIR-V → `zeModuleCreate` |
| Loaded with | `create_mcjit_compiler` | `newComputePipelineState` | `cuModuleLoadData` | `hipModuleLoadData` | `zeModuleCreate` + `zeKernelCreate` |
| Launch | ctypes call over `[start, end)` chunks | `dispatchThreads` | `cuLaunchKernel` | `hipModuleLaunchKernel` | `zeCommandListAppendLaunchKernel` |
| Group size | — (thread pool) | `min(pipeline max, 256)` | 256 | 256 | `min(256, maxGroupSizeX, maxTotalGroupSize)` |
| Array binding | typed pointer argument, passed as an address | pointer in one argument buffer | pointer argument | pointer argument | `__global` pointer argument |
| Scalar binding | by value | packed, into a new buffer per launch | packed into the variant's pack arrays | as CUDA | as CUDA |
| Launch returns | after the range has run | once encoded into the open command buffer | once queued on the context's stream | once queued on the device's stream | once appended to the asynchronous immediate command list |
| Host waits with | — | `waitUntilCompleted` on each committed command buffer | `cuStreamSynchronize` | `hipStreamSynchronize` | `zeCommandListHostSynchronize` |
| Disk-cache artifact | object code | MSL source | PTX | code object | native binary and SPIR-V |
| 3D textures | software, in LLVM | hardware | hardware | hardware, software without image support | hardware, software without samplers |
| `Array.sum/min/max` | NumPy | device kernel for `f32` | device kernel for `f32` | device kernel for `f32` | device kernel for `f32` |

## CPU — LLVM through llvmlite

**Initialization.** `CPUBackend.__init__` (`runtime/cpu.py`) chooses the
thread count from `TACK_CPU_THREADS`, or else `_physical_core_count()`,
which asks each platform for physical cores rather than logical processors.
It creates the variant cache and, at the first fan-out, reaches the
process's thread pool: the pool JIT'd from LLVM IR in `cpu_pool.py`, or a
`ThreadPoolExecutor` with `TACK_CPU_POOL=python`, on Windows, or where the
pool cannot be made ([CPU Threading Policy](cpu-threading.md#what-a-fan-out-costs)).
With `TACK_CPU_MALLOC=1` on Linux it also tells glibc to keep freed pages.
There is no device to select. `allocate_array` returns uninitialized
NumPy storage, first touched by the kernel that writes it.

**Compilation.** `_compile_kernel` generates an `llvmlite.ir` module with
`generate_llvm_ir`, parses and verifies it with `llvm.parse_assembly` and
`mod.verify()`, then optimizes and JIT-compiles:

- `_create_target_machine()` targets the default triple with
  `llvm.get_host_cpu_name()`, the host's feature string, and `opt=3`.
- `_optimize_module()` runs the new pass manager's O3 pipeline
  (`create_pipeline_tuning_options(3)`) with loop vectorization, SLP
  vectorization, loop unrolling and loop interleaving enabled.
- The JIT is `llvm.create_mcjit_compiler` with a *second* target machine,
  because MCJIT takes ownership of the one it is given.

No fast-math flags are emitted, so floating-point semantics follow
[Numerical Semantics](numerical-semantics.md). The entry point is looked up
by `kernel_entry_name()` (`codegen/identifiers.py`), which prefixes
`tack_kernel_` and encodes the name, because llvmlite's symbol lookup needs
ASCII and a kernel must not collide with a libm function.

**Calling convention.** The generated function is

```c
void tack_kernel_<name>(T0* array0, T1 scalar1, ..., int64 __loop_start__, int64 __loop_end__)
```

Arrays are pointers, scalars are passed by value, and the parallel loop runs
over `[__loop_start__, __loop_end__)`. `CompiledKernel` builds a
`ctypes.CFUNCTYPE` for this signature once, with `c_void_p` for each array.
`bind()` marshals the array addresses (`NumpyBuffer.address`, computed once
per array) and scalars once per dispatch; `call_range()` then runs each
chunk with its own bounds, because chunks run concurrently. The JIT'd pool
does not call through ctypes at all: `cpu_pool.chunk_wrapper` JITs, once per
compiled kernel and under a lock, a `__tack_chunk(pack, start, end)` that
loads the arguments from 8-byte slots (`pack_args`) and calls the kernel at
its address.

**Disk cache.** `set_object_cache` captures the object code MCJIT emits, and
`variant_artifact` stores it; `load_variant` adds it to a new engine with
`add_object_file`, which takes about 0.4 ms and runs neither LLVM's
optimizer nor its code generator.

**Launch.** Whether a dispatch runs serially or fans out across the pool is
decided by measurement, not a fixed threshold; [CPU Threading
Policy](cpu-threading.md) describes it in full. A second compiled variant
may exist per specialization, with `noalias` array pointers, for calls whose
arrays are proven not to overlap — see [Memory and
Aliasing](memory-and-aliasing.md#the-cpu-disjoint-arrays-specialization).

**Features.** `supports_workgroups` is false, so kernels using shared
memory, barriers, local thread IDs or block reductions are rejected before
compilation. Textures are sampled in software: `LLVMCodeGen` emits the
trilinear interpolation inline over the texture's private copy of its
array, with the extent baked in as constants.
Reductions use NumPy on the host — the data is already there. Loop indices
are 64-bit.

## Metal

**Initialization.** `MetalBackend.__init__` (`runtime/metal.py`) raises
`ImportError` if `pyobjc-framework-Metal` is missing, takes
`MTLCreateSystemDefaultDevice()` (`RuntimeError` if there is none), and
creates one command queue. There are no options.

**Compilation.** `_compile_kernel` generates MSL with
`generate_msl_source`, compiles it with `newLibraryWithSource_options_error_`
using an `MTLCompileOptions` whose `fastMathEnabled` is false, looks up the
entry function, and builds a compute pipeline state. Any of the three steps
failing raises `RuntimeError`; a library failure includes the MSL source.
Setting `TACK_DUMP_MSL` (a boolean flag read by `dispatch.env_flag`, so `0`,
`false`, `no` and `off` leave it off) writes the source to
`/tmp/tack_<entry>.msl`.

**Arguments.** Array pointers go into one argument buffer — the mechanism
that makes overlapping arrays legal in MSL, explained in [Memory and
Aliasing](memory-and-aliasing.md#how-each-backend-preserves-program-order).
`f64` never reaches the generator in normal dispatch, because it is not in
`supported_dtypes`; `MSLCodeGen` also raises `TypeError` for an `f64`
parameter on its own, since Apple GPUs do not implement double precision.

**Launch.** The threadgroup width is
`min(pipeline.maxTotalThreadsPerThreadgroup(), 256)`. The grid is exactly
`loop_end` threads (`dispatchThreads_threadsPerThreadgroup_`), so the last
threadgroup may be partial and the kernel needs no bounds guard. Each
dispatch encodes a compute pass into the backend's open command buffer
(`_open_batch`), under the backend's batch lock, and returns. The buffer is
committed every `_BATCH` (64) launches, with at most `_IN_FLIGHT` (4)
committed ones outstanding; `synchronize()` commits the open one and waits
for all of them, and a command-buffer error raises `RuntimeError` there,
at the next host access, rather than at the launch's call. Each launch
owns what it reads: a fresh argument buffer, its packed scalars in a buffer
of their own (`_Uniforms`), and references to its arrays until it
completes. A launch that binds memory handed outside Tack (`_shared`: an
exported or wrapped buffer) completes before returning. An
empty range returns before encoding anything. The loop index is declared
`long` but initialized from `[[thread_position_in_grid]]`, which Metal
types as `uint`, so one dispatch covers at most 2³² iterations;
`check_launch_size()` refuses a longer one against `_MAX_LAUNCH` with
`ValueError` before encoding.

## CUDA

**Initialization.** `CUDABackend.__init__` (`runtime/cuda_backend.py`) calls
`cuInit(0)` and `cuDeviceGet(0)`. If a context is already current on the
calling thread — an embedding application such as an AMReX simulation made
one — Tack adopts it. Otherwise it retains the device's primary context
(`cuDevicePrimaryCtxRetain`), which the CUDA runtime and the libraries on it
(PyTorch, CuPy, Warp) use: until 2026-10 Tack created its own with
`cuCtxCreate`, and its next launch after one of those libraries made the
primary context current failed with `CUDA_ERROR_INVALID_HANDLE` (found
comparing with Warp). Either way the context gets a `_ContextToken` that
counts the backends sharing it and records whether Tack owns it; when an
owned context's last backend goes away, a primary context is released
(others may hold it) and any other destroyed, and buffers allocated in it
are marked dead first.
[Memory and Aliasing](memory-and-aliasing.md#when-the-cuda-context-goes-away)
explains why.

!!! note "One thread's context"

    The context is current on the thread that made the device (the first
    `tack.device("cuda")`, whichever call made it). The
    launch path does not make it current on other threads, so a thread that
    dispatches must first call `cuCtxSetCurrent` with the backend's
    context, or its launch fails with `CUDA_ERROR_INVALID_CONTEXT`.

**Compilation.** `_compile_ptx` compiles with NVRTC using exactly these
options, the module constant `_NVRTC_OPTIONS`, which the disk cache's
target also names:

| Option | Why |
|---|---|
| `--ftz=false` | keep denormals rather than flushing to zero |
| `--prec-div=true` | IEEE-rounded division |
| `--prec-sqrt=true` | IEEE-rounded square root |
| `--fmad=true` | allow multiply-add contraction, which the contract permits; the generator emits a product outside a sum as `__fmul_rn`/`__dmul_rn`, since NVRTC also fuses across statements |
| `--extra-device-vectorization` | let NVRTC vectorize more aggressively |

`--use_fast_math` is never passed. No target architecture is given, so
NVRTC emits PTX for its default virtual architecture and the driver
JIT-compiles that PTX in `cuModuleLoadData`. A compile failure raises
`RuntimeError` with the NVRTC log and the source; the program is destroyed
on both paths.

The first compile in a process checks `nvrtcVersion` and warns if NVRTC is
older than 13 (`runtime/nvrtc_version.py`). NVRTC 12.0 miscompiled
tack-examples' lattice Boltzmann example on an H100 (density off by up to
1.6e-2; 2.3e-6 with 13.2), and was not reduced to a kernel, so there is no
workaround to apply. It also left the compiling thread in the C locale,
which `_compile_keeping_thread_locale` undoes. The check runs at the first
compile rather than when the device is made, so a machine without NVRTC fails where
it did before.

**Launch.** Block size is `WORKGROUP_SIZE` (256, from
`lang/workgroup_participation.py`) and the grid is
`(loop_end + 255) // 256` blocks. The kernel receives the loop end as a
`long long __n__` parameter and begins

```c
long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
if (i >= __n__) return;
```

Locals and loop indices are `long long`. The built-ins are 32-bit
unsigned, so `blockIdx.x` is widened *before* the multiply; a 32-bit
product would wrap once a launch reached 2³² threads. At initialization the
backend reads `CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X`, and a launch of more
than that many blocks of 256 is refused by `check_launch_size()` with
`ValueError`. Arguments are passed as a ctypes array of pointers to argument
values, and the launch is queued on the context's stream (`_ContextToken`
makes one per context, with `cuStreamCreate`, on first use) and returns. An
empty range returns before launching, since `cuLaunchKernel` rejects an
empty grid. Buffers are stream-ordered: `cuMemAllocAsync` and
`cuMemFreeAsync` on that stream, from the device's default memory pool
with its release threshold raised, so freed memory is handed back out
without a device round trip, and an array dropped while a queued launch
still uses it is freed after that launch. A device without memory pools
(a GPU in TCC mode on Windows) allocates with `cuMemAlloc` instead; since
`cuMemFree` waits for the whole device, a dropped buffer is queued on its
context and freed once the stream is idle, at the next allocation or sync
(`_free_dropped`). `from_numpy`, `fill` and the
allocation's zeroing are queued in order with the launches; `to_numpy` and
`read_range` queue their copy and wait with `cuStreamSynchronize`.

**Disk cache.** The PTX NVRTC made is stored with the entry name, the
parameter layout and the scalar pack layout, and loaded with
`cuModuleLoadData`; `load_variant` makes the pack arrays anew from the
call's arguments, since they are device memory of the process.

**Export.** `allocate_array(..., exportable=True)` returns an
`ExportableCUDABuffer`, allocated through the virtual memory management API
(`cuMemCreate` / `cuMemMap`) so it can be shared as an OS handle; see
[Interoperability](interoperability.md#exporting-device-memory-to-other-apis).

## HIP

**Initialization.** `HIPBackend.__init__` (`runtime/hip_backend.py`) calls
`hipInit(0)` and uses the current device from `hipGetDevice()`. Tack does
not create or destroy a context; the HIP runtime manages it. The backend
then asks whether the device has texture hardware at all
(`hipDeviceAttributeImageSupport`) and, if so, the smallest of its maximum
3D texture extents.

**Compilation.** `_compile_code_object` creates a hipRTC program and
compiles it with the options in `_HIPRTC_OPTIONS`, which is one:
`-ffp-contract=on`. hipRTC's default is clang's for HIP,
`-ffp-contract=fast`, which fuses a multiply into an add or subtract in a
later statement; on AMD GPUs it did so where the product had another use,
so one variable held two values (with `p = k * 0.2`, `p - p` came out as
the product's rounding error instead of zero), and a contour of a computed
array took finite values for infinite ones. `on` fuses only inside one
expression, which is what the contract permits. hipRTC targets the
current device by default, and no unsafe or relaxed math option is
requested. The result is a code object loaded with `hipModuleLoadData`.
The disk cache stores that code object, machine code for one
architecture, with the entry name, parameter and scalar pack layouts, and
`load_variant` remakes the pack arrays from the call's arguments.
`HIPCodeGen` (`codegen/hip_gen.py`) is the CUDA generator with two
overrides: it prepends `#include <hip/hip_runtime.h>`, and it spells the
texture handle `hipTextureObject_t` (`_TEXTURE_OBJECT_TYPE`), because
hipRTC rejects CUDA's `cudaTextureObject_t`.

**Launch.** Identical geometry to CUDA — 256-thread blocks, a ceiling-divided
grid, the same 64-bit index, guard and locals — through
`hipModuleLaunchKernel`, queued as on CUDA. HIP has no context object to
hold the stream, so it belongs to the device (`_Queue`, a blocking stream
made on first use, so it stays ordered with the null stream), and each
buffer keeps a reference to it. Buffers are stream-ordered
(`hipMallocAsync`/`hipFreeAsync`, with the default pool's release
threshold raised); a device without memory pools falls back to
`hipMalloc` and `hipFree`, which waits for the device. A host write from
pageable memory is queued behind the launches; measured on ROCm 7.1, HIP
stages a small one (16 KB) before `from_numpy` returns and waits for the
stream before a large one (4 MB), so either way the array gets what the
array held when the call returned (`test_hip_async.py`). The launch
limit (`_query_max_launch`) is the device's maximum `gridDim.x` times 256,
capped at 2³² − 256, because the dispatch packet on AMD GPUs holds the
grid's total work-item count in 32 bits.

**hip-python 10 is a compatibility package** over `rocm.bindings` (PyPI's newest, so the extra's lower bound resolves to it), and four things moved: no `hip.LONG_VERSION` (`VERSION` instead; `_HIP_PYTHON_VERSION` reads either), no `hip.chip` (the runtime library for the ctypes waits comes from `rocm.bindings.util.paths.get_library_path`, `_runtime_library`), a function whose first C parameter is an output struct allocates and returns it, `(err, struct)`, instead of taking it (`hipPointerGetAttributes`, `hipGetDeviceProperties`; `_call_with_output` reads which from the binding's signature); and a struct's character arrays (`hipDeviceProp_t.name`, `gcnArchName`) are `str`, not `bytes` (`_text`). Before this, importing `LONG_VERSION` failed, so HIP read as unavailable and every HIP test skipped. With them the suite passes on hip-python 10.0 against ROCm 7.1.1 (MI210, 2026-10-15) as on 7.2.2.

**Known toolchain defect.** ROCm 7.0.2's device compiler (AMD clang 20,
embedded in hipRTC's comgr) miscompiles one generated integer expression
at `-O1` and above: a 64-bit signed `-253 < -1` inside the full expression
evaluates false. The generated source is free of undefined behavior (UBSan
under clang and gcc, `-O0` to `-O3`) and is correct on CUDA and under
ROCm's clang 23, so it's a compiler defect and not a code generation one.
It depends on the surrounding expression, so Tack doesn't try to avoid it.
ROCm 7.1.1, still AMD clang 20, does the same: on an MI210 (gfx90a) the
generated kernel, built by `hipcc` into a host program with no Tack
involved, gives the right answer at `-O0` and the wrong one at `-O1` to
`-O3`. `test_integer_expression_differential.py` marks that one case as a
strict expected failure when the HIP runtime reports ROCm 7.0 or 7.1
(`hipRuntimeGetVersion()`). `hiprtcVersion()` can't identify the release:
it reports 9.0 on ROCm 7.0.2. A fixed toolchain reports XPASS, and the
mark is then removed.

## Level Zero

**Initialization.** `LevelZeroBackend.__init__`
(`runtime/level_zero_backend.py`) loads `libze_loader.so` through ctypes and
calls `zeInit(0)`. Without options it takes the first driver and that
driver's first device and creates its own context. With
`external_context={"driver", "device", "context"}` it adopts all three
handles from another library instead — see
[Interoperability](interoperability.md#level-zero-one-context-shared-out-of-band).
`_query_device` then reads:

| Query | Used for |
|---|---|
| `zeDeviceGetProperties` | `deviceId`, passed to `ocloc -device`; the device name |
| `zeDeviceGetComputeProperties` | `maxGroupSizeX` and `maxTotalGroupSize`, which cap the group size, and `maxGroupCountX`, which with it bounds one launch |
| `zeDeviceGetImageProperties` | `maxImageDims3D`, and `maxSamplers > 0` as "has texture samplers" |
| `zeDeviceGetModuleProperties` | `fp64flags != 0` adds `f64` to `supported_dtypes` |
| `zeDeviceGetCommandQueueGroupProperties` | the first queue group with the compute flag |

`supported_dtypes` is assigned on the instance, shadowing the class
attribute, so `supports_f64` follows the device automatically.
`_create_queues` makes the backend's queue: one asynchronous immediate
command list, to which every launch, fill and copy is appended, each
followed by a barrier (`_ordered`) that keeps them in program order. A
driver without `zeCommandListHostSynchronize` (API 1.6) gets a synchronous
immediate list instead, every append completing at once. The list, and
each kernel's argument state, are shared by every dispatch on the backend,
and Level Zero forbids using them from two threads at once, so a reentrant
`_launch_lock` is held around every launch, native reduction and host
copy.

**Compilation.** OpenCL C from `generate_opencl_source` is compiled to
SPIR-V *in process* by `_compile_to_spirv`, which calls `oclocInvoke` in
`libocloc.so` with

```
compile -spv_only -device 0x<deviceId> -options "-cl-std=CL2.0 -cl-fp32-correctly-rounded-divide-sqrt" -file kernel.cl
```

and passes the source from memory, null-terminated, because ocloc treats
in-memory sources as C strings. The options are the module constant
`_OCLOC_OPTIONS`. The `.spv` output is handed to `_create_module`, which
calls `zeModuleCreate` as `ZE_MODULE_FORMAT_IL_SPIRV` with build flags
`_MODULE_BUILD_FLAGS` (the same OpenCL options, `_OPENCL_OPTIONS`); a failure raises `RuntimeError`
with the module build log and the source. `zeKernelCreate` then returns the kernel handle. When the
source uses `double`, the generator adds
`#pragma OPENCL EXTENSION cl_khr_fp64 : enable`.

**The cost of ocloc.** Compilation on this backend is dominated by
`libocloc` itself. Measured on an Intel Data Center GPU Max 1100 on
2026-10-04, idle, an ocloc call took about 191 ms regardless of kernel size,
against 0.2 ms for `zeModuleCreate`; llvmlite on the CPU backend compiled
the same kernel shapes in 21–55 ms. The floor is startup inside the
library, not code generation. The disk cache is what avoids it: a variant
is stored as the native binary IGC made at `zeModuleCreate`
(`zeModuleGetNativeBinary`) beside the SPIR-V, and loaded as
`ZE_MODULE_FORMAT_NATIVE`, about 0.2 ms a module in a new process; a binary the
driver refuses loads from the SPIR-V instead (counted as
`native_refused`). SPIR-V alone still went through IGC at `zeModuleCreate`,
about 50 ms a kernel, unless the driver's own persistent cache, which lives
outside `TACK_CACHE_DIR` and is off or unwritable on many clusters, held
the result. The cache target therefore names the device and its IP version
(which names its stepping), the driver, ocloc's version and options, the
module build flags and f64 support. With the disk cache off, a device made
with options recompiles the same sources, because the in-memory variant
cache belongs to a backend instance; the plain `tack.device("level_zero")`
is one backend for the process, so switching the default away and back
keeps its variants.

**Launch.** The group size is `min(256, maxGroupSizeX, maxTotalGroupSize)`.
Each argument is set with `zeKernelSetArgumentValue`, followed by the loop
end as a 64-bit `__n__`; the kernel starts with
`long i = (long)get_group_id(0) * (long)get_local_size(0) + (long)get_local_id(0);
if (i >= __n__) return;`, because this driver's `get_global_id(0)` wraps
at 2^32 (see [Parallel execution](parallel-execution.md)). Dispatch appends
the launch with `ceil(loop_end / group_size)` groups to the queue, followed
by a barrier, and returns; the kernel's arguments are captured when the
launch is appended, so the next call may set its own. The host waits with
`zeCommandListHostSynchronize` where it reads an array or calls
`tack.sync()`. Ordering by barriers was chosen over an in-order list
(`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`): on driver 25.05.32567 an in-order list,
immediate or regular, cost about 130 µs of device time per kernel, a
barrier about 5. `from_numpy` queues a copy of a private snapshot of the
array, so the caller may change it at once; host memory a queued copy or
fill reads is held until the next wait, which comes early past 1024 objects
or 256 MiB. Arrays are freed with `zeMemFreeExt`'s deferred policy, after
the work queued so far, where the driver has the extension; otherwise
freeing waits for the queue. An empty
range returns first; a negative count would wrap as an unsigned group count.
The group count is a ctypes `uint32` array that truncates rather than
rejects, so a launch of more than `maxGroupCountX` groups is refused by
`check_launch_size()` first.

`LevelZeroBackend.__del__` intentionally does nothing: Level Zero cleanup
at interpreter shutdown can crash because the driver may already be
unloaded.

## Launches are queued

The CPU's dispatch returns once the range has run. The four GPU backends
queue a launch and return: Metal into the open command buffer, CUDA on one
stream per context, HIP on one stream per device, Level Zero on one
asynchronous immediate command list. Waiting after every launch kept
Python from preparing launch n + 1 while launch n ran, and for short
launches the host side was most of the call: on Metal a wait cost about
0.3 ms; a contour of a million cells on an RTX 4060 Ti went from 9.6 to
7.1 ms, about the host cost of its 31 launches alone; on an Intel Max 1100
a hundred small launches went from 8.7 to 2.6 ms.

- **Order.** Launches, fills and host-to-device copies on one backend run
  in the order they were made. On CUDA, HIP and Level Zero `from_numpy`
  and `fill` are queued with the launches; Metal's `MetalBuffer` methods
  wait for every live Metal backend's launches (`_synchronize_all`)
  before touching the shared memory.
- **Waits.** The host waits where it reads an array (`to_numpy`,
  `read_range` and so host indexing, reductions), where memory leaves
  Tack (`share()`, below), after each launch in debug mode (its check
  reads the errors array), at `Device.synchronize()`, and at `tack.sync()`,
  which calls every live backend's `synchronize()`. A launch's error surfaces at the next wait
  rather than at its call.
- **Lifetimes.** An array Python drops while a queued launch still uses it
  outlives that launch: CUDA and HIP allocate and free in stream order,
  Level Zero frees with the driver's deferred policy, and a Metal launch
  holds references to its arrays until it completes.
- **Memory outside Tack.** A wrapped pointer (`wrap_ptr`, so
  `array_from_ptr` and DLPack imports), a DLPack export and Metal's
  `export_memory()` mark the buffer shared (`DeviceBuffer.share()`): the
  export waits for the queued launches, and every later launch that binds
  a shared buffer completes before it returns, because the other library
  may read the memory with no Tack call in between. CUDA's
  `export_memory()` of an ordinary array waits, then copies it once into
  exportable memory. CUDA, HIP and Level
  Zero also complete a launch that samples a hardware texture, whose
  image could otherwise be destroyed before the launch read it.

`test_cuda_async.py`, `test_hip_async.py` and `test_level_zero_async.py`
cover order across launches and fills, per-launch scalars, host writes
while the queue is busy, dropped temporaries, waits and shared buffers;
`test_metal.py` covers Metal's order across batches, per-launch scalars
and host writes.

### How each backend queues and waits, as measured

The measurements behind each backend's queue and waits, kept from the
developer notes:

- **CUDA**: `CUDABuffer` holds a stream-ordered device pointer (`cuMemAllocAsync` on the context's stream, freed with `cuMemFreeAsync`, so a temporary dropped while a queued launch still uses it is freed after that launch, and the driver's pool, with its release threshold raised, hands memory back out without a round trip). Every launch, fill and copy goes through that one stream (`_ContextToken.queue()`) in order and returns once queued; `to_numpy`, `read_range` and reductions wait for the stream (`_drain`) and then copy: a copy to pageable host memory waits inside the driver with the context held, so no other thread could launch meanwhile (one launch per 910 ms wait on an H100, driver 595), while a thread in `cuStreamSynchronize` holds nothing a launch needs (511 launches at 43 µs); `from_numpy` and `fill` queue and return; `tack.sync()` is a stream synchronize, and a launch's error surfaces at the next wait. A launch's scalar upload is a pageable `cuMemcpyHtoDAsync`, staged by the driver; behind a long kernel the staging ring fills after about 510 queued launches and the next upload waits for the kernel. A buffer handed outside Tack (`wrap_ptr`, DLPack export, `share()`) completes every launch that uses it before the launch returns, and so does a texture launch, since its image could be destroyed before a queued launch read it. See `test_cuda_async.py`.
- **HIP**: `HIPBuffer` holds a stream-ordered device pointer, as CUDA's does: one blocking stream per device (`_Queue`, `_queue()` in `hip_backend.py`; ordered with the null stream), memory from `hipMallocAsync` on it and freed with `hipFreeAsync`, the default pool's release threshold raised. Launches, fills and copies are queued on it and return; `to_numpy`, `read_range`, reductions and `tack.sync()` wait on an event recorded after the queued work (`_StreamMarker`, `_wait_stream`), through ctypes on the runtime library (`_runtime`), never `hipStreamSynchronize`: hip-python's wrappers hold the GIL while the runtime waits (another thread ran 6 times during a 357 ms wait, 353 through ctypes), and ROCm 7.1's stream synchronize holds the stream, so another thread's launch on it took the whole wait (350 ms; 60 µs waiting on an event). A copy to pageable host memory waits inside `hipMemcpyAsync` the same way, so a read waits on an event first (about 8 µs idle). A launch's error surfaces at the next wait. A host write from pageable memory may be queued: HIP stages a small one before returning and waits for the stream before a large one (16 KB staged, 4 MB waited, ROCm 7.1). `wrap_ptr`, DLPack export (`share()`) and texture launches complete before returning. A device without memory pools falls back to `hipMalloc`/`hipFree`. See `test_hip_async.py`.
- **Level Zero**: `L0Buffer` holds a device pointer (`zeMemAllocDevice`). Every launch, fill and copy is appended to the backend's one asynchronous immediate command list (`_queue`) and returns once queued, each followed by a barrier (`_ordered`), which keeps program order: Intel's in-order lists (`ZE_COMMAND_QUEUE_FLAG_IN_ORDER`) cost about 100 µs of device time per kernel on driver 25.05, a barrier about 5. `to_numpy`, `read_range` and reductions queue a copy and a marker under `_launch_lock` and wait on the marker's event outside it (`_ListMarker`), so other threads launch meanwhile; `tack.sync()` is `zeCommandListHostSynchronize`; `from_numpy` copies the array privately and queues; copies to the device up to 16 MiB and from it up to 64 KiB go through pooled USM host memory (`_HostStaging`, guarded by `_launch_lock`), since from pageable memory Intel's driver does not return from appending one (to the device up to 4 MiB, from it up to 1 KiB, driver 25.05) until the queue drains, which made a launch's scalar upload wait for the kernels before it; host memory a queued copy or fill reads is held (`_hold`) until the next wait (a marker releases the holds before it), which comes early past 1024 objects or 256 MiB. Arrays are freed with `zeMemFreeExt(DEFER_FREE)`, after the work queued so far (`_free`; without the extension, wait then free). `share()`, `wrap_ptr`, DLPack export and textures complete their launches before returning, as on CUDA. A driver without `zeCommandListHostSynchronize` gets a synchronous list. See `test_level_zero_async.py`.
- **`asyncio`** (`runtime/aio.py`, optional): `await tack.sync_async()` and `await array.numpy_async()` wait in one of Tack's worker threads (`tack-wait`, 4) and resume the coroutine, so an event loop (trame, Jupyter, a server) keeps running while the device works; kernel calls stay plain calls. A backend opts in with `Backend.async_marker()` -- an object whose blocking `wait()` covers every launch queued so far and raises a launch's error: CUDA records a blocking-sync event on the context's stream (`_StreamMarker`), Metal commits the open batch and waits on the last command buffer (`_CommandBufferMarker`), Level Zero appends a barrier signalling a host-visible event from the backend's pools, reset and reused after the wait (`_ListMarker`, `_take_event`), HIP records a blocking-sync event on the device's stream and waits through ctypes (`_StreamMarker`) -- and buffers with `DeviceBuffer.waits_off_thread` and `thread_binding()` (CUDA makes its context current in the worker, `_BoundContext`; HIP makes the buffer's device current, `_BoundDevice`). Without a marker (the CPU) the wait is inline: correct, and the loop simply waits. The mark is taken at the call, so launches queued later are not waited for. cuda-bindings releases the GIL in `cuEventSynchronize`: during a 490 ms kernel another task ran 436 times awaiting `sync_async` and 0 times under `tack.sync()`; Level Zero's ctypes calls release it too (311-384 times during a 685 ms kernel on a Max 1100); hip-python's do not, hence HIP's ctypes waits (1353 times during a 1.45 s kernel on an MI210). See `test_asyncio.py`.

## Scalar arguments on the GPU

The CPU passes scalars by value. The four GPU backends instead run
`pack_scalars` (`lang/ir_pack_scalars.py`) on a copy of the variant's IR in
their `build` callback: every scalar parameter is replaced by a load from a
small array, one array per scalar dtype (`__pack_f32__`, `__pack_i32__`,
...), marked `_is_scalar_pack`. The pass was written for Metal, whose
kernels have a limited number of buffer bindings; packing makes the count
one per scalar type instead of one per scalar, and the other GPU backends
share the code path.

On CUDA, HIP and Level Zero the pack arrays are allocated once, when the
variant is built (`_create_pack_arrays` in `runtime/kernel_utils.py`), and
stored in the variant's payload. Each dispatch writes the new values with
`_update_pack_arrays`, a host-to-device copy queued after the launches
before it, so a queued launch still reads the values it was given. Metal
keeps no pack arrays: each launch copies its values into a new buffer
(`_Uniforms`) that the launch owns, since a shared pack rewritten for the
next call could change under a launch that has not run yet. Changing a
scalar's value therefore never recompiles. The loop range is resolved from
the *unpacked* IR, because packing rewrites the parameter list that the
range expression refers to.

Since the pack arrays belong to the variant, two threads dispatching the
same variant at once would each write their arguments into the same
buffers, and a launch could run with the other call's scalars. Reproduced
on CUDA with four threads, 248 of 400 dispatches computed with another
thread's scalars. CUDA and HIP therefore hold `KernelVariant.dispatch_lock`
from the pack update until the launch is queued, so different variants
still dispatch concurrently; uncontended, the lock cost about 0.25 µs
against a 95 µs CUDA dispatch. Level Zero takes its backend-wide
`_launch_lock` instead (see [Level Zero](#level-zero)). Metal launches own
their argument buffer and scalars and encode one at a time under the
backend's batch lock. The CPU binds arguments per call and takes no lock.

## Textures

`tack.texture3d(array, shape)` creates a `Texture3D` (`lang/array.py`)
holding a copy of the array; in a kernel, `tex.sample(u, v, w)`
interpolates trilinearly at normalized coordinates, where `0` and `1` are
the centers of the first and last texels. Hardware samplers put texel
centers at `(i + 0.5) / N`, so every hardware path rewrites each coordinate
to `(u * (N - 1) + 0.5) / N`.

**Storage.** The texture copies its array when it is created, on every
backend, and `tex.update()` copies it again; writes to the array in
between don't reach it. The copy is storage the texture owns, in
`tex._storage`. `Backend.texture_in_hardware(shape_3d)` chooses it: where
the answer is true, `create_texture_image()` returns a backend image
object (`MetalTextureImage`, `CUDATextureImage`, `HIPTextureImage` or
`L0TextureImage`) whose `upload(array)` does the copy; otherwise the
texture allocates a private `f32` array and `update()` copies into it with
the device copy kernel. Only one extra copy is held per texture. Images
are built eagerly and updated in place, so their texture objects stay
valid, and they free their device objects when the texture is collected
(CUDA's only while its context is alive). At dispatch, `bind_textures()`
(`runtime/kernel_utils.py`) replaces each `Texture3D` argument with its
storage, never its array. No compiled kernel caches texture objects.

| Backend | Hardware path | Software fallback when |
|---|---|---|
| CPU | none | always |
| Metal | `MTLTexture` (3D, `R32Float`), filled by a blit from the array's buffer; `sampler(normalized, linear, clamp_to_edge)` | never |
| CUDA | 3D CUDA array (one 32-bit float channel) filled with `cuMemcpy3D`; texture object with linear filtering, clamp, normalized coordinates | never |
| HIP | `hipMalloc3DArray` + `hipMemcpy3D`; texture object as on CUDA | the device reports no image support, or an extent exceeds its 3D texture limit |
| Level Zero | `zeImageCreate` (3D, 32-bit float), filled through a host staging allocation; `CLK_NORMALIZED_COORDS_TRUE \| CLK_ADDRESS_CLAMP_TO_EDGE \| CLK_FILTER_LINEAR` | `maxSamplers == 0` (Xe-HPC), or an extent exceeds `maxImageDims3D` |

The fallback decision changes the generated code, so HIP and Level Zero
ask `texture_in_hardware()` again in their own `_store_texture_shapes`,
passed to `resolve_variant`, which runs before the variant key is built.
The texture and the variant therefore agree on which storage is bound.
Falling back clears the parameter's `_is_texture` flag and the generator
emits a software trilinear helper over the texture's private array instead.
`tack.inspect` calls the same hook, so it shows the code dispatch compiles.

CDNA parts — gfx940/941/942, the MI300 family — have no texture hardware,
and hipRTC refuses `tex3D` outright there ("The image/texture API not
supported on the device"), which is why HIP asks
`hipDeviceAttributeImageSupport` rather than assuming. Xe-HPC similarly has
no samplers; filtered image reads would be driver-emulated.

**Why a snapshot on every backend.** Hardware samplers read an image, not
the array, so on those paths a texture is a copy whatever Tack does. The
only portable rule is therefore that a texture is a copy everywhere, taken
at points the program chooses: at creation and on `update()`. Owning the
image in the texture, rather than caching it in a compiled kernel under
the array's device address, means no texture is ever handed another's
image (a new array that reuses a freed array's address, or a second
texture over the same array, gets its own copy), and the image is released
with its texture. The images are single-channel 32-bit float, so
`texture3d()` rejects non-`f32` arrays rather than reinterpreting them,
and it rejects shapes whose `W*H*D` differs from the array's size, which
the copies would read past. `render_volume()` calls `update()` before each
render, since the mixed-scene path tracer reads the volume's array
directly. `test_texture_snapshot.py` pins these rules.

## Device reductions

`Array.sum()`, `min()` and `max()` call `backend.reduce_array()` when
`supports_device_reductions` is true and otherwise reduce with NumPy on the
host. All four GPU backends implement `reduce_array` only for `f32` arrays,
using a source from `array_reduction_source(dialect, op)`
(`codegen/reductions.py`): a 256-lane tree in each group, followed by
unordered atomic accumulation of the partial results. Any other dtype falls
back to NumPy, as does Level Zero when the device's group-size limits are
below 256. CUDA, HIP and Level Zero pass the element count as a 64-bit
argument and check it with `check_launch_size()`. Metal's kernel holds the
count in 32 bits, so Metal reduces an array of 2³² or more elements with
NumPy over its shared buffer. An empty array returns `0.0` for `sum` and raises `ValueError`
for `min` and `max` on every backend. The ordering and accuracy promises are
in the [contract](../reference/language-contract.md#array-and-parallel-reductions).

## Workarounds and why they exist

Generated code and runtime calls shaped around a vendor defect, each with
the vendor and version it was found on and the test that guards it, are
recorded in [Vendor workarounds](vendor-workarounds.md), so that each can
be retired when a fix ships.

## See also

- [Parallel Execution](parallel-execution.md) — the execution model these
  launches implement.
- [Memory and Aliasing](memory-and-aliasing.md) — the buffers and the
  aliasing rules each generator honors.
- [Interoperability](interoperability.md) — sharing memory with other
  libraries.
- [Conformance and Validation](../contracts/conformance.md) — how each
  backend is validated on hardware.

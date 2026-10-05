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

All five backends subclass `Backend` (`runtime/backend.py`). It declares
three required methods — `allocate_field()`, `wrap_ptr()` and `execute()` —
and a set of capability attributes that callers *read* instead of probing
with `hasattr`. Anything derivable is derived: `supports_f64` is a property
over `supported_dtypes`, and `supported_atomic_dtypes` is a property that
intersects the per-target table `ATOMIC_DTYPES` (`lang/atomic_support.py`)
with `supported_dtypes`, so neither can disagree with the field types.

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
| `init_options` | `num_threads` | — | — | — | `external_context` |
| `dlpack_refusal_note` | — | set | — | — | — |

"All ten" is `i8`, `u8`, `i16`, `u16`, `i32`, `u32`, `i64`, `u64`, `f32`,
`f64`. `supported_dtypes` is a `frozenset` on every backend. `init_options` lists the keyword arguments `tack.init()` forwards to
the backend's constructor; `tack.init()` (`runtime/dispatch.py`) raises
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
| Field binding | typed pointer argument | pointer in one argument buffer | pointer argument | pointer argument | `__global` pointer argument |
| Scalar binding | by value | packed into a field per dtype | packed | packed | packed |
| Synchronization | the call returns | `waitUntilCompleted` | `cuCtxSynchronize` | `hipDeviceSynchronize` | `zeCommandQueueSynchronize` |
| 3D textures | software, in LLVM | hardware | hardware | hardware, software without image support | hardware, software without samplers |
| `Field.sum/min/max` | NumPy | device kernel for `f32` | device kernel for `f32` | device kernel for `f32` | device kernel for `f32` |

## CPU — LLVM through llvmlite

**Initialization.** `CPUBackend.__init__` (`runtime/cpu.py`) chooses the
thread count from `TACK_CPU_THREADS`, or else `_physical_core_count()`,
which asks each platform for physical cores rather than logical processors.
It creates the variant cache and, lazily, a `ThreadPoolExecutor`. There is
no device to select.

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
void tack_kernel_<name>(T0* field0, T1 scalar1, ..., int64 __loop_start__, int64 __loop_end__)
```

Fields are pointers, scalars are passed by value, and the parallel loop runs
over `[__loop_start__, __loop_end__)`. `CompiledKernel` builds a
`ctypes.CFUNCTYPE` for this signature once. `bind()` marshals the field
pointers and scalars once per dispatch; `call_range()` then runs each chunk
with fresh `c_int64` bounds, because chunks run concurrently on the pool.

**Launch.** Whether a dispatch runs serially or fans out across the pool is
decided by measurement, not a fixed threshold; [CPU Threading
Policy](cpu-threading.md) describes it in full. A second compiled variant
may exist per specialization, with `noalias` field pointers, for calls whose
fields are proven not to overlap — see [Memory and
Aliasing](memory-and-aliasing.md#the-cpu-disjoint-fields-specialization).

**Features.** `supports_workgroups` is false, so kernels using shared
memory, barriers, local thread IDs or block reductions are rejected before
compilation. Textures are sampled in software: `LLVMCodeGen` emits the
trilinear interpolation inline over the texture's private copy of its
field, with the extent baked in as constants.
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

**Arguments.** Field pointers go into one argument buffer — the mechanism
that makes overlapping fields legal in MSL, explained in [Memory and
Aliasing](memory-and-aliasing.md#how-each-backend-preserves-program-order).
`f64` never reaches the generator in normal dispatch, because it is not in
`supported_dtypes`; `MSLCodeGen` also raises `TypeError` for an `f64`
parameter on its own, since Apple GPUs do not implement double precision.

**Launch.** The threadgroup width is
`min(pipeline.maxTotalThreadsPerThreadgroup(), 256)`. The grid is exactly
`loop_end` threads (`dispatchThreads_threadsPerThreadgroup_`), so the last
threadgroup may be partial and the kernel needs no bounds guard. Each
dispatch creates a command buffer and a compute encoder, commits, and waits
with `waitUntilCompleted`; a command-buffer error raises `RuntimeError`. An
empty range returns before encoding anything. The loop index is declared
`long` but initialized from `[[thread_position_in_grid]]`, which Metal
types as `uint`, so one dispatch covers at most 2³² iterations;
`check_launch_size()` refuses a longer one against `_MAX_LAUNCH` with
`ValueError` before encoding.

## CUDA

**Initialization.** `CUDABackend.__init__` (`runtime/cuda_backend.py`) calls
`cuInit(0)` and `cuDeviceGet(0)`. If a context is already current on the
calling thread — an embedding application such as an AMReX simulation made
one — Tack adopts it. Otherwise it creates one with `cuCtxCreate`. Either
way the context gets a `_ContextToken` that counts the backends sharing it
and records whether Tack owns it; only an owned context whose last backend
goes away is destroyed, and buffers allocated in it are marked dead first.
[Memory and Aliasing](memory-and-aliasing.md#when-the-cuda-context-goes-away)
explains why.

!!! note "One thread's context"

    The context is current on the thread that called `tack.init()`. The
    launch path does not make it current on other threads, so a thread that
    dispatches must first call `cuCtxSetCurrent` with the backend's
    context, or its launch fails with `CUDA_ERROR_INVALID_CONTEXT`.

**Compilation.** `_compile_ptx` compiles with NVRTC using exactly these
options:

| Option | Why |
|---|---|
| `--ftz=false` | keep denormals rather than flushing to zero |
| `--prec-div=true` | IEEE-rounded division |
| `--prec-sqrt=true` | IEEE-rounded square root |
| `--fmad=true` | allow multiply-add contraction, which the contract permits |
| `--extra-device-vectorization` | let NVRTC vectorize more aggressively |

`--use_fast_math` is never passed. No target architecture is given, so
NVRTC emits PTX for its default virtual architecture and the driver
JIT-compiles that PTX in `cuModuleLoadData`. A compile failure raises
`RuntimeError` with the NVRTC log and the source; the program is destroyed
on both paths.

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
values; every launch is followed by `cuCtxSynchronize()`. An empty range
returns before launching, since `cuLaunchKernel` rejects an empty grid.

**Export.** `allocate_field(..., exportable=True)` returns an
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

**Compilation.** `_compile_code_object` creates a hipRTC program and calls
`hiprtcCompileProgram(prog, 0, [])` — no options. hipRTC targets the
current device by default, and no unsafe or relaxed math option is
requested. The result is a code object loaded with `hipModuleLoadData`.
`HIPCodeGen` (`codegen/hip_gen.py`) is the CUDA generator with two
overrides: it prepends `#include <hip/hip_runtime.h>`, and it spells the
texture handle `hipTextureObject_t` (`_TEXTURE_OBJECT_TYPE`), because
hipRTC rejects CUDA's `cudaTextureObject_t`.

**Launch.** Identical geometry to CUDA — 256-thread blocks, a ceiling-divided
grid, the same 64-bit index, guard and locals — through
`hipModuleLaunchKernel`, followed by `hipDeviceSynchronize()`. The launch
limit (`_query_max_launch`) is the device's maximum `gridDim.x` times 256,
capped at 2³² − 256, because the dispatch packet on AMD GPUs holds the
grid's total work-item count in 32 bits.

**Known toolchain defect.** ROCm 7.0.2's device compiler (AMD clang 20,
embedded in hipRTC's comgr) miscompiles one generated integer expression
at `-O1` and above: a 64-bit signed `-253 < -1` inside the full expression
evaluates false. The generated source is free of undefined behavior (UBSan
under clang and gcc, `-O0` to `-O3`) and is correct on CUDA and under
ROCm's clang 23, so it's a compiler defect and not a code generation one.
It depends on the surrounding expression, so Tack doesn't try to avoid it.
`test_integer_expression_differential.py` marks that one case as a strict
expected failure when hipRTC reports 7.0. A fixed toolchain reports XPASS,
and the mark is then removed.

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
`_create_queues` makes a synchronous command queue, one reusable command
list for kernel launches, and one synchronous immediate command list for
memory copies. Both lists, and each kernel's argument state, are shared by
every dispatch on the backend, and Level Zero forbids using them from two
threads at once, so a reentrant `_launch_lock` is held around every launch,
native reduction and host copy.

**Compilation.** OpenCL C from `generate_opencl_source` is compiled to
SPIR-V *in process* by `_compile_to_spirv`, which calls `oclocInvoke` in
`libocloc.so` with

```
compile -spv_only -device 0x<deviceId> -options -cl-std=CL2.0 -file kernel.cl
```

and passes the source from memory, null-terminated, because ocloc treats
in-memory sources as C strings. The `.spv` output is handed to
`zeModuleCreate` as `ZE_MODULE_FORMAT_IL_SPIRV` with build flags
`-cl-std=CL2.0`; a failure raises `RuntimeError` with the module build log
and the source. `zeKernelCreate` then returns the kernel handle. When the
source uses `double`, the generator adds
`#pragma OPENCL EXTENSION cl_khr_fp64 : enable`.

**The cost of ocloc.** Compilation on this backend is dominated by
`libocloc` itself. Measured on an Intel Data Center GPU Max 1100 on
2026-10-04, idle, an ocloc call took about 191 ms regardless of kernel size,
against 0.2 ms for `zeModuleCreate`; llvmlite on the CPU backend compiled
the same kernel shapes in 21–55 ms. The floor is startup inside the
library, not code generation. Because the variant cache belongs to a
backend instance and `tack.init()` builds a new backend, a program that
re-initializes recompiles the same sources. Memoizing
`_compile_to_spirv(source, device_id)` — a pure function of its inputs —
would capture almost all of that cost; it has not been done.

**Launch.** The group size is `min(256, maxGroupSizeX, maxTotalGroupSize)`.
Each argument is set with `zeKernelSetArgumentValue`, followed by the loop
end as a 64-bit `__n__`; the kernel starts with
`long i = (long)get_group_id(0) * (long)get_local_size(0) + (long)get_local_id(0);
if (i >= __n__) return;`, because this driver's `get_global_id(0)` wraps
at 2^32 (see [Parallel execution](parallel-execution.md)). Dispatch resets the
reusable command list, appends the launch with
`ceil(loop_end / group_size)` groups, closes it, executes it on the queue,
and waits with `zeCommandQueueSynchronize` and an infinite timeout. An empty
range returns first; a negative count would wrap as an unsigned group count.
The group count is a ctypes `uint32` field that truncates rather than
rejects, so a launch of more than `maxGroupCountX` groups is refused by
`check_launch_size()` first.

`LevelZeroBackend.__del__` intentionally does nothing: Level Zero cleanup
at interpreter shutdown can crash because the driver may already be
unloaded.

## Scalar arguments on the GPU

The CPU passes scalars by value. The four GPU backends instead run
`pack_scalars` (`lang/ir_pack_scalars.py`) on a copy of the variant's IR in
their `build` callback: every scalar parameter is replaced by a load from a
small field, one field per scalar dtype (`__pack_f32__`, `__pack_i32__`,
...), marked `_is_scalar_pack`. The pass was written for Metal, whose
kernels have a limited number of buffer bindings; packing makes the count
one per scalar type instead of one per scalar, and the other GPU backends
share the code path.

The pack fields are allocated once, when the variant is built
(`_create_pack_fields` in `runtime/kernel_utils.py`), and stored in the
variant's payload. Each dispatch writes the new values with
`_update_pack_fields`, which is a host-to-device copy on CUDA, HIP and Level
Zero and a memory copy on Metal. Changing a scalar's value therefore never
recompiles. The loop range is resolved from the *unpacked* IR, because
packing rewrites the parameter list that the range expression refers to.

Since the pack fields belong to the variant, as does Metal's argument
buffer, two threads dispatching the same variant at once would each write
their arguments into the same buffers, and a launch could run with the
other call's scalars or fields. Reproduced on CUDA with four threads, 248
of 400 dispatches computed with another thread's scalars. CUDA, HIP and
Metal therefore hold `KernelVariant.dispatch_lock` from the pack update
through the synchronous launch, so different variants still run
concurrently; uncontended, the lock cost about 0.25 µs against a 95 µs CUDA
dispatch. Level Zero takes its backend-wide `_launch_lock` instead (see
[Level Zero](#level-zero)). The CPU binds arguments per call and takes no
lock.

## Textures

`tack.texture3d(field, shape)` creates a `Texture3D` (`lang/field.py`)
holding a copy of the field; in a kernel, `tex.sample(u, v, w)`
interpolates trilinearly at normalized coordinates, where `0` and `1` are
the centers of the first and last texels. Hardware samplers put texel
centers at `(i + 0.5) / N`, so every hardware path rewrites each coordinate
to `(u * (N - 1) + 0.5) / N`.

**Storage.** The texture copies its field when it is created, on every
backend, and `tex.update()` copies it again; writes to the field in
between don't reach it. The copy is storage the texture owns, in
`tex._storage`. `Backend.texture_in_hardware(shape_3d)` chooses it: where
the answer is true, `create_texture_image()` returns a backend image
object (`MetalTextureImage`, `CUDATextureImage`, `HIPTextureImage` or
`L0TextureImage`) whose `upload(field)` does the copy; otherwise the
texture allocates a private `f32` field and `update()` copies into it with
the device copy kernel. Only one extra copy is held per texture. Images
are built eagerly and updated in place, so their texture objects stay
valid, and they free their device objects when the texture is collected
(CUDA's only while its context is alive). At dispatch, `bind_textures()`
(`runtime/kernel_utils.py`) replaces each `Texture3D` argument with its
storage, never its field. No compiled kernel caches texture objects.

| Backend | Hardware path | Software fallback when |
|---|---|---|
| CPU | none | always |
| Metal | `MTLTexture` (3D, `R32Float`), filled by a blit from the field's buffer; `sampler(normalized, linear, clamp_to_edge)` | never |
| CUDA | 3D CUDA array (one 32-bit float channel) filled with `cuMemcpy3D`; texture object with linear filtering, clamp, normalized coordinates | never |
| HIP | `hipMalloc3DArray` + `hipMemcpy3D`; texture object as on CUDA | the device reports no image support, or an extent exceeds its 3D texture limit |
| Level Zero | `zeImageCreate` (3D, 32-bit float), filled through a host staging allocation; `CLK_NORMALIZED_COORDS_TRUE \| CLK_ADDRESS_CLAMP_TO_EDGE \| CLK_FILTER_LINEAR` | `maxSamplers == 0` (Xe-HPC), or an extent exceeds `maxImageDims3D` |

The fallback decision changes the generated code, so HIP and Level Zero
ask `texture_in_hardware()` again in their own `_store_texture_shapes`,
passed to `resolve_variant`, which runs before the variant key is built.
The texture and the variant therefore agree on which storage is bound.
Falling back clears the parameter's `_is_texture` flag and the generator
emits a software trilinear helper over the texture's private field instead.
`tack.inspect` calls the same hook, so it shows the code dispatch compiles.

CDNA parts — gfx940/941/942, the MI300 family — have no texture hardware,
and hipRTC refuses `tex3D` outright there ("The image/texture API not
supported on the device"), which is why HIP asks
`hipDeviceAttributeImageSupport` rather than assuming. Xe-HPC similarly has
no samplers; filtered image reads would be driver-emulated.

**Why a snapshot on every backend.** Hardware samplers read an image, not
the field, so on those paths a texture is a copy whatever Tack does. The
only portable rule is therefore that a texture is a copy everywhere, taken
at points the program chooses: at creation and on `update()`. Owning the
image in the texture, rather than caching it in a compiled kernel under
the field's device address, means no texture is ever handed another's
image (a new field that reuses a freed field's address, or a second
texture over the same field, gets its own copy), and the image is released
with its texture. The images are single-channel 32-bit float, so
`texture3d()` rejects non-`f32` fields rather than reinterpreting them,
and it rejects shapes whose `W*H*D` differs from the field's size, which
the copies would read past. `render_volume()` calls `update()` before each
render, since the mixed-scene path tracer reads the volume's field
directly. `test_texture_snapshot.py` pins these rules.

## Device reductions

`Field.sum()`, `min()` and `max()` call `backend.reduce_field()` when
`supports_device_reductions` is true and otherwise reduce with NumPy on the
host. All four GPU backends implement `reduce_field` only for `f32` fields,
using a source from `field_reduction_source(dialect, op)`
(`codegen/reductions.py`): a 256-lane tree in each group, followed by
unordered atomic accumulation of the partial results. Any other dtype falls
back to NumPy, as does Level Zero when the device's group-size limits are
below 256. CUDA, HIP and Level Zero pass the element count as a 64-bit
argument and check it with `check_launch_size()`. Metal's kernel holds the
count in 32 bits, so Metal reduces a field of 2³² or more elements with
NumPy over its shared buffer. An empty field returns `0.0` for `sum` and raises `ValueError`
for `min` and `max` on every backend. The ordering and accuracy promises are
in the [contract](../reference/language-contract.md#field-and-parallel-reductions).

## Workarounds and why they exist

Each of these is narrow, documented at its site, and meant to be removed
when its cause is fixed.

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
`-(-32768)` stored to an i32 field read `32768`, and `abs` of the i8, i16
and i32 minimum widened as its magnitude. The SPIR-V from `ocloc` was
correct, `-ze-opt-disable` hid the error, and no inlined form tried —
`as_short`, a select on the minimum, complement-plus-one, a sign mask —
survived the driver's folding; LLVM canonicalizes them all to the same
shape. `OpenCLCodeGen` sets `_opaque_negation`, and `IntegerCodeGen`
emits `neg` below 32 bits and `abs` below 64 bits as
`__attribute__((noinline))` helpers. i32 negation, i64 `abs` and every
other operation stay inline. `test_wrapped_negation_and_abs_survive_widening`
covers the widened results on each backend.

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

## See also

- [Parallel Execution](parallel-execution.md) — the execution model these
  launches implement.
- [Memory and Aliasing](memory-and-aliasing.md) — the buffers and the
  aliasing rules each generator honors.
- [Interoperability](interoperability.md) — sharing memory with other
  libraries.
- [Conformance and Validation](../contracts/conformance.md) — how each
  backend is validated on hardware.

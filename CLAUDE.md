# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

Tack is split into 3 packages under `packages/`:

| Package | Path | Contents |
|---------|------|----------|
| `tack-core` | `packages/tack-core/` | Kernels, fields, types, IR, codegen, backends, scan/copy |
| `tack-rendering` | `packages/tack-rendering/` | Path tracer (BVH, camera, scene, ColorTable) |
| `tack-vis` | `packages/tack-vis/` | Visualization algorithms (flying edges, normals, VTK interop) |

All share the `tack` namespace via `pkgutil.extend_path`.

**One catch worth knowing.** `extend_path` merges the *contents* of same-named directories, but only one `__init__.py` runs — the first found on the path. Both tack-core and tack-vis have a `tack/algorithms/` directory, so tack-core's `__init__.py` wins and is where the vis worklets are re-exported (guarded, so tack-core still works alone). Adding a second `algorithms/__init__.py` makes it dead code; `packages/tack-vis/tests/test_namespace.py` guards against that.

## Build & Test Commands

```bash
uv sync                                                     # install all packages (editable)
uv run pytest                                                # run all tests
uv run pytest packages/tack-core/tests/test_cpu_jit.py        # run one test file
uv run pytest packages/tack-core/tests/test_hip.py            # run HIP backend tests
uv run pytest -k "test_saxpy"                                # run tests matching a pattern
uv run python packages/tack-core/examples/validate_all.py     # validation suite
uv run python packages/tack-core/examples/01_hello_tack.py --arch hip  # example on backend
```

All examples accept `--arch cpu|metal|cuda|hip|level_zero` to select the backend.

No build step — pure Python with JIT compilation at runtime.

## Architecture

Tack is a Python-first GPU compute framework inspired by Taichi. Kernels are decorated Python functions that compile at call time through this pipeline:

```
@tack.kernel Python function
    → AST transform (ast_transform.py)
    → Tack IR (ir.py)
    → IR passes: resolve (ir_resolve.py) → type inference → optimize (ir_optimize.py)
    → Backend-specific codegen + dispatch
```

### Codegen backends from Tack IR

| Backend | Codegen | Runtime compilation | Dispatch |
|---------|---------|-------------------|----------|
| CPU | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes function call |
| Metal | `msl_gen.py` → MSL source | Metal API (pyobjc) | compute pipeline |
| CUDA | `cuda_gen.py` → CUDA C source | NVRTC → PTX | cuLaunchKernel |
| HIP | `hip_gen.py` → HIP C source (extends CUDA) | hipRTC → code object | hipLaunchKernel |
| Level Zero | `opencl_gen.py` → OpenCL C source (extends CUDA) | libocloc → SPIR-V | zeCommandListAppendLaunchKernel |

### Key abstraction: Field with DeviceBuffer

`Field` (in `lang/field.py`) holds a `DeviceBuffer` — the backend-specific storage:
- **CPU**: `NumpyBuffer` wraps a numpy array. CPU backend accesses via ctypes pointers.
- **Metal**: `MetalBuffer` — Metal shared buffer (zero-copy unified memory on Apple Silicon). The numpy array view points directly into Metal buffer memory.
- **CUDA**: `CUDABuffer` holds a device pointer (`cuMemAlloc`). Explicit host↔device copies on `from_numpy`/`to_numpy`.
- **HIP**: `HIPBuffer` holds a device pointer (`hipMalloc`). Explicit host↔device copies.
- **Level Zero**: `L0Buffer` holds a device pointer (`zeMemAllocDevice`). Explicit host↔device copies via immediate command list.

`tack.field()` calls `backend.allocate_field()` to create the appropriate buffer type.

### Backend contract

All five backends subclass `Backend` (`runtime/backend.py`), which declares the required methods (`allocate_field`, `wrap_ptr`, `execute`) and the capability attributes callers read instead of probing with `hasattr`: `name`, `display_name`/`label`, `supported_dtypes`, `supports_f64`, `supports_device_reductions`, `device_memory_spaces`.

Anything derivable is derived — `supports_f64` comes from `supported_dtypes`, so the two cannot disagree. Level Zero sets `supported_dtypes` in `__init__` because f64 depends on the device.

### Kernel execution flow

All five backends share one entry point: `resolve_variant()` in `runtime/kernel_utils.py`. It turns a call into a compiled variant, running the IR pass pipeline **only when that variant is new**.

Every dispatch:
1. `Kernel.__call__` → `backend.execute(kernel, args)` → `resolve_variant(...)`
2. Detect template arguments (`@tack.data_oriented` classes) and expand them
3. Detect vector fields and set up scalarization metadata
4. `kernel.get_ir(...)` returns the **pristine IR template** for this specialization
5. Type inference (`infer_param_types`) — annotates params from actual args, sets `_is_field`
6. Build the variant key and look it up (see below)
7. Resolve the loop range from the variant's IR; dispatch (CPU decides serial vs threads, GPU launches a grid)

Only on a cache miss:
1. Deep-copy the template — the passes below mutate IR in place and must not touch the template
2. Dimension size resolution (`ir_resolve.py`)
3. Dispatch-time type checking (`check_dispatch_types`) — validates field dtypes against the backend
4. IR optimization: LICM, copy propagation, CSE (`ir_optimize.py`)
5. Backend `build` callback: scalar packing (GPU), type annotation (`ir_type_annotate.py`), codegen, compile

### Variant cache key

Keyed per `Kernel` (weakly, so compiled code is released with the kernel), then by:
argument type signature + texture extents + template constants + **`shape_signature`**.

That last one matters for correctness, not speed. `ir_resolve` substitutes dimension sizes as literals — `a[i, j]` linearizes to `i * dim1 + j` with `dim1` baked in — so the row stride is part of the compiled code's identity. `shape_signature()` reports exactly the dimensions a kernel bakes in (memoized per IR; empty for 1-D kernels, so varying a flat length does **not** re-specialize).

Because the passes mutate IR in place, the template from `get_ir()` must be treated as immutable — a pass that consumed its `IRDimSize` nodes would leave nothing for a later shape to resolve.

### Field dimensions

`field.shape[k]` and `len(field)` both lower to `IRDimSize` in `ast_transform.py`, which `ir_resolve.py` folds to a literal wherever it appears — loop bounds, conditions, arithmetic, indices. The dimension index must be a literal (`x.shape[d]` with a runtime `d` raises).

**One exception, and it matters:** the outermost parallel loop's bound is left unresolved. It never reaches generated code — codegen reads the `__loop_end__` parameter — so `_resolve_range_expr` evaluates it per dispatch instead, and `ir_shape_deps` excludes it from the variant key. Without that, `for i in range(x.shape[0])` — the most common line in any kernel — would compile a new variant for every array length.

Everywhere else the dimension *is* compiled in, so it specializes:

```python
for i in range(x.shape[0]):        # one variant for all lengths
    out[i] = x[x.shape[0] - 1 - i] # ...but this bakes the length in → one variant per length
```

To avoid that, pass the length as a scalar argument (`def reverse(x, out, n)`) — scalars are runtime parameters and don't specialize.

### Kernel code inspection

`tack.inspect(kernel, *args, mode=...)` runs the compilation pipeline and returns the generated code as a string without executing. Modes: `"ir"` (Tack IR), `"source"` (backend code: LLVM IR / MSL / CUDA C / HIP C / OpenCL C), `"optimized"` (post-LLVM-O3 IR, CPU only). Implementation in `lang/inspect_kernel.py`.

### IR structure (lang/ir.py)

The IR is a simple tree of nodes:

**Loops**: `IRParallelFor` (outermost, maps to thread parallelism), `IRSequentialFor` (inner loops, supports `step`), `IRWhile`, `IRBreak`, `IRContinue`

**Control flow**: `IRIf`, `IRIfExp` (ternary)

**Field access**: `IRFieldLoad`, `IRFieldStore`, `IRAtomicOp` (atomic_add/min/max)

**GPU primitives**: `IRSharedAlloc` (threadgroup memory), `IRBarrier` (sync), `IRThreadId` (local thread index)

**Expressions**: `IRBinOp`, `IRUnaryOp`, `IRCompare`, `IRBoolOp`, `IRCall` (math builtins), `IRCast`, `IRConstant`, `IRName`, `IRAttribute`, `IRDimSize`

**Other**: `IRAssign`, `IRReturn`, `IRPrint` (kernel debugging)

### IR passes

- **ir_resolve.py**: Replaces `IRDimSize` nodes with concrete constants from field shapes, resolves `IRAtomicOp` sub-expressions, and resolves `shared_like` dtypes from fields
- **ir_optimize.py**: Three passes — Loop-Invariant Code Motion (LICM), copy propagation, Common Subexpression Elimination (CSE)
- **type_inference.py**: Annotates IR params with types from actual arguments. Fields get `_is_field=True`, scalars get `_is_field=False`. Float scalars auto-promote to `f64` when any field arg uses `f64`; otherwise default to `f32`. Int scalars exceeding i32 range auto-promote to `i64`. `check_dispatch_types()` validates field dtypes against backend capabilities.
- **ir_type_annotate.py**: Sets `dtype` (a `ScalarType`) on every expression IR node. Codegens read `node.dtype` directly instead of reimplementing type inference heuristics.

### Scalar kernel arguments

Kernels accept both fields and Python scalars (int, float) directly. The `_is_field` attribute on IR params controls codegen: fields become pointers, scalars become values. All four backends handle this distinction in their codegen and dispatch paths.

### Vector scalarization

`tack.Vector.field(n, dtype, shape)` creates a flat scalar field of size `prod(shape) * n`. In kernels, `field[i]` expands to n component loads/stores. Vector operations (add, dot, cross, normalize) are scalarized at the IR level.

### @tack.func inlining

Functions decorated with `@tack.func` are inlined at the AST level into kernels. Supports return values, multi-return (tuple), nested inlining, and vector propagation. Variables are renamed with unique suffixes to avoid collisions.

### @tack.data_oriented templates

Classes decorated with `@tack.data_oriented` can be passed as template arguments. Class-level scalar attributes become compile-time constants (part of cache key), instance scalar attributes become runtime kernel parameters (no recompilation on change), field attributes become kernel buffer parameters, and `@tack.func` methods are inlined with `self` resolved. Methods can call sibling methods on `self`.

### 64-bit loop indices on GPU

GPU backends use 64-bit integers for loop variables and index arithmetic (`long` on Metal, `long long` on CUDA/HIP) to support grids with more than 2^31 elements. The CPU backend already used i64 via LLVM. Metal's `thread_position_in_grid` attribute is limited to `uint`, so max single dispatch is 2^32 threads. The `int()` cast in kernel code remains 32-bit (user semantics).

### Algorithms (tack.algorithms)

`exclusive_scan` and `inclusive_scan` implement Blelloch-style parallel prefix sums. They use a `_read_last` kernel to return the total sum without copying the entire buffer to numpy. The Blelloch scan uses O(log n) kernel launches, so for small arrays (< ~1M elements) a numpy CPU roundtrip may be faster due to kernel launch overhead.

### ColorTable (tack.rendering)

`ColorTable` maps per-vertex scalar fields to RGB colors via a sampled lookup table. Presets: `viridis`, `cool_to_warm`, `inferno`, `plasma`, `grayscale`, `rainbow`. The `Actor` class accepts `scalars` (tack.field or numpy) + `color_table` (ColorTable) to enable scalar field coloring. During `Scene._prepare()`, scalars are mapped to per-vertex colors on GPU via linear interpolation into the lookup table. The pathtrace kernel's existing per-vertex color interpolation handles the rest — no kernel changes needed.

### Multiple lights (tack.rendering)

Multiple `PointLight` instances can be added to a `Scene`. Each light contributes independently with its own shadow ray. Light data is packed into a flat field (7 floats per light: x, y, z, intensity, r, g, b) and passed to the pathtrace kernel, which loops over all lights in the direct-illumination section. Light color (previously ignored) is now applied to each light's contribution. The `render()` function's `light_position` kwarg still works as a single-light override for backward compatibility.

### Volume rendering (tack.rendering)

`Volume` represents a uniform-grid scalar field for ray-casting volume rendering. `TransferFunction` maps scalars to RGBA (reuses ColorTable preset colors + user-defined opacity function). `render_volume(canvas, volume, camera)` ray marches through the volume with front-to-back compositing, trilinear interpolation via `tack.texture3d`, and opacity-corrected compositing.

### Unified render dispatcher (tack.rendering)

`render()` in `render.py` is the unified entry point. When a scene has both surfaces and volumes, the path tracer integrates volume ray marching directly — for each primary ray, after BVH traversal finds the closest surface hit, the volume is marched from the ray origin to the hit distance with front-to-back compositing. Volume opacity attenuates the throughput so surfaces are correctly visible through semi-transparent volumes. Volume-only scenes use the standalone ray caster. `render_volume()` remains available for direct single-volume rendering without a Scene.

### Annotations (tack.rendering)

`annotate(image, annotations)` draws overlays on `(H, W, 4) uint8` numpy arrays (from `Canvas.to_numpy()`). Three annotation types: `ColorBar` (gradient strip + tick labels from ColorTable/TransferFunction), `AxisIndicator` (projected XYZ axes from camera orientation), `TextOverlay` (arbitrary text). Lines and rectangles use pure numpy; text requires Pillow (soft dependency, gracefully skipped if absent). Camera stores `_right`, `_up`, `_forward` basis vectors for the axis indicator.

### Orthographic camera (tack.rendering)

`OrthographicCamera` generates parallel rays — all rays have the same direction, but origins vary per pixel. Uses the same scalar attribute pattern as `PerspectiveCamera` with added `odx_x/y/z` and `ody_x/y/z` origin-per-pixel deltas. `PerspectiveCamera` sets these to zero for backward compatibility. Works with both path tracer and volume renderer. The `view_height` parameter controls the world-space height of the view rectangle.

### Material system (tack.rendering)

`Material` class with three types: `MATTE` (Lambertian diffuse, default), `SPECULAR` (perfect mirror reflection), `TRANSPARENT` (glass with Snell's law refraction + Schlick Fresnel). Set per-actor via `Actor(..., material=Material(Material.SPECULAR))`. Per-triangle material IDs (`mat_ids` i32 field) index into a flat material table (`mat_table` f32 field, stride 4: type, ior, reserved, reserved). Direct lighting is only computed for matte surfaces. Specular bounces reflect perfectly; transparent surfaces refract or reflect based on Fresnel probability using Halton random numbers.

### Wireframe and point rendering (tack.rendering)

`Actor` accepts `render_mode="solid"` (default, path traced), `"wireframe"` (triangle edges), or `"points"` (vertex discs). Wireframe and point actors are dispatched to GPU rasterization kernels in `rasterize.py` instead of the path tracer. Uses an MVP projection matrix passed as a flat f32 field, Bresenham line rasterization for wireframe, disc rasterization for points, and `tack.atomic_min` depth testing. Supports perspective and orthographic cameras, per-actor colors, scalar coloring, and configurable point size.

### CPU threading decision

The CPU backend fans a loop range out to its `ThreadPoolExecutor` only when the serial run would cost meaningfully more than the fan-out. Both sides are measured, not assumed:

- Each `CompiledKernel` carries `ns_per_elem`, a smoothed estimate of its serial cost, updated on every serial dispatch. From it the backend precomputes `parallel_min_elems`, so the dispatch hot path is one integer compare.
- The backend measures its own fan-out cost once (`_fan_out_ns`), by dispatching *empty* ranges through the real path — no loop iterations, so the probe has no side effects.
- The first time a kernel is seen at a range large enough to matter, a small prefix is timed serially and the rest is decided on that sample.

A fixed element count cannot work here: the crossover moves ~1000× with arithmetic intensity (~4M elements for `out[i] = x[i]*2+1`, ~130K for a `sqrt`/`sin` expression, ~4K for a 20-iteration inner loop). The previous constant of 1024 sat below all of them, making mid-size dispatches of cheap kernels 3–10× slower than running them serially.

`TACK_CPU_THREADS` overrides the thread count; `1` keeps everything on the calling thread.

## Kernel language features

- **Loops**: `for i in range(n)`, `for i in range(start, end)`, `for i in range(start, end, step)`, `for i, j in tack.ndrange(w, h)`, `while`, `break`, `continue`
- **Math**: `sqrt`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `exp`, `exp2`, `log`, `log2`, `log10`, `floor`, `ceil`, `abs`, `min`, `max`, `pow`
- **Types**: `int()`, `float()` casts, plus explicit `tack.i8()`, `tack.u8()`, `tack.i16()`, `tack.u16()`, `tack.i32()`, `tack.u32()`, `tack.i64()`, `tack.u64()`, `tack.f32()`, `tack.f64()`
- **Atomics**: `tack.atomic_add(field, idx, val)`, `tack.atomic_min(...)`, `tack.atomic_max(...)`
- **GPU primitives**: `tack.shared(dtype, size)`, `tack.shared_like(field, size)`, `tack.barrier()`, `tack.thread_id()`
- **Debug**: `print("label:", value)` — emits printf on CPU/CUDA/HIP, no-op on Metal
- **Fields**: `field[i]`, `field[i, j]`, `field[None]`, `field.shape[k]`, `len(field)` — usable anywhere in a kernel (loop bounds, conditions, arithmetic, indices), not just as the outer loop bound. The dimension index must be a literal. See "Field dimensions" below for what specializes.
- **Reductions**: `field.sum()`, `field.min()`, `field.max()` — GPU-native on Metal, numpy on CPU

## Platform-specific dependencies

- **macOS (Metal)**: `pyobjc-framework-Metal`
- **Linux/Windows (CUDA)**: `cuda-python>=13.2`, NVIDIA driver + CUDA toolkit
- **Linux (HIP/ROCm)**: `hip-python`, ROCm toolkit
- **Linux (Level Zero/Intel)**: `libze_loader.so`, `libocloc.so` (Intel compute runtime)
- **CPU-only**: `llvmlite`, `numpy`

## HIP backend notes

The HIP codegen (`hip_gen.py`) extends `CUDACodeGen` — HIP device code uses the same syntax as CUDA (`blockIdx`, `threadIdx`, `__global__`, `__shared__`, `__syncthreads`). The only difference is `#include <hip/hip_runtime.h>`. The runtime (`hip_backend.py`) uses `hip-python` bindings for hipRTC compilation and dispatch.

To install hip-python (packages are on **Test PyPI**, not regular PyPI):

```bash
uv pip install --prerelease=allow --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
  "hip-python~=7.1.0"
```

Then: `tack.init(arch=tack.hip)`.

**Known issue**: `hiprtcDestroyProgram` segfaults in hip-python 7.1 bindings. The backend skips this call (minor leak, mitigated by kernel caching).

## Level Zero backend notes

The Level Zero codegen (`opencl_gen.py`) extends `CUDACodeGen` — OpenCL C kernel syntax mirrors CUDA with different qualifiers (`__kernel`/`__global`, `get_global_id(0)`/`blockIdx*blockDim+threadIdx`, `__local`/`__shared__`, `barrier()`/`__syncthreads()`). Math functions are overloaded (no `f` suffix). The runtime (`level_zero_backend.py`) uses ctypes bindings to `libze_loader.so` and `libocloc.so`.

Compilation pipeline: OpenCL C source → `libocloc.so` (in-process, via `oclocInvoke`) → SPIR-V → `zeModuleCreate` → `zeKernelCreate`. The ocloc library is part of the Intel compute runtime (`intel-opencl-icd` package).

Requires: `libze_loader.so` (Level Zero runtime), `libocloc.so` (Intel offline compiler). No Python packages needed.

Then: `tack.init(arch=tack.level_zero)`.

## Do not mention Claude in git commits

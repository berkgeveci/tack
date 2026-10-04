# Backends

!!! info "This page is the source of truth"

    Tack has five backends and the details differ in ways that matter. This
    table is the canonical one — the README and the guides link here rather
    than restating it, because a capability table copied into three places
    drifts, and this one has.

## The five

| Backend | Codegen | Runtime compiler | Dispatch | Buffer |
|---|---|---|---|---|
| **CPU** | `llvm_gen.py` → LLVM IR | llvmlite JIT | ctypes call, thread pool | `NumpyBuffer` |
| **Metal** | `msl_gen.py` → MSL | Metal API (pyobjc) | compute pipeline | `MetalBuffer` |
| **CUDA** | `cuda_gen.py` → CUDA C | NVRTC → PTX | `cuLaunchKernel` | `CUDABuffer` |
| **HIP** | `hip_gen.py` → HIP C | hipRTC | `hipLaunchKernel` | `HIPBuffer` |
| **Level Zero** | `opencl_gen.py` → OpenCL C | `libocloc` → SPIR-V | `zeCommandListAppendLaunchKernel` | `L0Buffer` |

`hip_gen.py` is nine statements: it subclasses the CUDA generator and swaps
the `#include`, because HIP device code uses CUDA's syntax. `opencl_gen.py`
also extends the CUDA generator, but overrides considerably more — OpenCL C
spells address spaces, thread indices, barriers and math differently, and the
wrong spelling still compiles as C.

## Scalar types

| | `i8`/`u8` | `i16`/`u16` | `i32`/`u32` | `i64`/`u64` | `f32` | `f64` |
|---|---|---|---|---|---|---|
| CPU | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Metal | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| CUDA | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| HIP | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Level Zero | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ |

**Metal has no `f64`** — Apple GPUs do not implement double precision, so this
is a hardware fact rather than a gap in Tack.

**Level Zero decides at runtime.** Some Intel parts implement `f64` and some do
not, so the backend queries `zeDeviceGetModuleProperties` during `init()` and
builds its supported set from the answer. Ask the backend rather than assuming:

```python
from tack.runtime.dispatch import get_backend
tack.init(arch=tack.level_zero)
print(get_backend().supports_f64)
```

`supports_f64` is derived from `supported_dtypes` rather than stored
separately, so the two cannot disagree.

## Workgroup execution

`supports_workgroups` is false on CPU and true on Metal, CUDA, HIP and
Level Zero. Shared memory, barriers, local thread indices and block
reductions require this capability. CPU rejects such kernels before
compilation or execution; use `local_array` / `local_array_like` for
private scratch storage instead.

This flag describes the execution model, not barrier uniformity, partial
workgroup safety or supported atomic types/scopes. See the
[workgroup contract](language-contract.md#workgroups-and-synchronization).

## Installing

```bash
pip install 'tack-core[cpu]'          # llvmlite
pip install 'tack-core[metal]'        # pyobjc-framework-Metal
pip install 'tack-core[cuda]'         # cuda-python; needs the NVIDIA driver
pip install 'tack-core[hip]'          # hip-python; needs ROCm
pip install 'tack-core[level_zero]'   # installs nothing — see below
```

Two of these need a word.

**HIP.** `hip-python`'s version tracks the ROCm release it binds to — 7.1.x
against ROCm 7.1, 7.2.x against 7.2 — so the extra sets a lower bound rather
than a pin. A system on a different ROCm wants it spelled out:

```bash
pip install 'hip-python~=7.2.0'
```

It ships manylinux x86_64 wheels only, so the dependency carries a platform
marker and is skipped elsewhere.

**Level Zero.** The extra is deliberately empty. Level Zero is reached through
ctypes to `libze_loader.so` (the Level Zero runtime) and `libocloc.so` (the
Intel offline compiler, part of `intel-opencl-icd`) — system libraries, not
Python packages. The extra exists so the name stays valid.

## Asking a backend what it can do

Backends declare their capabilities rather than being probed with `hasattr`:

```python
from tack.runtime.dispatch import get_backend

be = get_backend()
be.name                       # 'cpu', 'metal', 'cuda', 'hip', 'level_zero'
be.label                      # how to spell it in a message
be.supported_dtypes           # the set dispatch checks field arguments against
be.supports_f64               # derived from the above
be.supports_device_reductions # whether .sum()/.min()/.max() run on device
be.device_memory_spaces       # what memory_space() must return for a pointer
```

See [`Backend`](api.md#tack.runtime.backend.Backend) for the full contract.

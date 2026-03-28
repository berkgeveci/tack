# Backends

Tack compiles the same kernel source to different GPU APIs. Each backend has
its own compilation pipeline and memory model.

## Overview

| Backend | Platform | GPU | Compilation | Dependencies |
|---------|----------|-----|-------------|-------------|
| CPU | All | None (uses CPU cores) | LLVM JIT via llvmlite | `llvmlite` |
| Metal | macOS | Apple Silicon | MSL source → Metal API | `pyobjc-framework-Metal` |
| CUDA | Linux/Windows | NVIDIA | CUDA C → NVRTC → PTX | `cuda-python>=13.2` |
| HIP | Linux | AMD (ROCm) | HIP C → hipRTC | `hip-python` |
| Level Zero | Linux | Intel | OpenCL C → libocloc → SPIR-V | Level Zero runtime |

## CPU

The CPU backend uses LLVM JIT (via llvmlite) to compile kernels to native
machine code. Work is split across physical CPU cores using a persistent
thread pool. Below 1024 elements, single-threaded execution is used to
avoid thread dispatch overhead.

```bash
pip install 'tack[cpu]'
```

```python
tack.init(arch=tack.cpu)
```

## Metal

Apple Silicon unified memory means fields are zero-copy — the numpy view
and GPU buffer share the same physical memory. No host-device transfers.

```bash
pip install pyobjc-framework-Metal
```

```python
tack.init(arch=tack.metal)
```

Metal has a 31-buffer binding limit per kernel. Tack automatically packs
scalar parameters into constant buffers to stay within this limit.

Hardware 3D texture sampling is supported via `texture3d<float>.sample()`.

## CUDA

NVIDIA GPUs via the CUDA driver API. Fields use device memory (`cuMemAlloc`)
with explicit host-device copies.

```bash
pip install 'cuda-python>=13.2'
```

```python
tack.init(arch=tack.cuda)
```

## HIP

AMD GPUs via ROCm. The codegen extends CUDA — HIP device code uses the same
syntax (`blockIdx`, `threadIdx`, `__global__`).

```bash
uv pip install --prerelease=allow --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ --index-strategy unsafe-best-match \
  "hip-python~=7.1.0"
```

```python
tack.init(arch=tack.hip)
```

## Level Zero

Intel GPUs via the Level Zero API. OpenCL C source is compiled to SPIR-V
in-process using `libocloc`, then loaded via `zeModuleCreate`.

Hardware 3D texture sampling (`image3d_t` with `read_imagef`) is used on
devices with texture units (Xe-HPG/Xe-LPG). Xe-HPC (Ponte Vecchio) falls
back to software trilinear since it has no sampler hardware.

```python
tack.init(arch=tack.level_zero)
```

## Error Messages

If a backend is unavailable, `tack.init()` gives a clear error:

```
RuntimeError: Cannot initialize 'hip' backend: missing dependency.
  No module named 'hip'
  Requires AMD GPU with ROCm and hip-python
```

Kernel compilation errors show the kernel name, backend, and the relevant
error lines without dumping the full generated source. Set `TACK_DUMP_MSL=1`
environment variables to inspect generated code.

## Type Checking

Tack validates field dtypes at dispatch time before compilation. If a field
uses a dtype not supported by the target backend, you get a clear error:

```
TypeError: Kernel 'my_kernel': parameter 'data' has dtype tack.f64,
which is not supported on Metal.
Supported dtypes: f32, i32, i64, u32, u64
```

Supported dtypes per backend:

| Backend | Supported dtypes |
|---------|-----------------|
| CPU | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| Metal | i8, u8, i16, u16, i32, u32, i64, u64, f32 (no f64) |
| CUDA | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| HIP | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |
| Level Zero | i8, u8, i16, u16, i32, u32, i64, u64, f32, f64 |

# Backends

PGC compiles the same kernel source to different GPU APIs. Each backend has
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
pip install 'pgc[cpu]'
```

```python
pgc.init(arch=pgc.cpu)
```

## Metal

Apple Silicon unified memory means fields are zero-copy — the numpy view
and GPU buffer share the same physical memory. No host-device transfers.

```bash
pip install pyobjc-framework-Metal
```

```python
pgc.init(arch=pgc.metal)
```

Metal has a 31-buffer binding limit per kernel. PGC automatically packs
scalar parameters into constant buffers to stay within this limit.

Hardware 3D texture sampling is supported via `texture3d<float>.sample()`.

## CUDA

NVIDIA GPUs via the CUDA driver API. Fields use device memory (`cuMemAlloc`)
with explicit host-device copies.

```bash
pip install 'cuda-python>=13.2'
```

```python
pgc.init(arch=pgc.cuda)
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
pgc.init(arch=pgc.hip)
```

## Level Zero

Intel GPUs via the Level Zero API. OpenCL C source is compiled to SPIR-V
in-process using `libocloc`, then loaded via `zeModuleCreate`.

Hardware 3D texture sampling (`image3d_t` with `read_imagef`) is used on
devices with texture units (Xe-HPG/Xe-LPG). Xe-HPC (Ponte Vecchio) falls
back to software trilinear since it has no sampler hardware.

```python
pgc.init(arch=pgc.level_zero)
```

## Error Messages

If a backend is unavailable, `pgc.init()` gives a clear error:

```
RuntimeError: Cannot initialize 'hip' backend: missing dependency.
  No module named 'hip'
  Requires AMD GPU with ROCm and hip-python
```

Kernel compilation errors show the kernel name, backend, and the relevant
error lines without dumping the full generated source. Set `PGC_DUMP_MSL=1`
environment variables to inspect generated code.

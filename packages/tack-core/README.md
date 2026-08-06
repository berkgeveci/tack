# tack-core

Kernels, fields, types, IR, codegen and backends for [Tack](https://github.com/Kitware/tack) —
a Python-first GPU compute framework.

Write compute kernels as decorated Python functions; they are JIT-compiled at first call and
dispatched to whichever backend is active. The same kernel source runs on five backends.

```python
import tack, numpy as np
tack.init(arch=tack.cpu)          # or metal, cuda, hip, level_zero

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):   # outermost loop → thread parallelism
        out[i] = x[i] + y[i]

x = tack.field(dtype=tack.f32, shape=(1024,))
y = tack.field(dtype=tack.f32, shape=(1024,))
out = tack.field(dtype=tack.f32, shape=(1024,))
x.from_numpy(np.arange(1024, dtype=np.float32))
y.from_numpy(np.ones(1024, dtype=np.float32))

vector_add(x, y, out)
print(out.to_numpy())
```

## Install

Pick the extra matching your hardware:

```bash
pip install tack-core[cpu]          # LLVM JIT via llvmlite
pip install tack-core[metal]        # Apple Silicon
pip install tack-core[cuda]         # NVIDIA
pip install tack-core[hip]          # AMD/ROCm — see the repo for hip-python
pip install tack-core[level_zero]   # Intel — needs the system Level Zero runtime
```

## Backends

| Backend | Codegen | Runtime compilation |
|---------|---------|---------------------|
| CPU | LLVM IR | llvmlite JIT |
| Metal | MSL | Metal API |
| CUDA | CUDA C | NVRTC → PTX |
| HIP | HIP C | hipRTC |
| Level Zero | OpenCL C | libocloc → SPIR-V |

## Related packages

- `tack-rendering` — path tracer, volume rendering, rasterization
- `tack-vis` — flying edges, normals, cell↔point, VTK interop

Full documentation, examples and the kernel-language reference are in the
[repository](https://github.com/Kitware/tack).

BSD 3-Clause licensed.

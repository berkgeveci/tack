# tack-core

Kernels, arrays, types, IR, codegen and backends for [Tack](https://github.com/berkgeveci/tack) —
a Python-first GPU compute framework.

Write compute kernels as decorated Python functions; they are JIT-compiled at first call (and
kept in a disk cache, so a later process loads them instead) and dispatched to whichever backend
is active. The same kernel source runs on five backends, within
each backend's capabilities (for example, Metal has no `f64` and the CPU has no workgroup
primitives).

```python
import tack, numpy as np
tack.set_default_device("cpu")          # or metal, cuda, hip, level_zero

@tack.kernel
def vector_add(x, y, out):
    for i in range(x.shape[0]):   # outermost loop → thread parallelism
        out[i] = x[i] + y[i]

x = tack.empty((1024,))       # f32 unless a dtype is given
y = tack.empty((1024,))
out = tack.empty((1024,))
tack.copyto(x, np.arange(1024, dtype=np.float32))
tack.copyto(y, np.ones(1024, dtype=np.float32))

vector_add(x, y, out)
print(out.numpy())
```

## Install

Tack is not on PyPI yet. From a clone of
[the repository](https://github.com/berkgeveci/tack), pick the extra matching
your hardware:

```bash
pip install -e "packages/tack-core[cpu]"          # LLVM JIT via llvmlite
pip install -e "packages/tack-core[metal]"        # Apple Silicon
pip install -e "packages/tack-core[cuda]"         # NVIDIA
pip install -e "packages/tack-core[hip]"          # AMD/ROCm — see the repo for hip-python
pip install -e "packages/tack-core[level_zero]"   # Intel — needs the system Level Zero runtime
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

- `tack-vis` — `tack.dataset`: meshes, fields and visualization algorithms; VTK and MFEM interop

Full documentation, examples and the kernel-language reference are in the
[repository](https://github.com/berkgeveci/tack).

BSD 3-Clause licensed.

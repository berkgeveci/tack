# tack-core

Kernels, arrays, types, IR, codegen and backends for [Tack](https://github.com/Kitware/tack) —
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

Pick the extras for your hardware. Keep `cpu` beside a GPU's extra: without it
the CPU, the default device, cannot run kernels.

```bash
pip install "tack-core[cpu]"           # LLVM JIT via llvmlite
pip install "tack-core[cpu,metal]"     # Apple silicon
pip install "tack-core[cpu,cuda]"      # NVIDIA
pip install "tack-core[cpu,hip]"       # AMD, with ROCm (hip-python: Linux x86-64)
pip install "tack-core[cpu]"           # Intel: Level Zero is the system's runtime, no extra
```

Python 3.11 or newer. Tack is pure Python and compiles kernels when they are
first called.

## Backends

| Backend | Codegen | Runtime compilation |
|---------|---------|---------------------|
| CPU | LLVM IR | llvmlite JIT |
| Metal | MSL | Metal API |
| CUDA | CUDA C | NVRTC → PTX |
| HIP | HIP C | hipRTC |
| Level Zero | OpenCL C | libocloc → SPIR-V |

## Related packages

- `tack-vis`: `tack.dataset`, with meshes, fields and visualization algorithms,
  and the VTK and MFEM interop. It is not on PyPI; install it from the
  repository, at the tag of the `tack-core` release you have, for example:

  ```bash
  pip install "tack-vis @ git+https://github.com/Kitware/tack@v0.7.0#subdirectory=packages/tack-vis"
  ```

Full documentation, examples and the kernel-language reference are in the
[repository](https://github.com/Kitware/tack).

BSD 3-Clause licensed.

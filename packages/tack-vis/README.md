# tack-vis

Scientific visualization algorithms for [Tack](https://github.com/berkgeveci/tack).

Every algorithm is written as `@tack.kernel` functions, so it runs on every backend
`tack-core` supports — CPU, Metal, CUDA, HIP and Level Zero — from one source.

```python
import tack
from tack.algorithms.flying_edges import flying_edges, UniformGrid

tack.init(arch=tack.cpu)

grid = UniformGrid(nx, ny, nz, x0, y0, z0, dx, dy, dz)
result = flying_edges(scalar_field, grid, isovalue=0.0)

result["points"]  # (n, 3) float32
result["conn"]    # (m, 3) int32
```

## What's here

- **Flying edges** — merged-point isosurface extraction, single- and multi-block
- **Compute normals** — area-weighted vertex normals accumulated with atomics
- **Cell↔point** — cell-centred to node-centred averaging
- **AMR blanking** — ghost and refinement masks following VTK's conventions
- **Finite elements** — Lagrange bases, geometry maps, DOF accessors
- **VTK interop** — zero-copy wrapping of `vtkDataArray`, host and device

## Install

```bash
pip install tack-vis
```

Requires `tack-core` with a backend extra — see the
[tack-core README](https://pypi.org/project/tack-core/). The VTK interop module
additionally needs `vtk` at runtime; nothing else does.

BSD 3-Clause licensed.

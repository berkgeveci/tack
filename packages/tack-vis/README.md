# tack-vis

Datasets and scientific visualization for [Tack](https://github.com/berkgeveci/tack):
`tack.dataset`.

`tack.dataset` holds meshes and the fields on them -- structured grids,
unstructured meshes of the linear cells, polyhedra, polygons and polylines --
and the algorithms that work on them: contour, slice, clip, threshold,
external faces, gradients, projections between spaces, streamlines and more.
Every algorithm is a Tack kernel, so it runs on every backend `tack-core`
supports -- CPU, Metal, CUDA, HIP and Level Zero -- from one source, within
each backend's capabilities. Your own kernels see a mesh the same way, through
`for_each`.

```python
import tack
import tack.dataset as td

tack.init(arch=tack.cpu)
data = td.wavelet(extent=10)                    # VTK's wavelet: 21^3 points
surface = td.contour(data, "RTData", 157.0)     # an isosurface, as triangles
print(surface.num_cells, "triangles")
```

## What's here

- **Data model** -- topologies, function spaces (H1, L2, constant, per-entity
  values, side traces), fields as a space plus values, named sets
- **Algorithms** -- contour, slice, clip, threshold, extraction, gradients and
  flow quantities, projections, probing, streamlines, transforms, statistics
- **Interop** -- VTK datasets and arrays (zero-copy through DLPack where VTK
  supports it), MFEM meshes and grid functions

## Install

From a checkout of the repository:

```bash
uv sync --extra cpu
```

`tack-vis` requires `tack-core` with a backend extra. The VTK and MFEM interop
modules additionally need `vtk` and `mfem` at runtime (`--extra interop`);
nothing else does.

BSD 3-Clause licensed.

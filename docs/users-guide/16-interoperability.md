# Sharing arrays with other libraries

There are two choices when another library produces your data: copy its values
into a Tack field, or share compatible storage. A copy gives Tack an independent
allocation. A shared view needs agreement on the device, layout, lifetime and
permission to write.

## NumPy: an independent field

```python
import numpy as np
import tack

tack.init(arch=tack.cpu)
array = np.arange(12, dtype=np.float32).reshape(3, 4)
field = tack.field_like(array)
array[:] = -1                             # does not change the Tack field
np.testing.assert_array_equal(field.to_numpy(), np.arange(12, dtype=np.float32).reshape(3, 4))
```

`field_like` allocates and copies. `from_numpy` replaces the values in an existing
field and converts to its dtype. `to_numpy` produces a NumPy copy. These are
simple boundaries for a first program or for data that arrives infrequently.

## External pointers: borrowed storage

On CPU, a contiguous NumPy array can be wrapped directly:

```python
array = np.zeros(8, dtype=np.float32)
view = tack.field_from_ptr(array, dtype=tack.f32, shape=(8,), writable=True)

@tack.kernel
def fill_indices(out):
    for i in range(out.shape[0]):
        out[i] = float(i)

fill_indices(view)
np.testing.assert_array_equal(array, np.arange(8, dtype=np.float32))
```

The external allocation must remain alive. The supplied dtype and shape must
describe its actual storage, and the address must meet the dtype's alignment
requirements. Tack does not take ownership of a raw pointer. The default view
is read-only; request `writable=True` only when the owner permits writes.

Metal wraps an `MTLBuffer`, while CUDA, HIP and Level Zero wrap device addresses.
Those addresses must belong to the selected device/runtime. Selecting the same
kind of backend does not prove that two separately initialized contexts can
share an allocation. Keep context lifetimes aligned with the view.

## DLPack: exchange a tensor description

For a compatible library that implements DLPack, start with its object:

```python
# producer is a tensor from another library, on a compatible device.
view = tack.from_dlpack(producer)
```

Tack reads the producer's device, dtype and shape and retains the managed tensor
while its field uses the storage. A DLPack import cannot represent arbitrary
strided layouts: contiguous storage is required. Device support is
backend-dependent, and an imported read-only tensor remains read-only.

Metal currently has no zero-copy DLPack import. Use
`tack.from_dlpack(host_tensor, copy=True)` to copy a NumPy-readable host tensor
into a Metal field. Export from a Metal field describes its shared host storage
as `kDLCPU`, so it does not exchange an `MTLBuffer` handle with another GPU API.

Tack fields implement the DLPack protocol too. Pass a field to a consumer that
supports its device. For a CPU field, NumPy offers a compact example:

```python
data = tack.field_like(np.arange(8, dtype=np.float32))
shared_array = np.from_dlpack(data)
```

This shares compatible CPU storage; it does not request migration to the host
from an arbitrary GPU backend. Do not interpret a GPU address as a NumPy pointer.
A raw DLPack capsule can be consumed only once. Passing producer objects is
usually clearer than managing capsules yourself.

See the [runtime contract](../contracts/runtime-api.md#dlpack) for supported
device types, copy requests, read-only behavior and errors. The
[interoperability design](../design/interoperability.md) explains the lifetime
mechanism.

## VTK: tuple and component layout

With `tack-vis` and a compatible VTK build:

```python
from tack.interop.vtk import field_to_vtk, vtk_to_field

field = vtk_to_field(vtk_array)             # VTK tuples × components become field shape
vtk_array = field_to_vtk(field)
```

For a flat interleaved coordinate field, specify the component count when
exporting: `field_to_vtk(points, n_components=3)`. For input to flying edges or
normal calculation, `vtk_to_field(array, flatten=True)` gives their required
flat layout.

Level Zero sharing with VTK needs a shared context. Call
`tack.interop.vtk.init_level_zero()` before creating fields, using a VTK build
that exposes the necessary handles. The [Visualization](09-visualization.md#vtk-interop)
chapter describes that setup; a system VTK wheel does not necessarily offer
every device-interoperability path.

### A complete CPU exchange

The [VTK image example](../examples/vtk_interop.py) creates a `vtkImageData`,
wraps its scalar array with Tack, fills it in a kernel, and exports the field
back to VTK. Each image point stores its squared distance from the centre.
VTK image indices are x-fast, so the kernel decomposes the flat index accordingly:

```python
--8<-- "docs/examples/vtk_interop.py:kernel"
```

The two conversions below share the allocation and retain its owners:

```python
--8<-- "docs/examples/vtk_interop.py:share"
```

The script compares VTK's values against a NumPy reference and checks that a
write through Tack appears in both VTK arrays. It uses the CPU backend;
Metal cannot import this VTK host allocation without copying. A GPU exchange
needs compatible device arrays and the context rules described above.

Use a VTK build exposing `vtkmodules.util.dlpack_support`. Its Python version
must match the interpreter running Tack. Add the build's Python site-packages
directory to `PYTHONPATH` if it is not installed in that environment. A VTK
wheel sufficient for the tutorial figures may lack this DLPack module.

```bash
uv run python docs/examples/vtk_interop.py --output radius.vti
uv run python docs/examples/validate.py --arch cpu --vtk
```

`radius.vti` can be read by VTK or ParaView. The validator's VTK option is
explicit because the usual CPU and GPU numerical checks do not need VTK.

```python
--8<-- "docs/examples/vtk_interop.py"
```

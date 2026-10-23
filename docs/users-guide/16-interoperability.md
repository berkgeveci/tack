# Sharing arrays with other libraries

There are two choices when another library produces your data: copy its values
into a Tack array, or share compatible storage. A copy gives Tack an independent
allocation. A shared view needs agreement on the device, layout, lifetime and
permission to write.

## NumPy: an independent array

```python
import numpy as np
import tack

tack.set_default_device("cpu")
host = np.arange(12, dtype=np.float32).reshape(3, 4)
data = tack.array(host)
host[:] = -1                              # does not change the Tack array
np.testing.assert_array_equal(data.numpy(), np.arange(12, dtype=np.float32).reshape(3, 4))
tack.copyto(data, host)                   # now it does
np.testing.assert_array_equal(np.asarray(data), host)
```

`tack.array` allocates and copies. `tack.copyto` replaces the values in an
existing array and converts to its dtype. `numpy()` and `np.asarray` produce
a NumPy copy. These are simple boundaries for a first program or for data
that arrives infrequently.

## External pointers: borrowed storage

On CPU, a contiguous NumPy array can be wrapped directly:

```python
array = np.zeros(8, dtype=np.float32)
view = tack.array_from_ptr(array, (8,), tack.f32, writable=True)

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
while its array uses the storage. A DLPack import cannot represent arbitrary
strided layouts: contiguous storage is required. Device support is
backend-dependent, and an imported read-only tensor remains read-only.

Metal currently has no zero-copy DLPack import. Use
`tack.from_dlpack(host_tensor, copy=True)` to copy a NumPy-readable host tensor
into a Metal array. Export from a Metal array describes its shared host storage
as `kDLCPU`, so it does not exchange an `MTLBuffer` handle with another GPU API.

Tack arrays implement the DLPack protocol too. Pass an array to a consumer that
supports its device. For a CPU array, NumPy offers a compact example:

```python
data = tack.array(np.arange(8, dtype=np.float32))
shared_array = np.from_dlpack(data)
```

This shares compatible CPU storage; it does not request migration to the host
from an arbitrary GPU backend. Do not interpret a GPU address as a NumPy pointer.
A raw DLPack capsule can be consumed only once. Passing producer objects is
usually clearer than managing capsules yourself.

On Metal, CUDA, HIP and Level Zero a kernel call returns once its launch is
queued (see [When a call returns](08-backends.md#when-a-call-returns)), so
memory another library reads is kept in step for it. Exporting an array
through DLPack waits for the launches queued before it; from then on, every
launch that uses that array completes before the call returns, and so does
every launch that uses a wrapped pointer or a DLPack import. The other
library can read the results as soon as the Tack call returns; the cost is
a wait on each such launch.

See the [runtime contract](../contracts/runtime-api.md#dlpack) for supported
device types, copy requests, read-only behavior and errors. The
[interoperability design](../design/interoperability.md) explains the lifetime
mechanism.

## VTK: tuple and component layout

With `tack-vis` and a compatible VTK build:

```python
from tack.interop.vtk import array_to_vtk, vtk_to_array

data = vtk_to_array(vtk_array)              # VTK tuples × components become the array shape
vtk_array = array_to_vtk(data)
```

For a flat interleaved coordinate array, specify the component
count when exporting: `array_to_vtk(points, n_components=3)`; an array of
`vec3` elements needs no count. `vtk_to_array(array, flatten=True)` gives a
flat array of `tuples * components` values.

Level Zero sharing with VTK needs a shared context. Call
`tack.interop.vtk.init_level_zero()` before creating arrays, using a VTK build
that exposes the necessary handles. The [Visualization](09-visualization.md#vtk-interop)
chapter describes that setup; a system VTK wheel does not necessarily offer
every device-interoperability path.

### A complete CPU exchange

The [VTK image example](../examples/vtk_interop.py) creates a `vtkImageData`,
wraps its scalar array with Tack, fills it in a kernel, and exports the Tack
array back to VTK. Each image point stores its squared distance from the centre.
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

Tack's VTK interop needs VTK 9.7.1 or later, for VTK's pythonic API
(`uv sync --extra interop` installs it). Zero-copy array exchange also needs
a VTK build exposing `vtkmodules.util.dlpack_support`. Its Python version
must match the interpreter running Tack. Add the build's Python site-packages
directory to `PYTHONPATH` if it is not installed in that environment. A VTK
wheel sufficient for the tutorial figures may lack this DLPack module. The
PyPI wheel that `uv sync --extra interop` installs (VTK 9.7.1) lacks it; it
is enough for `vtk_to_dataset` and `dataset_to_vtk`, which copy the arrays.

```bash
uv run python docs/examples/vtk_interop.py --output radius.vti
uv run python docs/examples/validate.py --arch cpu --vtk
```

`radius.vti` can be read by VTK or ParaView. The validator's VTK option is
explicit because the usual CPU and GPU numerical checks do not need VTK.

```python
--8<-- "docs/examples/vtk_interop.py"
```

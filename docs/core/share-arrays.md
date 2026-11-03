# Share arrays with NumPy, PyTorch, CuPy and VTK

Data usually comes from somewhere else: a NumPy script, a PyTorch model, a
simulation's own GPU memory, a VTK pipeline. This page is for anyone who needs
to get such data into Tack kernels and the results back out. After reading it
you can copy values in and out, share memory with other libraries through
DLPack without copying, wrap a raw pointer from a simulation, and exchange
arrays with VTK.

There are two ways in. A *copy* gives Tack its own array, independent of the
source. A *shared* array uses the source's memory, so a write through either
side is seen by the other:

<!-- tack-check: share-first -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def add_one(a):
    for i in range(a.shape[0]):
        a[i] = a[i] + 1.0


host = np.zeros(4, dtype=np.float32)

copied = tack.array(host)          # Tack's own copy
add_one(copied)
assert host[0] == 0.0              # the source is unchanged

shared = tack.from_dlpack(host)    # the same memory, no copy
add_one(shared)
assert host[0] == 1.0              # the kernel wrote into the NumPy array

view = np.from_dlpack(shared)      # and back out, still no copy
add_one(shared)
assert view[0] == 2.0
```

## Decide whether to copy or share

A copy is the simple choice. It suits a first program, and data that arrives
once or rarely. Sharing saves the copy, and with it time and memory. In
exchange, both libraries must agree on four things: the device the memory is
on, its layout, how long it lives, and whether Tack may write to it.

| to | copy | share |
|---|---|---|
| bring data in | `tack.array(x)`, `tack.copyto(a, x)` | `tack.from_dlpack(x)`, `tack.array_from_ptr(ptr, ...)` |
| take results out | `a.numpy()`, `np.asarray(a)` | `np.from_dlpack(a)`, `torch.from_dlpack(a)`, `cupy.from_dlpack(a)` |
| exchange with VTK | `vtk_to_dataset`, `dataset_to_vtk` | `vtk_to_array`, `array_to_vtk` |

## Copy values in and out

`tack.array(x)` makes a new array holding a copy of `x`: a NumPy array, a
sequence, or another Tack array. It keeps a NumPy array's element type unless
you pass `dtype=`. `tack.copyto(a, x)` copies `x` into an existing array,
converting to its element type and broadcasting as `np.copyto` does.
`a.numpy()` and `np.asarray(a)` copy an array into a new NumPy array.

A vector or matrix array's components become trailing NumPy dimensions. An
array of 100 `vec3` elements reads back with shape `(100, 3)`, and
`tack.array(points, tack.vec3())` reads a `(100, 3)` NumPy array as 100
vectors. [Arrays and types](arrays-and-types.md) covers element types and
shapes.

## Share through DLPack

DLPack is the protocol Python array libraries use to hand each other memory
without copying. NumPy, PyTorch, CuPy, JAX and VTK speak it, and so does Tack.

`tack.from_dlpack(x)` wraps anything that has a `__dlpack__` method, or a
DLPack capsule. The array uses `x`'s memory and keeps `x` alive for as long
as the array lives, so you may drop `x` at once. It works for GPU tensors as
well as host ones. Going the other way, a Tack array has `__dlpack__`, so any
consumer can take it:

<!-- tack-check: share-dlpack -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")


@tack.kernel
def smooth(values, out):
    for i in range(values.shape[0]):
        left = values[max(i - 1, 0)]
        right = values[min(i + 1, values.shape[0] - 1)]
        out[i] = 0.25 * left + 0.5 * values[i] + 0.25 * right


signal = np.sin(np.linspace(0.0, 3.0, 64)).astype(np.float32)
values = tack.from_dlpack(signal)          # shares signal's memory
result = tack.empty_like(values)
smooth(values, result)
smoothed = np.from_dlpack(result)          # shares result's memory
assert abs(smoothed[10] - (0.25 * signal[9] + 0.5 * signal[10] + 0.25 * signal[11])) < 1e-6

# The same calls with PyTorch or CuPy, on a GPU:
#     values = tack.from_dlpack(torch_tensor)        # a CUDA tensor gives a CUDA array
#     torch_result = torch.from_dlpack(result)
#     cupy_result = cupy.from_dlpack(result)

read_only = np.zeros(4, dtype=np.float32)
read_only.flags.writeable = False
frozen = tack.from_dlpack(read_only)
try:
    smooth(values, frozen)                 # a kernel may not write it
except ValueError as error:
    print(error)
else:
    raise AssertionError("a read-only import was written")
```

Four rules govern an import:

- **The device.** The array goes on the default device when that device can
  wrap the memory, and otherwise on the memory's own device. A CUDA tensor
  imported while the CPU is the default becomes a CUDA array, not a copy.
  Pass `device=` to insist on one. A device that cannot wrap the memory then
  raises `RuntimeError`.
- **The layout.** Only C-contiguous memory can be wrapped. A strided view,
  such as every second column, raises `ValueError`. Make it contiguous in
  the producing library first, or copy it.
- **Write permission.** A tensor its producer marks read-only becomes a
  read-only array. Host writes and kernel stores to it raise `ValueError`.
- **A copy on request.** `tack.from_dlpack(x, copy=True)` makes an
  independent array instead. It reads `x` through NumPy, so it works only
  for memory NumPy can read on the host.

A raw capsule can be consumed only once. Passing the producer's object,
rather than a capsule you made yourself, avoids that trap.

## Wrap a pointer from a simulation

A simulation code written in C, C++ or Fortran often has its arrays as raw
addresses, with no DLPack object around them. `tack.array_from_ptr` wraps
such memory as a Tack array without copying:

<!-- tack-check: share-pointer -->
```python
import numpy as np
import tack

tack.set_default_device("cpu")

# Memory a simulation owns: 5 points of 3 doubles each, as a C code holds them.
positions = np.random.default_rng(1).normal(size=(5, 3))
before = positions.copy()
address = positions.ctypes.data

points = tack.array_from_ptr(address, (5,), tack.vec3(tack.f64), writable=True)
assert points.shape == (5,) and tack.memory_space(address) == "cpu"


@tack.kernel
def shift_x(points, dx):
    for i in range(points.shape[0]):
        points[i] = points[i] + tack.vec3(dx, 0.0, 0.0)


shift_x(points, 1.0)                       # writes the simulation's memory
assert np.allclose(positions, before + [1.0, 0.0, 0.0])
```

The pointer is an integer address on the CPU, CUDA, HIP and Level Zero. On
the CPU a NumPy array works too. On Metal it is an `MTLBuffer` object, or a
NumPy array of host memory that covers whole pages.

Tack does not own wrapped memory and never frees it. You are responsible for
four things:

- **Lifetime.** The memory must outlive the array and every kernel call that
  uses it.
- **Shape and type.** `shape` and `dtype` must describe the memory as it is.
  The address must be aligned for the element type.
- **The device.** The pointer must belong to the device you wrap it on. On
  CUDA, HIP and Level Zero, Tack checks this and refuses host memory with
  `ValueError`. `tack.memory_space(ptr)` says where a pointer is, for
  example `"cuda"`, `"cuda_pinned"` or `"cpu"`.
- **Write permission.** A wrapped array is read-only unless you pass
  `writable=True`. Ask for that only when the owner allows writes.

On CUDA, Tack uses the GPU's primary context, as PyTorch, CuPy, Warp and
the CUDA runtime do. A simulation that allocates through the runtime is in
that context already, so its pointers wrap directly. A Level Zero pointer
means something only in the context that allocated it, so the two libraries
must share one.

## Keep both libraries in step

On a GPU a kernel call returns once its launch is queued (see
[When a call returns](choose-a-device.md#know-when-a-call-returns)). Another
library reading shared memory knows nothing of Tack's queue, so Tack waits
for it. Exporting an array through DLPack waits for the launches queued
before it. From then on, every launch that uses that array finishes before
the call returns. The same holds for every launch that uses a wrapped
pointer or a DLPack import. The other library can therefore read the
results as soon as the Tack call returns. The cost is a wait at each such
launch.

## Share with VTK

The `tack.interop.vtk` module, in the `tack-vis` package, shares VTK data
arrays without copying:

- `vtk_to_array(vtk_array)` wraps a `vtkDataArray`. A VTK array of tuples
  with several components becomes a Tack array of shape
  `(tuples, components)`, and one with a single component a flat array.
  `flatten=True` gives a flat array of `tuples * components` values.
- `array_to_vtk(array, name=None)` wraps a Tack array as a `vtkDataArray`.
  A two-dimensional array and an array of vectors say their component count.
  A flat array of interleaved values needs `n_components=`.

Both hold the other side's owner alive. This example wraps a VTK image's
scalars, fills them in a kernel, and hands the result back. A VTK image's x
index varies fastest, so the kernel takes the flat index apart that way:

```python
--8<-- "docs/examples/vtk_interop.py:kernel"
```

```python
--8<-- "docs/examples/vtk_interop.py:share"
```

Run the whole script, which checks its values against NumPy, with
`uv run python docs/examples/vtk_interop.py --output radius.vti`.

Sharing VTK arrays needs VTK's DLPack module,
`vtkmodules.util.dlpack_support`. The VTK 9.7.1 wheel that
`uv sync --extra interop` installs does not have it. A local VTK build that
has it works when its Python version matches yours and its site-packages
directory is on `PYTHONPATH`. To share Level Zero device arrays, start Tack
with `tack.interop.vtk.init_level_zero()`, which makes a device in VTK's
context the default.

To exchange whole meshes with their fields, rather than single arrays, see
the Visualization layer's [VTK page](../visualization/vtk.md).

## Details

### Metal

Apple silicon shares memory between the CPU and the GPU, but a Metal array
needs an `MTLBuffer`, and an arbitrary host pointer is not one. So Metal has
no zero-copy DLPack import. With Metal as the default, `tack.from_dlpack` of a
host tensor gives a CPU array, and `device="metal"` refuses it with the
reason. Use `copy=True` to make a Metal array from it.

A Metal array exports through DLPack as host memory (`kDLCPU`), which NumPy
can read in place. It does not hand an `MTLBuffer` to another GPU API.
Tensors that carry an `MTLBuffer` (`kDLMetal`) cannot be imported yet, on any
device.

### What an export says

| device | DLPack device type |
|---|---|
| CPU, Metal | `kDLCPU` |
| CUDA | `kDLCUDA` |
| HIP | `kDLROCM` |
| Level Zero | `kDLOneAPI` |

All ten element types export. A consumer that asks for DLPack 1.0 or later
gets a capsule that carries the read-only flag. The `stream` and `dl_device`
arguments of `__dlpack__` are ignored, and `copy=True` raises `BufferError`.

### Memory for other graphics APIs

`a.export_memory()` returns an `ExportedMemory` description of a CUDA or
Metal array's memory, with a handle another API such as Vulkan can import:
a file descriptor or Windows handle on CUDA, the `MTLBuffer` on Metal. An
array allocated with `tack.empty(..., exportable=True)` is exported
directly. Any other array is first copied once into exportable memory.

## Where to go next

- [Choose a device](choose-a-device.md), for where arrays live and when
  calls finish.
- [Memory and parallel correctness](memory-and-correctness.md), for what
  kernels may do with shared and overlapping arrays.
- The [runtime API](../contracts/runtime-api.md#pointer-interop), for every
  argument and error of the sharing functions, and the
  [interoperability design note](../design/interoperability.md) for how
  lifetimes are kept.

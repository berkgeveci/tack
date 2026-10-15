# API

Look up the functions you call from Python here. The signatures and detailed
descriptions below are generated from the source. For a complete program, use
the [User's Guide](../users-guide/index.md); for a reminder of which operations
belong on the host or in a kernel, use the [quick reference](../users-guide/quick-reference.md).

| Task | Call | Result or effect |
|---|---|---|
| Select a backend | `tack.init(arch=tack.cpu, debug=False)` | Initializes the runtime; allocate arrays afterwards |
| Allocate an output | `tack.empty((n,))` | New f32 array with unspecified contents |
| Allocate with another dtype | `tack.empty((n,), tack.i32)` | Any Tack type or NumPy dtype (`np.int32`, `"int32"`) |
| Allocate zeros, ones or a value | `tack.zeros((n,))`, `tack.ones((n,))`, `tack.full((n,), 2.5)` | New array, filled on the device |
| Match storage without copying | `tack.empty_like(prototype)` | Same shape and element dtype; no values read or initialized |
| Match and fill | `tack.zeros_like(a)`, `tack.ones_like(a)`, `tack.full_like(a, v)` | Same shape and element dtype, filled |
| Upload NumPy data | `tack.array(host)` | New array with an independent copy; a NumPy array keeps its dtype |
| Use an array as is | `tack.asarray(obj)` | `obj` itself if it is already a Tack array (of `dtype`, if given), else `tack.array(obj)` |
| Overwrite values from the host | `tack.copyto(a, host)` | Copies into `a`; `host` has `a.numpy()`'s shape or broadcasts to it |
| Copy between arrays | `tack.copyto(dst, src)` | Device-side copy into an existing array of the same shape |
| Download data | `a.numpy()` or `np.asarray(a)` | Independent NumPy array; waits for earlier writes |
| Allocate vector elements | `tack.empty((n,), tack.vec3())` | Array of n 3-vectors; NumPy download shape `(n, 3)` |
| Inspect shape and size | `a.shape`, `a.ndim`, `a.size`, `a.nbytes` | Element counts (components excluded) and storage bytes |
| Reshape without copying | `a.reshape(rows, -1)` | Shares storage; one dimension may be `-1` |
| Wait for device work | `tack.sync()` | Waits for queued launches; returns `None` |
| Inspect a kernel | `tack.inspect(kernel, *args, mode="source")` | Generated source as a string; does not execute the kernel |

Without a `dtype`, an allocation is f32; values from Python make f32 from
floats and i32 from ints (i64 if a value needs it), and a NumPy array keeps
its own dtype, so a float64 array becomes f64.

For ownership, errors and precise guarantees, see the
[Runtime API contract](../contracts/runtime-api.md) and the
[Kernel Language Contract](language-contract.md). Compiler internals are in
the [Developer's Guide](../developers-guide/01-architecture.md).

## Initialization

::: tack.init

::: tack.runtime.dispatch.get_backend

::: tack.sync

::: tack.sync_async

## Arrays

`tack.Array` holds storage on the backend active at allocation. Its shape
counts elements; its dtype describes one scalar, vector or matrix element.
`tack.dataset.Field` instead couples coefficient values with a space on a topology.
See [Arrays and Types](../users-guide/02-fields-and-types.md) for examples.

::: tack.Array

::: tack.lang.field.DeviceBuffer

### Creating arrays

::: tack.empty

::: tack.zeros

::: tack.ones

::: tack.full

::: tack.empty_like

::: tack.zeros_like

::: tack.ones_like

::: tack.full_like

::: tack.array

::: tack.asarray

::: tack.arange

::: tack.concat

### Copying values

::: tack.copyto

### Element types

::: tack.ElementType

::: tack.vec

::: tack.vec2

::: tack.vec3

::: tack.vec4

::: tack.mat

::: tack.mat2

::: tack.mat3

::: tack.mat4

### Kernel values and textures

::: tack.Vector

::: tack.Matrix

::: tack.select

::: tack.texture3d
    options:
      docstring_options:
        warn_missing_types: false

::: tack.lang.field.Texture3D

### Sharing memory

::: tack.array_from_ptr
    options:
      docstring_options:
        warn_missing_types: false

::: tack.memory_space

::: tack.from_dlpack

::: tack.ExportedMemory

::: tack.lang.dlpack.field_to_dlpack

::: tack.lang.dlpack.dlpack_to_field

::: tack.lang.dlpack.dlpack_device

## Kernels and device functions

::: tack.kernel

::: tack.lang.kernel.Kernel

::: tack.func

::: tack.lang.func.Func

::: tack.device_class

::: tack.template

::: tack.constant

::: tack.lang.constant.ArrayConstant

## Shader-style helpers

::: tack.math

## Random numbers

::: tack.random

::: tack.lang.source_validation.UnsupportedSyntaxError

## Inspection

::: tack.inspect
    options:
      docstring_options:
        warn_missing_types: false

## Reductions, statistics and scans

How these behave on each backend — dtypes, accumulators, NaN and empty
inputs, repeatability — is in the User's Guide chapter
[Reductions and Scans](../users-guide/11-reductions-and-scans.md). Array
reductions are the methods `Array.sum`, `min`, `max` and `mean`, above.

### Statistics

Import these from `tack.algorithms`; they are defined in
`tack.algorithms.stats`.

::: tack.algorithms.stats.var

::: tack.algorithms.stats.std

::: tack.algorithms.stats.norm

::: tack.algorithms.stats.absmax

::: tack.algorithms.stats.count_nonzero

::: tack.algorithms.stats.dot

::: tack.algorithms.stats.histogram
    options:
      docstring_options:
        warn_missing_types: false

### Scans

Import these from `tack.algorithms`; they are defined in
`tack.algorithms.scan`.

::: tack.algorithms.scan.exclusive_scan
    options:
      docstring_options:
        warn_missing_types: false

::: tack.algorithms.scan.inclusive_scan
    options:
      docstring_options:
        warn_missing_types: false

### Sorting and segmented reductions

Import these from `tack.algorithms`; they are defined in
`tack.algorithms.sort`.

::: tack.algorithms.sort.argsort

::: tack.algorithms.sort.sort_by_key

::: tack.algorithms.sort.gather

::: tack.algorithms.sort.unique

::: tack.algorithms.sort.reduce_by_key

### Copy and fill

`copy` and `fill_value` are re-exported from `tack.algorithms`;
`copy_with_offset` is imported from `tack.algorithms.copy`.

::: tack.algorithms.copy.copy

::: tack.algorithms.copy.fill_value

::: tack.algorithms.copy.copy_with_offset

### Block reductions

Usable only inside a `@tack.kernel`, on backends with workgroups (not CPU).

::: tack.block_sum

::: tack.block_min

::: tack.block_max

## VTK interop

`tack.interop.vtk` (in the `tack-vis` package) provides `vtk_to_array`,
`array_to_vtk` and `init_level_zero`. Their contracts are in
[Runtime API: VTK](../contracts/runtime-api.md#vtk), and their docstrings
are in `packages/tack-vis/src/tack/interop/vtk.py`.

## The backend contract

Every backend subclasses this. It exists so callers can *ask* what a backend
supports instead of probing for methods — the distinction that
`hasattr(backend, 'reduce_field')` got wrong, since it answers "is this
defined" rather than "is this supported".

::: tack.runtime.backend.Backend

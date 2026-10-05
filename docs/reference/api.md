# API

Generated from the source. Tack's docstrings are prose rather than
`Args:`/`Returns:` tables — they tend to explain why something is the way it
is, which is usually the part that is hard to recover later. Read them as
notes, not as a specification.

For how these pieces fit together, start with the
[Developer's Guide](../developers-guide/01-architecture.md).

For the guarantees, caller constraints and errors of these entry points,
see the [Runtime API contract](../contracts/runtime-api.md). Where a
docstring and the contract disagree, the contract describes the current
behavior.

## Initialization

::: tack.runtime.dispatch.init

::: tack.runtime.dispatch.get_backend

## Fields

The container everything else operates on: an n-dimensional array bound to
whichever backend was active when it was allocated.

::: tack.lang.field.Field

::: tack.lang.field.DeviceBuffer

### Creating fields

::: tack.lang.field.field

::: tack.lang.field.field_like

::: tack.lang.field.zeros

::: tack.lang.field.ones

::: tack.lang.field.full

::: tack.lang.field.arange

::: tack.lang.field.concat

::: tack.lang.field.Vector

::: tack.texture3d
    options:
      docstring_options:
        warn_missing_types: false

::: tack.lang.field.Texture3D

### Sharing memory

::: tack.lang.field.field_from_ptr
    options:
      docstring_options:
        warn_missing_types: false

::: tack.lang.field.memory_space

::: tack.lang.field.from_dlpack

::: tack.lang.field.ExportedMemory

::: tack.lang.dlpack.field_to_dlpack

::: tack.lang.dlpack.dlpack_to_field

::: tack.lang.dlpack.dlpack_device

## Kernels and device functions

::: tack.lang.kernel.kernel

::: tack.lang.kernel.Kernel

::: tack.lang.func.func

::: tack.lang.func.Func

::: tack.lang.data_oriented.data_oriented

::: tack.lang.types.template

::: tack.lang.source_validation.UnsupportedSyntaxError

## Inspection

::: tack.lang.inspect_kernel.inspect
    options:
      docstring_options:
        warn_missing_types: false

## VTK interop

`tack.interop.vtk` (in the `tack-vis` package) provides `vtk_to_field`,
`field_to_vtk` and `init_level_zero`. Their contracts are in
[Runtime API: VTK](../contracts/runtime-api.md#vtk), and their docstrings
are in `packages/tack-vis/src/tack/interop/vtk.py`.

## The backend contract

Every backend subclasses this. It exists so callers can *ask* what a backend
supports instead of probing for methods — the distinction that
`hasattr(backend, 'reduce_field')` got wrong, since it answers "is this
defined" rather than "is this supported".

::: tack.runtime.backend.Backend

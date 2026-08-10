# API

Generated from the source. Tack's docstrings are prose rather than
`Args:`/`Returns:` tables — they tend to explain why something is the way it
is, which is usually the part that is hard to recover later. Read them as
notes, not as a specification.

For how these pieces fit together, start with the
[Developer's Guide](../developers-guide/01-architecture.md).

## Fields

The container everything else operates on: an n-dimensional array bound to
whichever backend was active when it was allocated.

::: tack.lang.field.Field

::: tack.lang.field.DeviceBuffer

## Kernels

::: tack.lang.kernel.Kernel

## The backend contract

Every backend subclasses this. It exists so callers can *ask* what a backend
supports instead of probing for methods — the distinction that
`hasattr(backend, 'reduce_field')` got wrong, since it answers "is this
defined" rather than "is this supported".

::: tack.runtime.backend.Backend

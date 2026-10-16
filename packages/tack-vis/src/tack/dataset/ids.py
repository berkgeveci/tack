"""The integer type of a topology's ids: ``i32``, or ``i64`` when it is too large.

Every id a topology stores or derives -- point and cell ids, offsets into its
connectivity, faces, edges, links, the orders and keys that sort them -- has
one type, the topology's ``id_dtype``. It is ``i32`` unless the topology
asks for ``i64`` (``id_dtype=tack.i64``), is built from ``i64`` fields, or
has an id, offset or derived array past ``2**31 - 1`` entries; a topology
asked for ``i32`` that is too large for it is refused, never wrapped. Kernels
specialize on their fields' types, so the same kernels serve both.

A filter's output takes ``i64`` when its input's ids are ``i64`` or its own
can be too large for ``i32`` (``for_output``), so a pipeline that starts in
``i64`` stays there.
"""

import numpy as np

import tack
from tack.lang.array import Array

__all__ = ["LIMIT", "as_ids", "at_least", "check_bounds", "choose", "for_output"]

#: The largest id, offset or count an ``i32`` holds.
LIMIT = 2**31 - 1

#: The narrowest id type chosen. The test suite sets ``tack.i64`` (``--ids=i64``)
#: to run every path in 64 bits at sizes that fit in 32.
_minimum = tack.i32


def choose(dtype=None, extent=0, given=()):
    """The id type for ids, offsets and counts up to ``extent``: ``dtype`` if given
    (refused if ``i32`` cannot hold ``extent``), else ``i64`` when ``extent``
    needs it or any field in ``given`` is ``i64``, else ``i32``."""
    if dtype is None:
        wide = (extent > LIMIT or _minimum == tack.i64
                or any(isinstance(f, Array) and f.dtype == tack.i64 for f in given))
        return tack.i64 if wide else tack.i32
    if dtype not in (tack.i32, tack.i64):
        raise TypeError(f"ids are i32 or i64, not {getattr(dtype, 'name', dtype)}")
    if dtype == tack.i32 and extent > LIMIT:
        raise ValueError(f"ids reach {extent}, past the i32 limit {LIMIT}: use tack.i64")
    return dtype


def at_least(dtype, extent):
    """``dtype``, or ``i64`` when an index array of it must reach ``extent``: the
    type of an array derived from a topology that may hold more than its ids do
    (an order-2 field's values, an L2 field's)."""
    return tack.i64 if dtype == tack.i64 or extent > LIMIT else tack.i32


def for_output(data, extent=0):
    """The id type of a filter's output from ``data`` whose ids, offsets and counts
    reach at most ``extent``: ``data``'s own, or ``i64`` when ``extent`` needs it."""
    return at_least(data.id_dtype, extent)


@tack.kernel
def _convert(src, dst):
    for i in range(dst.shape[0]):
        dst[i] = src[i]


def as_ids(values, dtype, check=True):
    """``values`` -- a field or an array of ids -- as a field of ``dtype``; a field
    already of it is returned as is. Ids ``i32`` cannot hold are refused --
    unless ``check`` is false, for a topology's own arrays, which ``choose``
    has already sized: a valid mesh's ids lie below its point count, its
    offsets within its connectivity."""
    if isinstance(values, Array):
        if values.dtype == dtype:
            return values
        if values.dtype not in (tack.i32, tack.i64):
            raise TypeError(f"ids are i32 or i64 arrays, not {values.dtype.name}")
        out = tack.empty(values.shape, dtype)
        if values.size:
            if check and dtype == tack.i32:
                _check_fits(values.numpy())
            _convert(values, out)
        return out
    values = np.ascontiguousarray(values)
    if check and values.size and dtype == tack.i32:
        _check_fits(values)
    out = tack.empty(values.shape, dtype)
    if values.size:
        tack.copyto(out, values.astype(dtype.numpy_dtype))
    return out


@tack.kernel
def _count_outside(ids, n, bad):
    for i in range(ids.shape[0]):
        k = ids[i]
        if k < 0 or k >= n:
            tack.atomic_add(bad, 0, 1)


def check_bounds(ids, n, what):
    """Refuse ``ids`` (a field) outside ``[0, n)`` with ``IndexError``: counted
    in a kernel on the ids' device, so only the count comes back to the host.
    A kernel given one would read and write outside the arrays it indexes."""
    if ids.shape[0]:
        bad = tack.zeros((1,), tack.i32, device=ids.device)
        _count_outside(ids, n, bad)
        if bad[0]:
            raise IndexError(f"{int(bad[0])} {what} ids lie outside [0, {n})")


def host_or_field(values):
    """A field as it is; anything else (a list, an array) as an array."""
    return values if isinstance(values, Array) else np.asarray(values)


def _check_fits(values):
    if values.size and (values.max() > LIMIT or values.min() < -LIMIT - 1):
        raise ValueError(f"ids up to {int(values.max())} do not fit i32: use tack.i64")

"""Small helpers every algorithm shares: fields by name or object, new
fields of values, compaction of 0/1 flags into ids, closing offsets."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.data import arrays
from tack.data.arrays import size_of
from tack.data.ids import at_least


def _vectors(n, dtype, width=3):
    return tack.Vector.field(width, dtype, shape=(n,))


def _scalars(n, dtype):
    return tack.field(dtype, shape=(n,))


def _same_space(*fields):
    space = fields[0].space
    if any(f.space is not space for f in fields[1:]):
        raise ValueError("the fields must be in one space")
    n = size_of(fields[0].values)
    return space, n


def _field(data, field):
    """A field by name or object, checked against the dataset before launch."""
    from tack.data.dataset import Field

    field = data.fields[field] if isinstance(field, str) else field
    if not isinstance(field, Field):
        raise TypeError("a field argument is a tack.data.Field or its name")
    if field.space.topology is not data.topology:
        raise ValueError("a field argument belongs to another topology")
    return field


def _floating_dtype(*values):
    """Fractional arithmetic: f64 for an f64 operand, otherwise f32.

    Integer storage is converted before arithmetic; large integers therefore
    follow the chosen floating precision, rather than integer overflow rules.
    """
    return tack.f64 if any(arrays.dtype_of(v) is tack.f64 for v in values) else tack.f32


def _converted(values, dtype):
    """Materialize in the computation's type before doing arithmetic."""
    return arrays.materialize(values if arrays.dtype_of(values) is dtype
                              else arrays.Cast(values, dtype))


def _same_dtype(name, *fields):
    """Same-type vector operations preserve storage; conversions are explicit."""
    if len({arrays.dtype_of(f.values) for f in fields}) != 1:
        raise TypeError(f"{name}'s fields differ in type: convert them to one with Cast")


def _scalar_floating(name, field):
    if arrays.width_of(field.values):
        raise TypeError(f"{name} needs a scalar field, not one of "
                        f"{arrays.width_of(field.values)}-vectors")
    dtype = arrays.dtype_of(field.values)
    if dtype not in (tack.f32, tack.f64):
        raise TypeError(f"{name} needs floating-point values; the field is {dtype.name}")


@tack.kernel
def _close(offsets, n, total):
    for i in range(1):
        offsets[n] = total


@tack.kernel
def _kept_ids(flags, slots, out):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            out[slots[i]] = i


def _compact(flags, dtype):
    """``(new ids, kept ids, count)`` of a 0/1 field: each kept index's new number,
    and the kept indices in order, both ``dtype``."""
    n = flags.shape[0]
    slots = tack.field(dtype, shape=(n,))
    count = exclusive_scan(flags, slots, n) if n else 0
    kept = tack.field(dtype, shape=(count,))
    if count:
        _kept_ids(flags, slots, kept)
    return slots, kept, count


def _all(n):
    return tack.arange(n, at_least(tack.i32, n))

"""Small helpers every algorithm shares: fields by name or object, new
fields of values, compaction of 0/1 flags into ids, closing offsets."""

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import size_of
from tack.dataset.ids import at_least


@tack.func
def _crossing(level, a, b):
    """Where ``level`` lies between ``a`` and ``b``: 0 at ``a``, 1 at ``b``. A value
    that is not finite is never reached, so a crossing beside one lies at the
    other end, rather than at the NaN its difference gives."""
    w = (level - a) / (b - a)
    if a - a != 0.0:        # true for infinity and NaN alike
        w = 1.0
    elif b - b != 0.0:
        w = 0.0
    return w


def _vectors(n, dtype, width=3):
    return tack.empty((n,), tack.vec(width, dtype))


def _scalars(n, dtype):
    return tack.empty((n,), dtype)


def _same_space(*fields):
    space = fields[0].space
    if any(f.space is not space for f in fields[1:]):
        raise ValueError("the fields must be in one space")
    n = size_of(fields[0].values)
    return space, n


def _field(data, field):
    """A field by name or object, checked against the dataset before launch."""
    from tack.dataset.dataset import Field

    field = data.fields[field] if isinstance(field, str) else field
    if not isinstance(field, Field):
        raise TypeError("a field argument is a tack.dataset.Field or its name")
    if field.space.topology is not data.topology:
        raise ValueError("a field argument belongs to another topology")
    return field


def _with_basis(data, field):
    """Values on points or cells read as the space they are: order-1 ``H1`` and
    ``Constant``, so every algorithm that takes a field with a basis takes them
    (``point_ids``, a gradient at the points)."""
    from tack.dataset.dataset import Field
    from tack.dataset.spaces import H1, Constant, Values

    space = field.space
    if isinstance(space, Values) and space.on == "points":
        return Field(H1(data), field.values, field.unit)
    if isinstance(space, Values) and space.on == "cells":
        return Field(Constant(data), field.values, field.unit)
    return field


def _cell_dimension(data):
    """The one dimension of ``data``'s cells (3 for solids, 2 for surfaces, 1 for
    lines), or ``None`` where they mix dimensions or there are none."""
    dims = {group.shape.DIMENSION for group in data.topology.groups() if group.count}
    return dims.pop() if len(dims) == 1 and 0 not in dims else None


def _measure_unit(data):
    """The unit of ``data``'s cells' measures: the spatial unit to the cells'
    dimension (volume, area, length), or ``None`` where the cells mix
    dimensions or the spatial unit is unknown."""
    from tack.dataset import units

    k = _cell_dimension(data)
    return None if k is None else units.derived((data.unit, k))


def _time_unit(data, velocity):
    """The unit of time a particle moved by ``velocity`` (a field or its name) keeps:
    the spatial unit over the velocity's, or ``None`` if either is unknown."""
    from tack.dataset import units

    return units.derived(data.unit, (_field(data, velocity).unit, -1))


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
    slots = tack.empty((n,), dtype)
    count = exclusive_scan(flags, slots, n) if n else 0
    kept = tack.empty((count,), dtype)
    if count:
        _kept_ids(flags, slots, kept)
    return slots, kept, count


def _all(n):
    return tack.arange(n, dtype=at_least(tack.i32, n))

# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers every algorithm shares: fields by name or object, new
fields of values, compaction of 0/1 flags into ids, closing offsets."""

import weakref

import numpy as np

import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.arrays import size_of
from tack.dataset.ghosts import _before
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


@tack.func
def _first_end(xa, xb, a, b):
    """Whether an edge's crossing is interpolated from its end ``a`` (at ``xa``) rather
    than ``b``: the end first by position (x, then y, then z, as ghost marks give
    a crossing its owner), the lower id only between coincident points. Point
    ids are a piece's own, positions every piece's alike, so every piece -- and
    every cell -- holding the edge computes its crossing, and the values carried
    there, from the same end, to the same bits."""
    return _before(xa, xb) or (not _before(xb, xa) and a < b)


@tack.func
def _end_hit(level, a, b, va, vb, xa, xb):
    """The end (``a`` or ``b``, point ids at ``xa`` and ``xb`` with values ``va`` and
    ``vb``) that an edge's crossing of ``level`` is, or -1: an end whose value is
    the level, or one the crossing's position -- interpolated from the end first
    by position, as every cut interpolates -- lies within half an ulp of the
    edge's largest coordinate of (``s + d == s``), which its coordinates cannot
    tell from the end. A level passing a hair from a point (a symmetry plane's
    points at y = 1e-45, or a plane through grid points whose distances round
    to 1e-7) crosses every edge there at, or an ulp from, that point, so several
    crossings would land on one position; named by the end, they are one point,
    owned as the point is, and pieces can match their points by position.
    Values and positions are every piece's alike, so every piece decides alike."""
    hit = -1
    if va == level:
        hit = a
    elif vb == level:
        hit = b
    else:
        first = _first_end(xa, xb, a, b)
        x0 = xa if first else xb
        x1 = xb if first else xa
        p = x0 + _crossing(level, va if first else vb, vb if first else va) * (x1 - x0)
        scale = max(abs(xa).max(), abs(xb).max())
        if scale + abs(p - x0).max() == scale:
            hit = a if first else b
        elif scale + abs(p - x1).max() == scale:
            hit = b if first else a
    return hit


@tack.func
def edge_crossing(level, a, b, va, vb, xa, xb):
    """Where ``level`` crosses the edge between points ``a`` and ``b`` (ids, at
    positions ``xa`` and ``xb``, with values ``va`` and ``vb``), as every built-in
    cut places it: ``(first, second, weight)``, ``Interpolate``'s ends and
    weight, for a ``@tack.kernel`` writing them.

    - The crossing is interpolated from the end first by position (x, then y,
      then z), so every piece and cell holding the edge computes it, and the
      values ``carry`` brings there, to the same bits; ghost marks give the
      point that end's owner.
    - At an end whose value is the level, or one the crossing lies within half
      an ulp of (relative to the edge's largest coordinate), the crossing is that
      end: ``(end, end, 0)``. Every edge's crossing there is then one point --
      name it by the end, as the built-ins do, so the cut has no two points at
      one position.
    """
    hit = _end_hit(level, a, b, va, vb, xa, xb)
    first = a
    second = b
    w = _crossing(level, va, vb)
    if hit >= 0:
        first = hit
        second = hit
        w = 0.0
    elif not _first_end(xa, xb, a, b):
        first = b
        second = a
        w = _crossing(level, vb, va)
    return first, second, w


#: Constant tables as device arrays, per backend (released with it) and key.
_CONSTANTS = weakref.WeakKeyDictionary()


def _table(key, values, dtype=tack.i32):
    """``values`` (a sequence, or a function giving one) as a constant array of
    ``dtype`` on the current device, made once per device and ``key``: a filter's
    case tables and corner orders, which uploading on every call cost a small
    mesh's filter as much as its kernels (clip's tables, 0.25 ms a call). Never
    to be written to."""
    from tack.runtime.dispatch import get_backend

    per_device = _CONSTANTS.setdefault(get_backend(), {})
    full = (key, dtype)
    if full not in per_device:
        made = values() if callable(values) else values
        per_device[full] = tack.array(np.asarray(made, dtype.numpy_dtype).reshape(-1), dtype)
    return per_device[full]


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


_RESOLVERS = []
_DERIVED = {}             # name -> {location: compute}
_LOCATIONS = ("points", "cells", "faces")


def register_derived_field(name, compute, at):
    """Lets ``name`` stand, at ``at`` (``"points"``, ``"cells"`` or ``"faces"``), for a
    field a dataset makes from itself when it stores none of that name there:
    ``compute(data)`` gives the ``Field`` on ``data``, made the first time it is
    asked for and kept with the dataset while its geometry is the same. One name
    is one quantity, registered at each location it has: Tack registers
    ``normals`` (a surface's at its points and cells, ``surface_normals``'; every
    mesh's at its faces, ``face_geometry``'s), ``measures`` (each cell's volume, area
    or length; each face's area) and ``centroids`` (each cell's)."""
    if at not in _LOCATIONS:
        raise ValueError(f"at is one of {_LOCATIONS}, not {at!r}")
    _DERIVED.setdefault(name, {})[at] = compute


def derived_names():
    """The names ``register_derived_field`` has registered."""
    return tuple(_DERIVED)


def _location(field):
    from tack.dataset.spaces import H1, L2, Constant, Values

    space = field.space
    if isinstance(space, H1) or (isinstance(space, Values) and space.on == "points"):
        return "points"
    if isinstance(space, (Constant, L2)) or (isinstance(space, Values) and space.on == "cells"):
        return "cells"
    if isinstance(space, Values) and space.on == "faces":
        return "faces"
    return None


def _derived(data, name, at):
    made = data.__dict__.setdefault("_derived_fields", {})
    hit = made.get((name, at))
    if hit is not None and hit[0] is data.geometry:
        return hit[1]
    field = _DERIVED[name][at](data)
    made[(name, at)] = (data.geometry, field)
    return field


def named_field(data, name, at=None):
    """The field ``name`` names on ``data``, read at ``at`` (``"points"``, ``"cells"``,
    ``"faces"``, or None for wherever it is). A field stored under the name there
    comes first, then the one the dataset derives there
    (``register_derived_field``), then one stored under the name elsewhere. With
    no ``at``, the stored field, else the derived one at the first location it is
    registered at that the dataset has. A ``KeyError`` for none."""
    if not isinstance(name, str) or name not in _DERIVED:
        return data.fields[name]
    stored = data.fields.named(name)
    places = _DERIVED[name]
    if at is not None:
        for field in stored:
            if _location(field) == at:
                return field
        if at in places:
            return _derived(data, name, at)
        return data.fields[name]
    if stored:
        return data.fields[name]
    reasons = []
    for where in places:
        try:
            return _derived(data, name, where)
        except (ValueError, TypeError, NotImplementedError) as e:
            reasons.append(f"at {where}: {e}")
    raise KeyError(f"{name!r} is derived from no part of this dataset ({'; '.join(reasons)})")


def register_field_resolver(resolve):
    """Lets every algorithm's field argument be text that names no field: ``resolve(data,
    text)`` gives a ``Field`` on ``data`` for it, or ``None`` when the text is not its kind
    (the field expressions of ``tack_expr`` register one). The last registered is asked
    first."""
    if resolve not in _RESOLVERS:
        _RESOLVERS.append(resolve)


def _resolved(data, text):
    if not isinstance(text, str) or text in data.fields or text in _DERIVED:
        return named_field(data, text)
    for resolve in reversed(_RESOLVERS):
        field = resolve(data, text)
        if field is not None:
            return field
    return data.fields[text]


def _field(data, field):
    """A field by name, object or registered resolver (``register_field_resolver``),
    checked against the dataset before launch."""
    from tack.dataset.dataset import Field, is_field_key

    field = _resolved(data, field) if is_field_key(field) else field
    if not isinstance(field, Field):
        raise TypeError("a field argument is a tack.dataset.Field, its name, or text a "
                        "registered resolver reads")
    if field.space.topology is not data.topology:
        raise ValueError("a field argument belongs to another topology")
    return field


def _stored_field(data, field):
    """``_field``, with a field evaluated where it is read (an expression's) replaced
    by its values, as its space's ``stored`` gives them: for an algorithm that
    reduces a field's values rather than reading it through its view."""
    field = _field(data, field)
    stored = getattr(field.space, "stored", None)
    return stored(field, data) if callable(stored) else field


def _with_basis(data, field):
    """Values on points or cells read as the space they are: order-1 ``H1`` and
    ``Constant``, so every algorithm that takes a field with a basis takes them
    (``point_ids``, a gradient at the points)."""
    from tack.dataset.dataset import Field
    from tack.dataset.spaces import H1, Constant, Values

    space = field.space
    if isinstance(space, Values) and space.on == "points":
        return Field(H1(data), field.values, field.unit, gradient=field.gradient)
    if isinstance(space, Values) and space.on == "cells":
        return Field(Constant(data), field.values, field.unit, gradient=field.gradient)
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
    the spatial unit over the velocity's, or ``None`` if either is unknown. Over a
    partitioned dataset, every rank's pieces'."""
    from tack.dataset import units
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        every = data.world._allgather([piece.fields[velocity].unit for piece in data])
        speed = units.common("the pieces' velocity", [u for part in every for u in part])
        return units.derived(_spatial_unit(data), (speed, -1))
    return units.derived(data.unit, (_field(data, velocity).unit, -1))


def _spatial_unit(data):
    """A dataset's spatial unit, or a partitioned dataset's: every rank's pieces'
    one unit (a rank holding none agrees with the rest)."""
    from tack.dataset import units
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        every = data.world._allgather([piece.unit for piece in data])
        return units.common("the pieces' geometry", [u for part in every for u in part])
    return data.unit


def _length(data, value, what):
    """A length parameter (a radius, a size, a periodic box) as numbers in ``data``'s
    spatial unit: a quantity (``"10 cm"``, ``(0.1, "m")``) converted, a plain
    number as given."""
    from tack.dataset import units

    if not _has_quantity(value):
        return value
    return units.magnitude(value, _spatial_unit(data), what)


def _tolerance(data, value, who):
    """A matching tolerance: a distance in ``data``'s spatial unit (a quantity
    converted), at least zero."""
    value = _length(data, value, f"{who}'s tolerance")
    if not value >= 0:
        raise ValueError(f"{who}'s tolerance must be at least zero, not {value!r}")
    return float(value)


def _inward(lower, upper, field):
    """``[lower, upper]`` as ``field``'s floating type holds it, rounded inward: the
    least value of the type at or above ``lower``, the greatest at or below
    ``upper``. A value of the field is then in the range exactly when it is in
    ``[lower, upper]`` as given, on every device; the type's nearest values may lie
    just outside it (f32(-0.1) is -0.10000000149), keeping a value VTK, comparing
    in double, leaves out. Integer fields and f64 compare exactly as they are."""
    dtype = arrays.dtype_of(field.values).numpy_dtype
    if not np.issubdtype(dtype, np.floating) or dtype == np.float64:
        return lower, upper
    lo, hi = dtype.type(lower), dtype.type(upper)
    if np.isfinite(lo) and float(lo) < lower:
        lo = np.nextafter(lo, dtype.type(np.inf))
    if np.isfinite(hi) and float(hi) > upper:
        hi = np.nextafter(hi, dtype.type(-np.inf))
    return float(lo), float(hi)


def _in_unit_of(field, value, what):
    """A value compared with ``field``'s (an isovalue, a threshold) as a number in
    its unit: a quantity converted, offsets included (``"20 degC"`` on a field
    in kelvin), a plain number as given."""
    from tack.dataset import units

    return units.magnitude(value, field.unit, what) if _has_quantity(value) else value


def _has_quantity(value):
    from tack.dataset import units

    return units.quantity(value) is not None or (
        isinstance(value, (tuple, list)) and any(units.quantity(v) is not None for v in value))


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


def _csr_rows(offsets, connectivity, cells):
    """The point ids of ``cells`` (CSR ``offsets``/``connectivity``), one after
    another, and each cell's size: one gather on the host, not a slice a
    cell (a Python loop over 8.7M cells took 7 of merging ten pieces' 8
    seconds)."""
    sizes = offsets[cells + 1] - offsets[cells]
    starts = np.concatenate(([0], np.cumsum(sizes)))
    at = (np.arange(starts[-1], dtype=np.int64) - np.repeat(starts[:-1], sizes)
          + np.repeat(offsets[cells], sizes))
    return connectivity[at], sizes

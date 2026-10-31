"""``convert``: a field, or a dataset's field or geometry, in another unit."""

import numpy as np

import tack
from tack import arrays
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.dataset import units
from tack.dataset.dataset import DataSet, Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['convert']


@tack.kernel
def _scaled(values, out, factor, offset):
    for i in range(out.shape[0]):
        out[i] = values[i] * factor + offset


def _times(values, factor, offset):
    """``values`` (any array of floating scalars or vectors) times ``factor`` plus
    ``offset``, a new array of the same element type."""
    values = arrays.materialize(values)
    out = tack.empty_like(values)
    if arrays.size_of(values):
        _scaled(values, out, factor, offset)
    return out


def _converted(given, target, what):
    """``given`` in ``target`` (canonical): its values scaled and offset on the
    device, or as they are where the conversion is a relabelling."""
    factor, offset = units.conversion(given.unit, target)
    if factor == 1.0 and offset == 0.0:
        values = given.values
    elif arrays.dtype_of(given.values).numpy_dtype.kind != "f":
        raise TypeError(f"{what} holds integers, which cannot hold its values in "
                        f"{target!r}: give it floating-point values first")
    else:
        values = _times(given.values, factor, offset)
    return Field(given.space, values, target)


@on_inputs_device
def convert(data, unit, field=None):
    """``data`` -- a ``Field``, or a dataset -- in ``unit``. A field gives a new
    field in the same space, its values scaled (and offset, as from degrees
    Celsius to kelvin) on the device. A dataset gives a new dataset with
    ``field`` (a name) in ``unit``, or its geometry when ``field`` is ``None``,
    the topology, sets, marks and other fields kept.

    The field's unit, or the spatial unit, must be known (``Field.unit``,
    ``DataSet.unit``) and of ``unit``'s dimensions. Converting the geometry
    scales the origin too, in double precision on the host, so survey
    coordinates in US survey feet become metres without losing digits; an
    image stays an image and a rectilinear grid keeps its axes. Positions are
    the dataset's to convert: the geometry field alone would leave the origin
    in the old unit. Integer fields are refused unless the conversion is a
    relabelling (``"1000*m"`` to ``"km"``)."""
    from tack.dataset.algorithms._moving import _moved

    target = units.canonical(unit)
    if isinstance(data, Field):
        if field is not None:
            raise TypeError("convert of a field takes no field name: give the dataset and "
                            "the name, or the field alone")
        return _converted(data, target, "the field")
    if field is None:
        factor, offset = units.conversion(data.unit, target)
        if offset:
            raise ValueError(f"positions in {data.unit!r} are not lengths with a zero")
        values = data.geometry.values
        if isinstance(values, UniformCoordinates):
            moved = UniformCoordinates(values.dims, np.asarray(values._origin) * factor,
                                       np.asarray(values._spacing) * factor,
                                       values._matrix if values.ORIENTED else None,
                                       dtype=values.dtype, device=data.device)
        elif isinstance(values, CartesianProduct):
            axes = [arrays.to_host(axis).astype(np.float64).reshape(-1) * factor
                    for axis in (values.xs, values.ys, values.zs)]
            moved = CartesianProduct(*axes, dtype=values.dtype, device=data.device)
        else:
            moved = _times(values, factor, 0.0)
        out = _moved(data, moved)
        out._geometry = Field(out.geometry.space, out.geometry.values, target)
        out.origin = np.asarray(data.origin) * factor
        return out
    if not isinstance(field, str):
        raise TypeError(f"convert takes a field's name, not {field!r}")
    if field not in data.fields:
        raise KeyError(f"no field {field!r}: the dataset has "
                       f"{', '.join(map(repr, data.fields)) or 'none'}")
    fields = dict(data.fields)
    fields[field] = _converted(data.fields[field], target, f"field {field!r}")
    from tack.dataset.ghosts import keep_marks

    return keep_marks(data, DataSet(data.topology, data.geometry, fields=fields,
                                    sets=dict(data.sets)))

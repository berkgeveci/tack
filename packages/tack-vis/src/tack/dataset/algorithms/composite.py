"""``composite``: scalar fields of one space as the components of one vector field (Viskores' CompositeVectors), in the order given."""

import tack
from tack import arrays
from tack.arrays import width_of
from tack.dataset import units
from tack.dataset.algorithms._common import _field, _same_dtype, _same_space, _vectors
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['composite']


@tack.kernel
def _set_component(values, component, out):
    for i in range(len(values)):
        out[i][component] = values[i]


@on_inputs_device
def composite(data, *fields):
    """Scalar fields of one space as the components of one vector field (Viskores'
    CompositeVectors), in the order given. The fields must be in one unit, the
    vector's (an unknown one is not checked)."""
    fields = [_field(data, f) for f in fields]
    if len(fields) < 2:
        raise ValueError("composite takes at least two scalar fields")
    space, n = _same_space(*fields)
    if any(width_of(f.values) for f in fields):
        raise TypeError("composite takes scalar fields")
    _same_dtype("composite", *fields)
    unit = units.common("composite's fields", [f.unit for f in fields])
    out = _vectors(n, arrays.dtype_of(fields[0].values), len(fields))
    for k, f in enumerate(fields):
        if n:
            _set_component(f.values, k, out)
    return Field(space, out, unit)

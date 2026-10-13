"""``composite``: scalar fields of one space as the components of one vector field (Viskores' CompositeVectors), in the order given."""

import tack
from tack.data import arrays
from tack.data.algorithms._common import _field, _same_dtype, _same_space, _vectors
from tack.data.arrays import materialize, width_of
from tack.data.dataset import Field

__all__ = ['composite']


@tack.kernel
def _set_component(values, component, out):
    for i in range(values.shape[0]):
        out[i][component] = values[i]


def composite(data, *fields):
    """Scalar fields of one space as the components of one vector field (Viskores'
    CompositeVectors), in the order given."""
    fields = [_field(data, f) for f in fields]
    if len(fields) < 2:
        raise ValueError("composite takes at least two scalar fields")
    space, n = _same_space(*fields)
    if any(width_of(f.values) for f in fields):
        raise TypeError("composite takes scalar fields")
    _same_dtype("composite", *fields)
    out = _vectors(n, arrays.dtype_of(fields[0].values), len(fields))
    for k, f in enumerate(fields):
        if n:
            _set_component(materialize(f.values), k, out)
    return Field(space, out)

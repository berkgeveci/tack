"""``magnitude``: each vector's length (Viskores' VectorMagnitude)."""

import tack
from tack.arrays import size_of, width_of
from tack.dataset.algorithms._common import _converted, _field, _floating_dtype, _scalars
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['magnitude']


@tack.kernel
def _magnitudes(values, out):
    for i in range(out.shape[0]):
        out[i] = values[i].norm()


@on_inputs_device
def magnitude(data, field):
    """Each vector's length (Viskores' VectorMagnitude)."""
    field = _field(data, field)
    if not width_of(field.values):
        raise TypeError("magnitude takes a vector field")
    n = size_of(field.values)
    dtype = _floating_dtype(field.values)
    out = _scalars(n, dtype)
    if n:
        _magnitudes(_converted(field.values, dtype), out)
    return Field(field.space, out, field.unit)

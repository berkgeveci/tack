"""``magnitude``: each vector's length (Viskores' VectorMagnitude)."""

import tack
from tack.data import arrays
from tack.data.algorithms._common import _scalars
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import Field

__all__ = ['magnitude']


@tack.kernel
def _magnitudes(values, out):
    for i in range(out.shape[0]):
        out[i] = values[i].norm()


def magnitude(field):
    """Each vector's length (Viskores' VectorMagnitude)."""
    if not width_of(field.values):
        raise TypeError("magnitude takes a vector field")
    n = size_of(field.values)
    out = _scalars(n, arrays.dtype_of(field.values))
    if n:
        _magnitudes(materialize(field.values), out)
    return Field(field.space, out)

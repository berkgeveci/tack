"""``implicit_values``: an implicit function (``tack.dataset.implicit``) at the geometry's values: a field in the geometry's space, negative inside the function's region. On an ``H1`` geometry that is one value per point."""

import tack
from tack.dataset import arrays
from tack.dataset.arrays import materialize, size_of
from tack.dataset.dataset import Field

__all__ = ['implicit_values']


@tack.kernel
def _implicit_values(positions, function: tack.template(), out):
    for i in range(out.shape[0]):
        out[i] = function.value(positions[i])


def implicit_values(data, function):
    """An implicit function (``tack.dataset.implicit``) at the geometry's values: a field
    in the geometry's space, negative inside the function's region. On an ``H1``
    geometry that is one value per point."""
    positions = materialize(data.geometry.values)
    n = size_of(data.geometry.values)
    out = tack.field(arrays.dtype_of(data.geometry.values), shape=(n,))
    if n:
        _implicit_values(positions, function, out)
    return Field(data.geometry.space, out)

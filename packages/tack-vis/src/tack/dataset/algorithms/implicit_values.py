"""``implicit_values``: an implicit function (``tack.dataset.implicit``) at the geometry's values: a field in the geometry's space, negative inside the function's region. On an ``H1`` geometry that is one value per point."""

import tack
from tack import arrays
from tack.arrays import materialize, size_of
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['implicit_values']


@tack.kernel
def _implicit_values(positions, function, out):
    for i in range(out.shape[0]):
        out[i] = function.value(positions[i])


@on_inputs_device
def implicit_values(data, function):
    """An implicit function (``tack.dataset.implicit``) at the geometry's values: a field
    in the geometry's space, negative inside the function's region. On an ``H1``
    geometry that is one value per point."""
    positions = materialize(data.geometry.values)
    n = size_of(data.geometry.values)
    dtype = arrays.dtype_of(data.geometry.values)
    out = tack.empty((n,), dtype)
    # A function holding arrays (Planes) gives a copy at the data's precision and
    # on its device; the others hold Python numbers, which take it anyway.
    matching = getattr(type(function), "_matching", None)
    if matching is not None:
        function = matching(function, dtype, data.device)
    if n:
        _implicit_values(positions, function, out)
    return Field(data.geometry.space, out)

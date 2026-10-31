"""``implicit_values``: an implicit function (``tack.dataset.implicit``) at the geometry's values: a field in the geometry's space, negative inside the function's region. On an ``H1`` geometry that is one value per point."""

import tack
from tack import arrays
from tack.arrays import size_of
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['implicit_values']


@tack.kernel
def _implicit_values(positions, function, out):
    for i in range(out.shape[0]):
        out[i] = function.value(positions[i])


@on_inputs_device
def implicit_values(data, function):
    """``function``, an implicit function (``tack.dataset.implicit``: ``Plane``,
    ``Sphere``, ``Box``, ...), at the geometry's values: a field in the
    geometry's space and type, negative inside the function's region. On an
    ``H1`` geometry that is one value per point. The function is in world
    positions, taken relative to the dataset's origin in double precision."""
    positions = data.geometry.values             # an image's computed as they are read
    n = size_of(data.geometry.values)
    dtype = arrays.dtype_of(data.geometry.values)
    out = tack.empty((n,), dtype)
    # A function holding arrays (Planes) gives a copy at the data's precision and
    # on its device; the others hold Python numbers, which take it anyway.
    from tack.dataset.implicit import relative_to

    function = relative_to(function, data.origin)    # the geometry's values are relative
    matching = getattr(type(function), "_matching", None)
    if matching is not None:
        function = matching(function, dtype, data.device)
    if n:
        _implicit_values(positions, function, out)
    return Field(data.geometry.space, out)

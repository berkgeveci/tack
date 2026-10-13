"""``dot``: ``a . b`` at every value of two vector fields of one width and space."""

import tack
from tack.data import arrays
from tack.data.algorithms._common import _field, _same_dtype, _same_space, _scalars
from tack.data.arrays import materialize, width_of
from tack.data.dataset import Field

__all__ = ['dot']


@tack.kernel
def _dots(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i].dot(b[i])


def dot(data, a, b):
    """``a . b`` at every value of two vector fields of one width, dtype and space."""
    a, b = _field(data, a), _field(data, b)
    space, n = _same_space(a, b)
    if not width_of(a.values) or width_of(a.values) != width_of(b.values):
        raise TypeError("dot takes two vector fields of one width")
    _same_dtype("dot", a, b)
    out = _scalars(n, arrays.dtype_of(a.values))
    if n:
        _dots(materialize(a.values), materialize(b.values), out)
    return Field(space, out)

"""``cross``: ``a x b`` at every value of two 3-vector fields in one space."""

import tack
from tack import arrays
from tack.arrays import materialize, width_of
from tack.dataset.algorithms._common import _field, _same_dtype, _same_space, _vectors
from tack.dataset.dataset import Field

__all__ = ['cross']


@tack.kernel
def _crosses(a, b, out):
    for i in range(out.shape[0]):
        out[i] = a[i].cross(b[i])


def cross(data, a, b):
    """``a x b`` at every value of two 3-vector fields of one dtype and space."""
    a, b = _field(data, a), _field(data, b)
    space, n = _same_space(a, b)
    if width_of(a.values) != 3 or width_of(b.values) != 3:
        raise TypeError("cross takes two 3-vector fields")
    _same_dtype("cross", a, b)
    out = _vectors(n, arrays.dtype_of(a.values))
    if n:
        _crosses(materialize(a.values), materialize(b.values), out)
    return Field(space, out)

"""``log_values``: the logarithm of a scalar field in ``base`` (e, 2, 10, or any), values below ``min_value`` -- by default the smallest positive normal float, as in Viskores' LogValues -- taken as ``min_value``."""

import math

import numpy as np

import tack
from tack.data import arrays
from tack.data.algorithms._common import _scalars
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import Field

__all__ = ['log_values']


@tack.kernel
def _logs(values, scale, smallest, out):
    for i in range(out.shape[0]):
        out[i] = log(max(values[i], smallest)) * scale


def log_values(field, base=math.e, min_value=None):
    """The logarithm of a scalar field in ``base`` (e, 2, 10, or any), values below
    ``min_value`` -- by default the smallest positive normal float, as in
    Viskores' LogValues -- taken as ``min_value``."""
    if width_of(field.values):
        raise TypeError("log_values takes a scalar field")
    if not base > 0 or base == 1:
        raise ValueError(f"no logarithm in base {base}")
    dtype = arrays.dtype_of(field.values)
    if min_value is None:
        min_value = float(np.finfo(dtype.numpy_dtype).tiny)
    n = size_of(field.values)
    out = _scalars(n, dtype)
    if n:
        _logs(materialize(field.values), 1.0 / math.log(base), float(min_value), out)
    return Field(field.space, out)

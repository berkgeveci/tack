"""``log_values``: the logarithm of a scalar field in ``base`` (e, 2, 10, or any), values below ``min_value`` -- by default the smallest positive normal float, as in Viskores' LogValues -- taken as ``min_value``."""

import math

import numpy as np

import tack
from tack.arrays import size_of, width_of
from tack.dataset import units
from tack.dataset.algorithms._common import _converted, _field, _floating_dtype, _scalars
from tack.dataset.dataset import Field
from tack.runtime.dispatch import on_inputs_device

__all__ = ['log_values']


@tack.kernel
def _logs(values, scale, smallest, out):
    for i in range(out.shape[0]):
        out[i] = log(max(values[i], smallest)) * scale


@on_inputs_device
def log_values(data, field, base=math.e, min_value=None):
    """The logarithm of a scalar field in ``base`` (e, 2, 10, or any), values below
    ``min_value`` -- by default the smallest positive normal float, as in
    Viskores' LogValues -- taken as ``min_value``. The field must be a pure number
    (``"dimensionless"``) or of unknown unit: a logarithm of metres is refused."""
    field = _field(data, field)
    if width_of(field.values):
        raise TypeError("log_values takes a scalar field")
    unit = units.dimensionless_or_none("log_values", field.unit)
    if not base > 0 or base == 1:
        raise ValueError(f"no logarithm in base {base}")
    dtype = _floating_dtype(field.values)
    if min_value is None:
        min_value = float(np.finfo(dtype.numpy_dtype).tiny)
    n = size_of(field.values)
    out = _scalars(n, dtype)
    if n:
        _logs(_converted(field.values, dtype), 1.0 / math.log(base), float(min_value), out)
    return Field(field.space, out, unit)

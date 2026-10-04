"""Host reductions and identities shared by Field and device fallbacks."""

import numpy as np

REDUCTION_IDENTITIES = {'sum': 0.0, 'min': float('inf'), 'max': -float('inf')}


def empty_reduction(op):
    if op == 'sum':
        return 0.0
    if op in ('min', 'max'):
        raise ValueError(f'{op} reduction requires a nonempty field')
    raise ValueError(f'Unknown reduction: {op}')


def reduce_numpy(values, op):
    """Return a Python float, with portable extrema NaN and zero rules."""
    if op not in REDUCTION_IDENTITIES:
        raise ValueError(f'Unknown reduction: {op}')
    values = np.asarray(values)
    if values.size == 0:
        return empty_reduction(op)
    if op == 'sum':
        # Promote integer accumulators independently of the host word size.
        dtype = values.dtype
        if dtype.kind in 'iu':
            dtype = np.dtype('uint64' if dtype.kind == 'u' else 'int64')
        with np.errstate(over='ignore', invalid='ignore'):
            return float(np.sum(values, dtype=dtype))
    result = float(getattr(values, op)())
    if values.dtype.kind == 'f' and result == 0.0:
        negative = np.signbit(values[values == 0.0])
        use_negative = negative.any() if op == 'min' else negative.all()
        return -0.0 if use_negative else 0.0
    return result

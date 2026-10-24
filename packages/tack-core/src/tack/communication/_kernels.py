"""Element moves the communication algorithms share: whole elements of any type,
through each array's flat scalar view, on the array's device."""

import numpy as np

import tack


@tack.kernel
def _gather(source, indices, out, n):
    for t in range(out.shape[0]):
        out[t] = source[tack.i64(indices[t // n]) * n + t % n]


@tack.kernel
def _scatter(values, indices, out, n):
    for t in range(values.shape[0]):
        out[tack.i64(indices[t // n]) * n + t % n] = values[t]


@tack.kernel
def _take(source, out, start):
    for t in range(out.shape[0]):
        out[t] = source[start + t]


@tack.kernel
def _search_right(keys, splitters, positions, n):
    # For each splitter, how many keys are at most it (keys sorted ascending).
    for s in range(splitters.shape[0]):
        lo = tack.i64(0)
        hi = tack.i64(n)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] <= splitters[s]:
                lo = mid + 1
            else:
                hi = mid
        positions[s] = lo


def width(array):
    """Scalars per element."""
    return array.scalar_size // array.size if array.size else 1


def gather(array, indices):
    """``array``'s elements at ``indices`` (an integer array on its device): a new
    array."""
    out = tack.empty(indices.size, array.dtype, device=array.device)
    if indices.size:
        _gather(array._scalar_view(), indices, out._scalar_view(), width(array))
    return out


def scatter(values, indices, out):
    """``out``'s elements at ``indices`` set to ``values``' elements, in place."""
    if indices.size:
        _scatter(values._scalar_view(), indices, out._scalar_view(), width(out))


def take(array, start, count):
    """Elements ``start`` to ``start + count`` of a one-dimensional array: a new array."""
    out = tack.empty(count, array.dtype, device=array.device)
    if count:
        n = width(array)
        _take(array._scalar_view(), out._scalar_view(), start * n)
    return out


def search_right(keys, splitters):
    """For each of ``splitters`` (host values), how many of the sorted ``keys`` are at
    most it: a NumPy array."""
    if not len(splitters) or not keys.size:
        return np.zeros(len(splitters), np.int64)
    bounds = tack.array(np.asarray(splitters, keys.scalar_dtype.numpy_dtype), keys.dtype,
                        device=keys.device)
    positions = tack.empty(len(splitters), tack.i64, device=keys.device)
    _search_right(keys, bounds, positions, keys.size)
    return positions.numpy()

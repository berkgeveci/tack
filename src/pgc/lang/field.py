"""PGC Field — n-dimensional arrays living on a device.

Fields are created by ``pgc.field()`` and are bound to the currently active
backend.  Data transfer between host (numpy) and device is explicit:

    x.from_numpy(np_array)   # host → device
    result = x.to_numpy()    # device → host

On CPU the "device" is just numpy.  On Metal (Apple Silicon unified memory)
transfers are zero-copy.  On CUDA transfers go over PCIe.
"""

import numpy as np

from pgc.lang.types import ScalarType, f32, from_numpy_dtype


class DeviceBuffer:
    """Abstract interface for backend-specific field storage."""

    def from_numpy(self, arr: np.ndarray):
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    def fill(self, value):
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        raise NotImplementedError


class NumpyBuffer(DeviceBuffer):
    """CPU backend buffer — just a numpy array."""

    def __init__(self, numpy_dtype, shape):
        self._data = np.zeros(shape, dtype=numpy_dtype)

    def from_numpy(self, arr: np.ndarray):
        np.copyto(self._data, arr)

    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    def fill(self, value):
        self._data.fill(value)

    @property
    def nbytes(self) -> int:
        return self._data.nbytes


class Field:
    """An n-dimensional array bound to a specific device backend."""

    def __init__(self, dtype: ScalarType, shape: tuple[int, ...], buffer: DeviceBuffer):
        self.dtype = dtype
        self.shape = shape
        self._buffer = buffer

    def from_numpy(self, arr: np.ndarray):
        """Copy data from a numpy array to the device."""
        expected = self.dtype.numpy_dtype
        if arr.dtype != expected:
            arr = arr.astype(expected)
        if arr.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: field is {self.shape}, got {arr.shape}"
            )
        self._buffer.from_numpy(arr)

    def to_numpy(self) -> np.ndarray:
        """Copy data from the device to a new numpy array."""
        return self._buffer.to_numpy()

    def fill(self, value):
        """Fill the field with a scalar value."""
        self._buffer.fill(value)

    def __repr__(self):
        return f"Field(dtype={self.dtype}, shape={self.shape})"


def field(dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
    """Create a new field on the currently active backend."""
    from pgc.runtime.dispatch import get_backend

    if isinstance(shape, int):
        shape = (shape,)
    backend = get_backend()
    buf = backend.allocate_field(dtype, shape)
    return Field(dtype, shape, buf)

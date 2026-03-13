"""PGC Field — n-dimensional arrays living on a device."""

import numpy as np

from pgc.lang.types import ScalarType, f32, from_numpy_dtype


class Field:
    """An n-dimensional array backed by numpy (CPU) or device memory."""

    def __init__(self, dtype: ScalarType, shape: tuple[int, ...]):
        self.dtype = dtype
        self.shape = shape
        self._data = np.zeros(shape, dtype=dtype.numpy_dtype)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def from_numpy(self, arr: np.ndarray):
        """Copy data from a numpy array into this field."""
        expected = self.dtype.numpy_dtype
        if arr.dtype != expected:
            arr = arr.astype(expected)
        if arr.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: field is {self.shape}, got {arr.shape}"
            )
        np.copyto(self._data, arr)

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the field data as a numpy array."""
        return self._data.copy()

    def fill(self, value):
        """Fill the field with a scalar value."""
        self._data.fill(value)

    def __repr__(self):
        return f"Field(dtype={self.dtype}, shape={self.shape})"


def field(dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
    """Create a new field (n-dimensional array on device)."""
    if isinstance(shape, int):
        shape = (shape,)
    return Field(dtype, shape)

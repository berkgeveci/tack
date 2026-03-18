"""PGC Field — n-dimensional arrays living on a device.

Fields are created by ``pgc.field()`` and are bound to the currently active
backend.  Data transfer between host (numpy) and device is explicit:

    x.from_numpy(np_array)   # host → device
    result = x.to_numpy()    # device → host

On CPU the "device" is just numpy.  On Metal (Apple Silicon unified memory)
transfers are zero-copy.  On CUDA transfers go over PCIe.
"""

import numpy as np

from pgc.lang.types import ScalarType, f32, f64, from_numpy_dtype


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

    def sum(self):
        """Return the sum of all elements."""
        from pgc.runtime.dispatch import get_backend
        backend = get_backend()
        if hasattr(backend, 'reduce_field'):
            return backend.reduce_field(self, 'sum')
        return float(self._buffer.to_numpy().sum())

    def min(self):
        """Return the minimum element."""
        from pgc.runtime.dispatch import get_backend
        backend = get_backend()
        if hasattr(backend, 'reduce_field'):
            return backend.reduce_field(self, 'min')
        return float(self._buffer.to_numpy().min())

    def max(self):
        """Return the maximum element."""
        from pgc.runtime.dispatch import get_backend
        backend = get_backend()
        if hasattr(backend, 'reduce_field'):
            return backend.reduce_field(self, 'max')
        return float(self._buffer.to_numpy().max())

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


def field_like(arr: np.ndarray, dtype: ScalarType = None) -> Field:
    """Create a field from a numpy array, inferring shape and dtype.

    Allocates the field and copies the data in one step.

    Args:
        arr: numpy array to copy
        dtype: override dtype (default: inferred from arr.dtype)

    Returns:
        A new Field with the data copied to the device.
    """
    from pgc.runtime.dispatch import get_backend

    if dtype is None:
        dtype = from_numpy_dtype(arr.dtype)
    shape = arr.shape
    backend = get_backend()
    buf = backend.allocate_field(dtype, shape)
    f = Field(dtype, shape, buf)
    f.from_numpy(arr)
    return f


class Vector:
    """Vector type for creating vector fields.

    Usage:
        # Create a vector field (3-component vectors, n elements)
        v = pgc.Vector.field(3, dtype=pgc.f32, shape=(n,))

        # In kernels, vectors are scalarized:
        # v[i] loads 3 components from v[i*3], v[i*3+1], v[i*3+2]
    """

    @staticmethod
    def field(n: int, dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
        """Create a vector field with n components per element.

        The underlying storage is a flat scalar field of size
        prod(shape) * n elements.  In kernels, field[i] accesses
        components at i*n, i*n+1, ..., i*n+(n-1).
        """
        from pgc.runtime.dispatch import get_backend

        if isinstance(shape, int):
            shape = (shape,)

        # Flatten: total elements = prod(shape) * n
        total = 1
        for s in shape:
            total *= s
        flat_shape = (total * n,)

        backend = get_backend()
        buf = backend.allocate_field(dtype, flat_shape)
        f = Field(dtype, flat_shape, buf)
        # Mark as vector field so kernel dispatch can handle it
        f._vector_n = n
        f._logical_shape = shape
        return f


class Texture3D:
    """A 3D texture wrapping a Field, enabling hardware-accelerated sampling.

    Created via ``pgc.texture3d(field, interp='linear')``.  In kernels,
    ``tex.sample(u, v, w)`` samples at normalized [0,1] coordinates using
    trilinear interpolation.

    On GPU backends this maps to native texture hardware; on CPU it emits
    software trilinear interpolation against the raw field data.
    """

    def __init__(self, source_field: Field, shape_3d: tuple, interp: str = 'linear'):
        if source_field.dtype not in (f32, f64):
            raise ValueError("texture3d requires f32 or f64 dtype")
        self.field = source_field
        self.shape_3d = shape_3d   # (W, H, D) logical 3D shape
        self.interp = interp       # 'linear' or 'nearest'

    @property
    def dtype(self):
        return self.field.dtype

    def sample(self, u, v, w):
        """Sample at normalized coordinates. Only usable inside @pgc.kernel."""
        raise RuntimeError("Texture3D.sample() can only be used inside a @pgc.kernel")

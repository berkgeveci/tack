"""PGC Field — n-dimensional arrays living on a device.

Fields are created by ``pgc.field()`` and are bound to the currently active
backend.  Data transfer between host (numpy) and device is explicit:

    x.from_numpy(np_array)   # host → device
    result = x.to_numpy()    # device → host

On CPU the "device" is just numpy.  On Metal (Apple Silicon unified memory)
transfers are zero-copy.  On CUDA transfers go over PCIe.
"""

from dataclasses import dataclass
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


@dataclass
class ExportedMemory:
    """Handle for sharing GPU memory across APIs (e.g. PGC → Dawn/Vulkan).

    Contains only plain Python values — no dependency on any GPU library.
    The consumer dispatches on ``backend`` to choose the right import path
    (e.g. MTLBuffer pointer for Metal, POSIX fd for CUDA/Vulkan).
    """
    backend: str            # "metal", "cuda", "hip", "level_zero"
    size: int               # usable data size in bytes
    allocation_size: int    # actual allocation size (may be rounded up)
    handle: object          # backend-specific: MTLBuffer ptr (int), fd (int), etc.
    device_uuid: bytes | None = None  # GPU device UUID (CUDA/HIP/L0)


class Field:
    """An n-dimensional array bound to a specific device backend."""

    def __init__(self, dtype: ScalarType, shape: tuple[int, ...], buffer: DeviceBuffer,
                 writable: bool = True):
        self.dtype = dtype
        self.shape = shape
        self._buffer = buffer
        self._writable = writable

    def _check_writable(self):
        if not self._writable:
            raise RuntimeError(
                "Field is read-only (created from an external pointer). "
                "Use writable=True in field_from_ptr() to enable writes.")

    def from_numpy(self, arr: np.ndarray):
        """Copy data from a numpy array to the device."""
        self._check_writable()
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
        self._check_writable()
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

    def export_memory(self) -> ExportedMemory:
        """Export the field's GPU memory for cross-API sharing.

        Returns an ExportedMemory with backend tag and a handle suitable for
        importing into another API (e.g. Dawn, Vulkan, pycuda).
        If the field was not allocated with ``exportable=True``, a one-time
        copy into exportable memory is performed automatically.
        """
        if not hasattr(self._buffer, 'export_memory'):
            raise RuntimeError(
                f"Backend {type(self._buffer).__name__} does not support memory export")
        return self._buffer.export_memory()

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Export this field as a DLPack capsule for zero-copy interop.

        Usage:
            torch_tensor = torch.from_dlpack(pgc_field)
            cupy_array = cupy.from_dlpack(pgc_field)
            np_array = np.from_dlpack(pgc_field)  # numpy 1.25+
        """
        from pgc.lang.dlpack import field_to_dlpack
        if copy is True:
            raise BufferError("PGC DLPack export does not support copy=True")
        return field_to_dlpack(self)

    def __dlpack_device__(self):
        """Return (device_type, device_id) for the DLPack protocol."""
        from pgc.lang.dlpack import dlpack_device
        return dlpack_device(self)

    def __repr__(self):
        return f"Field(dtype={self.dtype}, shape={self.shape})"


def field(dtype: ScalarType = f32, shape: tuple[int, ...] = (),
          exportable: bool = False) -> Field:
    """Create a new field on the currently active backend.

    Args:
        dtype: scalar element type (default: f32)
        shape: dimensions of the field
        exportable: if True, allocate with cross-API export capability
                    (e.g. CUDA VMM with POSIX FD handles). Enables zero-copy
                    ``field.export_memory()`` without a re-allocation.
    """
    from pgc.runtime.dispatch import get_backend

    if isinstance(shape, int):
        shape = (shape,)
    backend = get_backend()
    buf = backend.allocate_field(dtype, shape, exportable=exportable)
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


def from_dlpack(capsule) -> Field:
    """Create a PGC field from a DLPack capsule or any object with __dlpack__.

    Zero-copy when the source is on CPU. Copies for GPU sources that
    don't match the current backend.

    Usage:
        field = pgc.from_dlpack(torch_tensor)
        field = pgc.from_dlpack(cupy_array)
        field = pgc.from_dlpack(numpy_array)
    """
    # If it has __dlpack__, call it to get the capsule
    if hasattr(capsule, '__dlpack__'):
        arr = np.from_dlpack(capsule)
    elif isinstance(capsule, np.ndarray):
        arr = capsule
    else:
        arr = np.from_dlpack(capsule)

    return field_like(arr)


def field_from_ptr(ptr, dtype: ScalarType, shape: tuple[int, ...],
                   writable: bool = False) -> Field:
    """Wrap an existing device pointer as a PGC field without copying.

    Use this for interop with external libraries (pycuda, cupy, Catalyst)
    or when receiving a device pointer from a simulation framework.

    The field does NOT own the memory — PGC will not free it.
    Read-only by default; pass writable=True to enable writes.

    Args:
        ptr: device pointer (integer) or backend-specific buffer object.
             - CPU: integer address or numpy array
             - Metal: MTLBuffer object (from PyObjC)
             - CUDA/HIP: device pointer as integer
             - Level Zero: device pointer as integer
        dtype: scalar type (pgc.f32, pgc.i32, etc.)
        shape: tuple of dimensions
        writable: if False (default), from_numpy() and fill() raise errors

    Returns:
        A Field wrapping the external memory.
    """
    from pgc.runtime.dispatch import get_backend

    if isinstance(shape, int):
        shape = (shape,)
    backend = get_backend()
    buf = backend.wrap_ptr(ptr, dtype, shape)
    return Field(dtype, shape, buf, writable=writable)


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

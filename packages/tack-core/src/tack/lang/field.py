"""Tack Field — n-dimensional arrays living on a device.

Fields are created by ``tack.field()`` and are bound to the currently active
backend.  Data transfer between host (numpy) and device is explicit:

    x.from_numpy(np_array)   # host → device
    result = x.to_numpy()    # device → host

On CPU the "device" is just numpy.  On Metal (Apple Silicon unified memory)
transfers are zero-copy.  On CUDA transfers go over PCIe.
"""

from dataclasses import dataclass
import numpy as np

from tack.lang.types import ScalarType, f32, f64, i32, from_numpy_dtype


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
    """Handle for sharing GPU memory across APIs (e.g. Tack → Dawn/Vulkan).

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

    def _reduce(self, op: str):
        """Reduce on the device where that is supported, else via numpy."""
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        if backend.supports_device_reductions:
            return backend.reduce_field(self, op)
        return float(getattr(self._buffer.to_numpy(), op)())

    def sum(self):
        """Return the sum of all elements."""
        return self._reduce('sum')

    def min(self):
        """Return the minimum element."""
        return self._reduce('min')

    def max(self):
        """Return the maximum element."""
        return self._reduce('max')

    def mean(self):
        """Return the mean of all elements (GPU sum / size)."""
        return self.sum() / self.size

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
            torch_tensor = torch.from_dlpack(tack_field)
            cupy_array = cupy.from_dlpack(tack_field)
            np_array = np.from_dlpack(tack_field)  # numpy 1.25+
        """
        from tack.lang.dlpack import field_to_dlpack
        if copy is True:
            raise BufferError("Tack DLPack export does not support copy=True")
        return field_to_dlpack(self)

    def __dlpack_device__(self):
        """Return (device_type, device_id) for the DLPack protocol."""
        from tack.lang.dlpack import dlpack_device
        return dlpack_device(self)

    @property
    def size(self) -> int:
        """Total number of elements in the field."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    def __len__(self) -> int:
        """Number of elements along the first dimension."""
        return self.shape[0] if self.shape else 0

    def copy(self) -> 'Field':
        """Return a new field with a copy of this field's data (GPU kernel, no host roundtrip)."""
        from tack.runtime.dispatch import get_backend
        from tack.algorithms.copy import copy as _copy
        backend = get_backend()
        buf = backend.allocate_field(self.dtype, self.shape)
        new_field = Field(self.dtype, self.shape, buf)
        _copy(self, new_field, self.size)
        return new_field

    def astype(self, new_dtype: ScalarType) -> 'Field':
        """Return a new field with data converted to a different dtype (GPU kernel, no host roundtrip)."""
        if new_dtype is self.dtype:
            return self.copy()
        from tack.runtime.dispatch import get_backend
        from tack.algorithms.copy import copy as _copy
        backend = get_backend()
        buf = backend.allocate_field(new_dtype, self.shape)
        new_field = Field(new_dtype, self.shape, buf)
        # The copy kernel handles cross-dtype conversion naturally:
        # dst[i] = src[i] where dst and src have different dtypes
        _copy(self, new_field, self.size)
        return new_field

    def reshape(self, new_shape: tuple[int, ...]) -> 'Field':
        """Return a new field with the same data but a different shape.

        The total number of elements must match. This is a metadata-only
        operation — the underlying buffer is shared (no copy).
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        new_size = 1
        for s in new_shape:
            new_size *= s
        if new_size != self.size:
            raise ValueError(
                f"Cannot reshape {self.shape} ({self.size} elements) "
                f"to {new_shape} ({new_size} elements)")
        return Field(self.dtype, new_shape, self._buffer, self._writable)

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
    from tack.runtime.dispatch import get_backend

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
    from tack.runtime.dispatch import get_backend

    if dtype is None:
        dtype = from_numpy_dtype(arr.dtype)
    shape = arr.shape
    backend = get_backend()
    buf = backend.allocate_field(dtype, shape)
    f = Field(dtype, shape, buf)
    f.from_numpy(arr)
    return f


def zeros(dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
    """Create a field filled with zeros."""
    f = field(dtype=dtype, shape=shape)
    f.fill(0)
    return f


def ones(dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
    """Create a field filled with ones."""
    f = field(dtype=dtype, shape=shape)
    f.fill(1)
    return f


def full(dtype: ScalarType, shape: tuple[int, ...], value) -> Field:
    """Create a field filled with a constant value."""
    f = field(dtype=dtype, shape=shape)
    f.fill(value)
    return f


def arange(n: int, dtype: ScalarType = i32) -> Field:
    """Create a field with values [0, 1, 2, ..., n-1]."""
    arr = np.arange(n, dtype=dtype.numpy_dtype)
    return field_like(arr, dtype=dtype)


def concat(fields_list: list[Field]) -> Field:
    """Concatenate a list of 1D fields into a single field (GPU kernel, no host roundtrip).

    All fields must have the same dtype. Returns a new field with
    shape (total_elements,).
    """
    if not fields_list:
        raise ValueError("concat requires at least one field")
    dt = fields_list[0].dtype
    for f in fields_list[1:]:
        if f.dtype is not dt:
            raise TypeError(
                f"concat: all fields must have the same dtype, "
                f"got {dt} and {f.dtype}")
    total = sum(f.size for f in fields_list)
    result = field(dtype=dt, shape=(total,))
    from tack.algorithms.copy import copy_with_offset
    offset = 0
    for f in fields_list:
        n = f.size
        copy_with_offset(f, result, offset, n)
        offset += n
    return result


def from_dlpack(capsule) -> Field:
    """Create a Tack field from a DLPack capsule or any object with __dlpack__.

    Zero-copy when the source is on CPU. Copies for GPU sources that
    don't match the current backend.

    Usage:
        field = tack.from_dlpack(torch_tensor)
        field = tack.from_dlpack(cupy_array)
        field = tack.from_dlpack(numpy_array)
    """
    # If it has __dlpack__, call it to get the capsule
    if hasattr(capsule, '__dlpack__'):
        arr = np.from_dlpack(capsule)
    elif isinstance(capsule, np.ndarray):
        arr = capsule
    else:
        arr = np.from_dlpack(capsule)

    return field_like(arr)


def memory_space(ptr) -> str:
    """Query the memory space of a pointer.

    Uses the active backend to determine where the pointer resides.

    Returns:
        'cpu'          — unregistered host memory (regular malloc/new)
        'cuda'         — CUDA device memory (cudaMalloc)
        'cuda_pinned'  — CUDA pinned host memory (cudaMallocHost)
        'cuda_managed' — CUDA unified memory (cudaMallocManaged)
        'hip'          — HIP device memory (hipMalloc)
        'hip_pinned'   — HIP pinned host memory (hipHostMalloc)
        'hip_managed'  — HIP unified memory (hipMallocManaged)
    """
    from tack.runtime.dispatch import get_backend
    return get_backend().memory_space(ptr)


def field_from_ptr(ptr, dtype: ScalarType, shape: tuple[int, ...],
                   writable: bool = False) -> Field:
    """Wrap an existing device pointer as a Tack field without copying.

    Use this for interop with external libraries (pycuda, cupy, Catalyst)
    or when receiving a device pointer from a simulation framework.

    The field does NOT own the memory — Tack will not free it.
    Read-only by default; pass writable=True to enable writes.

    Raises ValueError if the pointer's memory space does not match the
    active backend (e.g. a CPU pointer with the CUDA backend).

    Args:
        ptr: device pointer (integer) or backend-specific buffer object.
             - CPU: integer address or numpy array
             - Metal: MTLBuffer object (from PyObjC)
             - CUDA/HIP: device pointer as integer
             - Level Zero: device pointer as integer
        dtype: scalar type (tack.f32, tack.i32, etc.)
        shape: tuple of dimensions
        writable: if False (default), from_numpy() and fill() raise errors

    Returns:
        A Field wrapping the external memory.
    """
    from tack.runtime.dispatch import get_backend

    if isinstance(shape, int):
        shape = (shape,)
    backend = get_backend()

    # Validate the pointer's memory space against what this backend expects.
    # Skipped for non-integer pointers (Metal MTLBuffer objects, numpy arrays
    # on CPU) and for backends that do not distinguish device memory.
    if isinstance(ptr, int) and backend.device_memory_spaces:
        space = backend.memory_space(ptr)
        if space not in backend.device_memory_spaces:
            raise ValueError(
                f"Pointer is in '{space}' memory but the active backend "
                f"is '{backend.label}'. field_from_ptr() requires a device "
                f"pointer. Use tack.field() + field.from_numpy() to copy "
                f"host data to the device.")

    buf = backend.wrap_ptr(ptr, dtype, shape)
    return Field(dtype, shape, buf, writable=writable)


class Vector:
    """Vector type for creating vector fields.

    Usage:
        # Create a vector field (3-component vectors, n elements)
        v = tack.Vector.field(3, dtype=tack.f32, shape=(n,))

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
        from tack.runtime.dispatch import get_backend

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

    Created via ``tack.texture3d(field, interp='linear')``.  In kernels,
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
        """Sample at normalized coordinates. Only usable inside @tack.kernel."""
        raise RuntimeError("Texture3D.sample() can only be used inside a @tack.kernel")

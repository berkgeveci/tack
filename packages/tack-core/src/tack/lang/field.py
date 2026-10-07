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

try:
    from numpy.lib.array_utils import byte_bounds
except ImportError:  # NumPy 1.x exposes this in its top-level namespace.
    from numpy import byte_bounds

from tack.lang.types import ScalarType, f32, from_numpy_dtype, i32


class DeviceBuffer:
    """Abstract interface for backend-specific field storage."""

    #: `Backend.name` of whoever allocates this kind of buffer. A class
    #: attribute, so it costs nothing and survives buffers built with
    #: `__new__` (Metal's `wrap_ptr` does that). Only a diagnostic reads
    #: it — see `_fields_from_another_backend` in lang/kernel.py, which
    #: uses it to explain the AttributeError a backend switch produces.
    backend_name: str = ""

    def from_numpy(self, arr: np.ndarray):
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    def read_range(self, start: int, count: int) -> np.ndarray:
        """Elements ``[start, start + count)`` of the flat storage, as a copy.

        Serves ``field[i]`` on the host. Every shipped buffer overrides it:
        CPU and Metal slice their host-visible views, and CUDA, HIP and
        Level Zero copy just the range. This default, for a buffer that
        does not, copies the whole buffer, which is correct and slow.
        """
        return self.to_numpy().reshape(-1)[start:start + count].copy()

    def fill(self, value):
        raise NotImplementedError

    @property
    def address(self) -> int:
        """Address used by kernels, for validating imported atomic storage."""
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        raise NotImplementedError


class NumpyBuffer(DeviceBuffer):
    """CPU backend buffer — just a numpy array."""

    backend_name = "cpu"

    def __init__(self, numpy_dtype, shape):
        self._data = np.zeros(shape, dtype=numpy_dtype)

    def from_numpy(self, arr: np.ndarray):
        # A reshaped field shares this buffer under another shape; the
        # element count already matches, so copy in the buffer's own shape.
        np.copyto(self._data, arr.reshape(self._data.shape))

    def to_numpy(self) -> np.ndarray:
        return self._data.copy()

    def read_range(self, start: int, count: int) -> np.ndarray:
        return self._data.reshape(-1)[start:start + count].copy()

    def fill(self, value):
        self._data.fill(value)

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    @property
    def address(self) -> int:
        return self._data.ctypes.data

    @property
    def span(self) -> tuple[int, int]:
        """The half-open byte range ``[start, end)`` this buffer occupies.

        Read on every CPU dispatch to decide whether the field arguments
        overlap, so it is computed once per array rather than per call.
        """
        try:
            data, span = self._span
            if data is self._data:
                return span
        except AttributeError:
            pass
        span = byte_bounds(self._data)
        self._span = (self._data, span)
        return span


@dataclass
class ExportedMemory:
    """Handle for sharing GPU memory across APIs (e.g. Tack → Dawn/Vulkan).

    Contains only plain Python values — no dependency on any GPU library.
    The consumer dispatches on ``handle_type`` to choose the right import
    path. ``backend`` alone does not settle it: CUDA exports a POSIX fd on
    Linux and a Win32 KMT handle on Windows, and those are imported by
    different calls.
    """
    backend: str            # "metal", "cuda", "hip", "level_zero"
    size: int               # usable data size in bytes
    allocation_size: int    # actual allocation size (may be rounded up)
    handle: object          # backend-specific: MTLBuffer ptr (int), fd (int), etc.
    handle_type: str | None = None    # "mtl_buffer", "posix_fd", "win32_kmt"
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

    def _vector_shape(self):
        """``(*shape, n)`` for a vector field, ``(*shape, rows, columns)`` for a
        matrix field, and None for any other."""
        n = getattr(self, '_vector_n', None)
        if n is None:
            return None
        element = getattr(self, '_matrix_shape', None) or (n,)
        return (*self._logical_shape, *element)

    def from_numpy(self, arr: np.ndarray):
        """Copy data from a numpy array to the device.

        The array has the field's shape. A vector field also takes one row
        per vector, ``(*shape, n)``, besides its flat storage shape.
        """
        self._check_writable()
        expected = self.dtype.numpy_dtype
        if arr.dtype != expected:
            arr = arr.astype(expected)
        vector_shape = self._vector_shape()
        if vector_shape is not None and arr.shape == vector_shape:
            # Components are stored together, in row-major order.
            arr = arr.reshape(self.shape)
        if arr.shape != self.shape:
            accepted = f"{self.shape}" if vector_shape is None \
                else f"{vector_shape} or flat {self.shape}"
            raise ValueError(
                f"Shape mismatch: field is {accepted}, got {arr.shape}"
            )
        self._buffer.from_numpy(arr)

    def to_numpy(self, vectors: bool = False) -> np.ndarray:
        """Copy data from the device to a new numpy array.

        A vector field comes back in its flat storage shape,
        ``(prod(shape) * n,)``; with ``vectors=True`` it has one row per
        vector, ``(*shape, n)``. Other fields ignore the argument.
        """
        arr = self._buffer.to_numpy()
        # The buffer keeps the shape it was allocated with, so a reshaped
        # view has to impose its own -- otherwise field.shape and
        # to_numpy().shape disagree, while from_numpy() validates against
        # field.shape. Same data either way; only the view differs.
        if arr.shape != self.shape:
            arr = arr.reshape(self.shape)
        vector_shape = self._vector_shape() if vectors else None
        if vector_shape is not None:
            arr = arr.reshape(vector_shape)
        return arr

    def fill(self, value):
        """Fill the field with a scalar value.

        A vector or matrix field also takes one element's value, a sequence
        of its components: ``colors.fill([1.0, 1.0, 1.0])``.
        """
        self._check_writable()
        if np.ndim(value) == 0:
            self._buffer.fill(value)
            return
        vector_shape = self._vector_shape()
        element = np.asarray(value)
        if vector_shape is None:
            raise TypeError(
                f"fill() takes a scalar for a field of scalars, "
                f"got a value of shape {element.shape}")
        element_shape = vector_shape[len(self._logical_shape):]
        if element.shape != element_shape:
            raise ValueError(
                f"fill() value has shape {element.shape}; "
                f"the elements of this field have shape {element_shape}")
        self.from_numpy(np.broadcast_to(element, vector_shape))

    def _reduce(self, op: str):
        """Reduce on the device where that is supported, else via numpy."""
        from tack.runtime.dispatch import get_backend
        from tack.runtime.reductions import empty_reduction, reduce_numpy
        if self.size == 0:
            return empty_reduction(op)
        backend = get_backend()
        if backend.supports_device_reductions:
            return backend.reduce_field(self, op)
        return reduce_numpy(self.to_numpy(), op)

    def sum(self):
        """Return a float sum; parallel floating addition order may vary."""
        return self._reduce('sum')

    def min(self):
        """Return a float minimum; propagate NaNs and prefer negative zero."""
        return self._reduce('min')

    def max(self):
        """Return a float maximum; propagate NaNs and prefer positive zero."""
        return self._reduce('max')

    def mean(self):
        """Return sum / size, or NaN for an empty field."""
        if self.size == 0:
            return float('nan')
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

        A consumer passing max_version >= (1, 0) gets a versioned capsule,
        which is the only form that can carry the read-only flag. Without
        it the consumer must assume the memory is writable, so NumPy marks
        what it gets read-only and then cannot re-export it -- which breaks
        round trips through anything that hands the array back.
        """
        from tack.lang.dlpack import field_to_dlpack
        if copy is True:
            raise BufferError("Tack DLPack export does not support copy=True")
        versioned = max_version is not None and max_version[0] >= 1
        return field_to_dlpack(self, versioned=versioned)

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

    def __getitem__(self, index):
        """Read one element from the host: ``f[i]``, ``f[i, j]``, ``f[None]``.

        A field of scalars gives a Python number; a vector or matrix field
        gives a NumPy array of the element. Negative indices count from
        the end, as in Python. This is for inspection and for host-side
        decisions: on backends without host-visible memory each read is a
        transfer of that element, so many elements are read with
        ``to_numpy()``. Fields are not written this way; see ``from_numpy``
        and ``fill``.
        """
        shape = tuple(getattr(self, '_logical_shape', None) or self.shape)
        if index is None:
            if shape != ():
                raise TypeError(
                    f"field[None] reads a zero-dimensional field; this one has shape {shape}")
            indices = ()
        else:
            indices = index if isinstance(index, tuple) else (index,)
            if len(indices) != len(shape) or not all(
                    isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in indices):
                plural = "one integer index" if len(shape) == 1 else f"{len(shape)} integer indices"
                raise TypeError(
                    f"a field of shape {shape} takes {plural} on the host, not {index!r}; "
                    f"to read many elements use to_numpy()")
        flat = 0
        for i, extent in zip(indices, shape, strict=True):
            i = int(i)
            if not -extent <= i < extent:
                raise IndexError(f"index {index!r} is out of range for a field of shape {shape}")
            flat = flat * extent + (i + extent if i < 0 else i)
        vector_shape = self._vector_shape()
        if vector_shape is None:
            return self._buffer.read_range(flat, 1)[0].item()
        element_shape = vector_shape[len(shape):]
        count = int(np.prod(element_shape))
        return self._buffer.read_range(flat * count, count).reshape(element_shape)

    def __setitem__(self, index, value):
        raise TypeError(
            "fields are not written by element from the host; build the values in a NumPy "
            "array and call from_numpy(), use fill(), or write them in a kernel")

    def __iter__(self):
        raise TypeError(
            "a field is not iterated by element from the host; iterate over to_numpy()")

    def copy(self) -> 'Field':
        """Return a new field with a copy of this field's data (GPU kernel, no host roundtrip)."""
        from tack.algorithms.copy import copy as _copy
        from tack.runtime.dispatch import get_backend
        backend = get_backend()
        buf = backend.allocate_field(self.dtype, self.shape)
        new_field = Field(self.dtype, self.shape, buf)
        _copy(self, new_field, self.size)
        return new_field

    def astype(self, new_dtype: ScalarType) -> 'Field':
        """Return a new field with data converted to a different dtype (GPU kernel, no host roundtrip)."""
        if new_dtype is self.dtype:
            return self.copy()
        from tack.algorithms.copy import copy as _copy
        from tack.runtime.dispatch import get_backend
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
        reshaped = Field(self.dtype, new_shape, self._buffer, self._writable)
        # An imported field holds its DLPack source open. The reshaped view
        # points at the same memory, so it has to hold it too -- otherwise
        # dropping the original releases memory the view is still using.
        hold = getattr(self, '_dlpack_hold', None)
        if hold is not None:
            reshaped._dlpack_hold = hold
        return reshaped

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


def from_dlpack(source, copy: bool = False) -> Field:
    """Create a Tack field from a DLPack capsule or any object with __dlpack__.

    Zero-copy: the field shares memory with the source, and the source is
    held alive for as long as the field lives. Works for device tensors as
    well as host ones, provided the active backend can wrap them.

    Wrap, not address: the Metal backend reads host memory perfectly well
    but cannot turn a host pointer into the MTLBuffer a field needs, so it
    refuses host tensors and says so. Pass copy=True for an independent
    field instead.

    Usage:
        field = tack.from_dlpack(torch_tensor)
        field = tack.from_dlpack(cupy_array)
        field = tack.from_dlpack(numpy_array)
        field = tack.from_dlpack(vtk_capsule)
    """
    if copy:
        arr = source if isinstance(source, np.ndarray) else np.from_dlpack(source)
        return field_like(arr)

    from tack.lang.dlpack import dlpack_to_field
    return dlpack_to_field(source)


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
    active backend (e.g. a CPU pointer with the CUDA backend), and
    TypeError if a CUDA, HIP or Level Zero pointer is not an address.

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
    from tack.runtime.kernel_utils import as_address

    if isinstance(shape, int):
        shape = (shape,)
    backend = get_backend()

    # Validate the pointer's memory space against what this backend expects,
    # on backends that distinguish device memory; Metal MTLBuffer objects
    # and CPU numpy arrays are not addresses and need no check. Any form a
    # device pointer arrives in -- int, numpy integer, CUdeviceptr -- is
    # checked, not only a Python int.
    if backend.device_memory_spaces:
        if as_address(ptr) is None:
            raise TypeError(
                f"field_from_ptr() on the {backend.label} backend takes a "
                f"device address (an integer, or an object int() accepts), "
                f"not {type(ptr).__name__}.")
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


class Matrix:
    """Small fixed-size matrices: fields of them, and matrix values in kernels.

    Usage:
        F = tack.Matrix.field(2, 2, dtype=tack.f32, shape=(n,))

        # In kernels a matrix is scalarized, like a vector:
        #   A = tack.Matrix([[a, b], [c, d]])      rows of scalars, or of vectors
        #   I = tack.Matrix.identity(2)
        #   F[p] = (I + dt * C) @ F[p]             '@' multiplies; '*' is elementwise
        #   A[i, j], A.transpose(), A.trace(), A.determinant(), A.inverse()
    """

    @staticmethod
    def field(n: int, m: int, dtype: ScalarType = f32, shape: tuple[int, ...] = ()) -> Field:
        """Create a field of n-by-m matrices.

        The storage is that of ``Vector.field(n * m, dtype, shape)``, each
        matrix in row-major order. ``from_numpy`` takes ``(*shape, n, m)``
        or the flat array, and ``to_numpy(vectors=True)`` returns
        ``(*shape, n, m)``.
        """
        if not (1 <= n <= 4 and 1 <= m <= 4):
            raise ValueError(f"Matrix.field({n}, {m}): matrices are at most 4x4")
        f = Vector.field(n * m, dtype, shape)
        f._matrix_shape = (n, m)
        return f


class Texture3D:
    """A 3D texture over a snapshot of a Field, sampled trilinearly.

    Created via ``tack.texture3d(field, shape=(W, H, D))``.  In kernels,
    ``tex.sample(u, v, w)`` samples at normalized [0,1] coordinates using
    trilinear interpolation.

    The texture copies the field's data when it is created, and again on
    ``update()``; writes to the field in between do not reach it. That
    holds on every backend. Where the backend samples in hardware the copy
    is a texture image (a CUDA or HIP array, an MTLTexture, a Level Zero
    image); elsewhere it is a private field the generated code interpolates
    in software. Either way the texture owns it, so it is released with the
    texture and no other texture can be handed it.
    """

    def __init__(self, source_field: Field, shape_3d: tuple, interp: str = 'linear'):
        from tack.runtime.dispatch import get_backend

        # Every hardware path builds a single-channel 32-bit float image,
        # so f64 data would be reinterpreted rather than converted.
        if source_field.dtype is not f32:
            raise ValueError(
                f"texture3d requires an f32 field, got {source_field.dtype}; "
                f"convert it first with field.astype(tack.f32)")
        # The generated code interpolates linearly on every backend; nothing
        # implements another mode.
        if interp != 'linear':
            raise ValueError(
                f"texture3d supports interp='linear' only, got {interp!r}")
        shape_3d = tuple(int(s) for s in shape_3d)
        W, H, D = shape_3d
        if W * H * D != source_field.size:
            raise ValueError(
                f"texture3d shape {shape_3d} holds {W * H * D} elements, but "
                f"the field has {source_field.size} elements")
        self.field = source_field
        self.shape_3d = shape_3d   # (W, H, D) logical 3D shape
        self.interp = interp

        backend = get_backend()
        if backend.texture_in_hardware(shape_3d):
            self._storage = backend.create_texture_image(shape_3d)
        else:
            self._storage = Field(f32, (source_field.size,),
                                  backend.allocate_field(f32, (source_field.size,)))
        self.update()

    def update(self):
        """Copy the field's current data into the texture.

        Sampling after this sees the field as it is now, on every backend.
        """
        if isinstance(self._storage, Field):
            from tack.algorithms.copy import copy as _copy
            _copy(self.field.reshape((self.field.size,)), self._storage,
                  self.field.size)
        else:
            self._storage.upload(self.field)

    @property
    def dtype(self):
        return self.field.dtype

    def sample(self, u, v, w):
        """Sample at normalized coordinates. Only usable inside @tack.kernel."""
        raise RuntimeError("Texture3D.sample() can only be used inside a @tack.kernel")

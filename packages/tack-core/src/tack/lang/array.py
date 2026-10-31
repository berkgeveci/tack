"""Tack arrays — typed storage living on a device.

Arrays are made by the NumPy-style constructors (``tack.empty``,
``tack.zeros``, ``tack.array``, ...) and belong to the backend active when
they are made. Moving data between the host and the device is explicit:

    tack.copyto(x, host_values)   # host → device
    host = x.numpy()              # device → host (also np.asarray(x))

On CPU the "device" is just NumPy. On Metal (Apple Silicon unified memory)
transfers copy values into or out of backend storage. On CUDA transfers
can go over PCIe.
"""

import math
from dataclasses import dataclass

import numpy as np

try:
    from numpy.lib.array_utils import byte_bounds
except ImportError:  # NumPy 1.x exposes this in its top-level namespace.
    from numpy import byte_bounds

from tack.lang.types import (
    ElementType,
    ScalarType,
    as_dtype,
    f32,
    from_numpy_dtype,
    get_default_float,
    i32,
    i64,
    integer_type_for_value,
    mat,
    vec,
)


class DeviceBuffer:
    """Abstract interface for backend-specific array storage."""

    #: `Backend.name` of whoever allocates this kind of buffer. A class
    #: attribute, so it costs nothing and survives buffers built with
    #: `__new__` (Metal's `wrap_ptr` does that). Only a diagnostic reads
    #: it — see `_arrays_from_another_backend` in lang/kernel.py, which
    #: uses it to explain the AttributeError a backend switch produces.
    backend_name: str = ""

    #: Whether a newly allocated buffer of this kind already holds zeros, so
    #: `tack.zeros` need not fill it. Metal's does (`MetalBuffer`); filling
    #: it anyway waited for every queued launch, a round trip per `zeros`.
    starts_zeroed: bool = False

    #: How many times Tack has written the buffer: counted by every host write
    #: (`Array._check_writable`, which `fill`, `copyto` and uploads call) and by
    #: every kernel launch that may store to it (`check_writable_arrays`), so
    #: something derived from the values (a `CellLocator` from positions) can
    #: tell it is stale. An instance attribute once written; views of one
    #: buffer share it. Writes Tack does not make -- through memory it shares
    #: (DLPack, `export_memory`, `wrap_ptr`) -- are not counted.
    version: int = 0

    def from_numpy(self, arr: np.ndarray):
        raise NotImplementedError

    def to_numpy(self) -> np.ndarray:
        raise NotImplementedError

    def read_range(self, start: int, count: int) -> np.ndarray:
        """Elements ``[start, start + count)`` of the flat storage, as a copy.

        Serves ``array[i]`` on the host. Every shipped buffer overrides it:
        CPU and Metal slice their host-visible views, and CUDA, HIP and
        Level Zero copy just the range. This default, for a buffer that
        does not, copies the whole buffer, which is correct and slow.
        """
        return self.to_numpy().reshape(-1)[start:start + count].copy()

    def fill(self, value):
        raise NotImplementedError

    def fill_bytes(self, pattern: bytes) -> bool:
        """Fill the storage with ``pattern``, one element's bytes, repeated.

        Returns false when this buffer cannot, and the caller uploads the
        array instead: right for host-visible storage, which this default
        serves. CUDA, HIP and Level Zero override it to fill on the device
        (``runtime/device_fill.py``).
        """
        return False

    def synchronize(self):
        """Wait for the launches queued on this storage's device, if any
        are: a backend that queues launches overrides it."""

    #: Whether `Array.numpy_async` should read this storage in a worker
    #: thread: true where reading waits for queued device work. Storage
    #: whose reads never wait (the CPU's) is read inline.
    waits_off_thread: bool = False

    def thread_binding(self):
        """A context manager that makes this storage usable from a thread
        other than the one that made it -- CUDA's context made current --
        for `Array.numpy_async`'s worker. Nothing by default."""
        import contextlib
        return contextlib.nullcontext()

    def share(self):
        """Hand this storage outside Tack: wait for queued launches, and
        have later launches that use it complete before returning. A
        backend that queues launches overrides it."""
        self.synchronize()

    @property
    def address(self) -> int:
        """Address used by kernels, for validating imported atomic storage."""
        raise NotImplementedError

    @property
    def host_address(self) -> int:
        """The address a CPU kernel reads and writes through: only memory in the
        host's address space has one. A CPU kernel bound to another device's
        memory wrote device pointers from the host and segfaulted; this
        raises instead."""
        raise TypeError(f"a CPU kernel cannot address {self.backend_name or 'device'} memory")

    @property
    def nbytes(self) -> int:
        raise NotImplementedError


class NumpyBuffer(DeviceBuffer):
    """CPU backend buffer — just a numpy array, uninitialized when new."""

    backend_name = "cpu"

    def __init__(self, numpy_dtype, shape):
        self._data = np.empty(shape, dtype=numpy_dtype)

    def from_numpy(self, arr: np.ndarray):
        # A reshaped array shares this buffer under another shape; the
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
        """The data pointer, computed once per array: NumPy's ``.ctypes``
        builds an object per access, 1.6 µs each, and every CPU dispatch
        binds every array argument."""
        try:
            data, address = self._address
            if data is self._data:
                return address
        except AttributeError:
            pass
        address = self._data.ctypes.data
        self._address = (self._data, address)
        return address

    #: Host memory, so the address is a CPU kernel's (one property, no cost).
    host_address = address

    @property
    def span(self) -> tuple[int, int]:
        """The half-open byte range ``[start, end)`` this buffer occupies.

        Read on every CPU dispatch to decide whether the array arguments
        overlap, so it is computed once per array rather than per call.
        """
        try:
            data, span = self._span
            if data is self._data:
                return span
        except AttributeError:
            pass
        data = self._data
        if data.flags.c_contiguous:
            # `byte_bounds` costs 3 µs; a contiguous array's bounds are its
            # pointer and its size, and the filters make dozens a call.
            start = self.address
            span = (start, start + data.nbytes)
        else:
            span = byte_bounds(data)
        self._span = (data, span)
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


class Array:
    """An n-dimensional array of scalar, vector or matrix elements on a device.

    ``shape`` counts elements and ``dtype`` describes one element, so an array
    of shape ``(n,)`` and dtype ``vec3(f32)`` holds ``3 * n`` scalars and
    downloads as ``(n, 3)``.

    ``layout`` says how a vector or matrix array stores its components:
    ``"aos"`` (the default) interleaves them in one buffer, ``"soa"`` keeps
    one scalar array per component (`SoaArray`). An array of scalars says
    ``"aos"``.
    """

    #: How vector or matrix components are stored; see `SoaArray`.
    layout = "aos"

    def __init__(self, dtype: ScalarType, shape: tuple[int, ...], buffer: DeviceBuffer,
                 writable: bool = True, device=None):
        self._dtype = dtype
        self._shape = tuple(shape)
        self._buffer = buffer
        self._writable = writable
        # The backend that made the buffer. Every Tack constructor says; an
        # array built without one takes the default device when asked.
        self._device = device

    @property
    def _backend(self):
        if self._device is None:
            from tack.runtime.dispatch import get_backend
            self._device = get_backend()
        return self._device

    @property
    def device(self):
        """The `tack.Device` this array's memory belongs to; kernels given the
        array run there."""
        from tack.runtime.dispatch import as_device
        return as_device(self._backend)

    def to_device(self, device) -> 'Array':
        """This array on ``device`` (a `tack.Device` or a spec such as ``"cpu"``):
        itself if its memory is already usable there, else a copy."""
        from tack.runtime.dispatch import backend_of
        target = backend_of(device)
        if target is self._backend or target.memory_domain == self._backend.memory_domain:
            return self
        return array(self.numpy(), self.dtype, device=target)

    @property
    def scalar_dtype(self):
        """The scalar component type used by backend storage and code generation."""
        return self._dtype

    @property
    def dtype(self):
        """The type of one element, including vector or matrix dimensions."""
        n = getattr(self, '_vector_n', None)
        if n is None:
            return self._dtype
        shape = getattr(self, '_matrix_shape', None)
        return _element(self._dtype, shape or (n,))

    @property
    def shape(self):
        """Array dimensions, excluding an element's component dimensions."""
        return getattr(self, '_logical_shape', self._shape)

    @property
    def ndim(self) -> int:
        """Number of array dimensions, excluding an element's components."""
        return len(self.shape)

    @property
    def storage_shape(self):
        """The scalar storage dimensions (components are flattened)."""
        return self._shape

    @property
    def scalar_size(self):
        """Number of scalar components in storage."""
        return int(math.prod(self.storage_shape))

    @property
    def nbytes(self) -> int:
        """Bytes of storage the elements occupy."""
        return self.scalar_size * self._dtype.numpy_dtype.itemsize

    @property
    def _starts_zeroed(self) -> bool:
        """Whether new storage holds zeros already, so `zeros` need not fill it."""
        return self._buffer.starts_zeroed

    def _check_writable(self):
        """Refuse a host write to a read-only array; count one to any other (every
        host write calls this first)."""
        if not self._writable:
            raise RuntimeError(
                "Array is read-only (created from an external pointer). "
                "Use writable=True in array_from_ptr() to enable writes.")
        self._buffer.version += 1

    def _vector_shape(self):
        """``(*shape, n)`` for a vector array, ``(*shape, rows, columns)`` for a
        matrix array, and None for any other."""
        n = getattr(self, '_vector_n', None)
        if n is None:
            return None
        element = getattr(self, '_matrix_shape', None) or (n,)
        return (*self._logical_shape, *element)

    @property
    def _host_shape(self):
        """The shape of `numpy()`: array dimensions, then element dimensions."""
        return self._vector_shape() or self.shape

    def _upload(self, host: np.ndarray):
        """Copy host values of the right dtype and `_host_shape` to storage."""
        self._check_writable()
        self._buffer.from_numpy(np.ascontiguousarray(host).reshape(self.storage_shape))

    def numpy(self) -> np.ndarray:
        """Copy the values into a new NumPy array.

        Vector and matrix components become trailing NumPy dimensions: an
        array of shape ``(n,)`` and dtype ``vec3(f32)`` gives ``(n, 3)``.
        """
        arr = self._buffer.to_numpy()
        # The buffer keeps the shape it was allocated with, so a reshaped
        # view has to impose its own -- the same data either way.
        if arr.shape != self._host_shape:
            arr = arr.reshape(self._host_shape)
        return arr

    def __array__(self, dtype=None, copy=None):
        """``np.asarray(a)``: the values as a NumPy array (always a copy)."""
        if copy is False:
            raise ValueError("a Tack array cannot be viewed by NumPy without a copy")
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    async def numpy_async(self) -> np.ndarray:
        """`numpy` for ``asyncio``: ``await a.numpy_async()``.

        Where reading waits for queued launches (CUDA, HIP, Metal, Level
        Zero), the wait and the copy run in a worker thread and the event
        loop goes on; elsewhere it is `numpy`. The result holds every write
        queued before the call; one queued later by another task may or may
        not be in it.
        """
        from tack.runtime.aio import numpy_async
        return await numpy_async(self)

    def fill(self, value):
        """Set every element to ``value``.

        A vector or matrix array also takes one element's value, a sequence
        of its components: ``colors.fill([1.0, 1.0, 1.0])``.
        """
        self._check_writable()
        if np.ndim(value) == 0:
            if not self._backend.fill_array(self, value):
                self._buffer.fill(value)
            return
        vector_shape = self._vector_shape()
        element = np.asarray(value)
        if vector_shape is None:
            raise TypeError(
                f"fill() takes a scalar for an array of scalars, "
                f"got a value of shape {element.shape}")
        element_shape = vector_shape[len(self._logical_shape):]
        if element.shape != element_shape:
            raise ValueError(
                f"fill() value has shape {element.shape}; "
                f"the elements of this array have shape {element_shape}")
        from tack.runtime.device_fill import element_pattern
        if not self._buffer.fill_bytes(element_pattern(element, self.scalar_dtype.numpy_dtype)):
            self._upload(np.broadcast_to(element.astype(self.scalar_dtype.numpy_dtype),
                                         vector_shape))

    def _reduce(self, op: str):
        """Reduce on the device where that is supported, else via numpy."""
        from tack.runtime.reductions import empty_reduction, reduce_numpy
        if self.size == 0:
            return empty_reduction(op)
        backend = self._backend
        if backend.supports_device_reductions:
            return backend.reduce_array(self._scalar_view(), op)
        return reduce_numpy(self.numpy(), op)

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
        """Return the mean of scalar components, or NaN for an empty array."""
        if self.size == 0:
            return float('nan')
        return self.sum() / self.scalar_size

    def export_memory(self) -> ExportedMemory:
        """Export the array's GPU memory for cross-API sharing.

        Returns an ExportedMemory with backend tag and a handle suitable for
        importing into another API (e.g. Dawn, Vulkan, pycuda).
        If the array was not allocated with ``exportable=True``, a one-time
        copy into exportable memory is performed automatically.
        """
        if not hasattr(self._buffer, 'export_memory'):
            raise RuntimeError(
                f"Backend {type(self._buffer).__name__} does not support memory export")
        return self._buffer.export_memory()

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        """Export this array as a DLPack capsule for zero-copy interop.

        Usage:
            torch_tensor = torch.from_dlpack(tack_array)
            cupy_array = cupy.from_dlpack(tack_array)
            np_array = np.from_dlpack(tack_array)  # numpy 1.25+

        A consumer passing max_version >= (1, 0) gets a versioned capsule,
        which is the only form that can carry the read-only flag. Without
        it the consumer must assume the memory is writable, so NumPy marks
        what it gets read-only and then cannot re-export it -- which breaks
        round trips through anything that hands the array back.
        """
        from tack.lang.dlpack import array_to_dlpack
        if copy is True:
            raise BufferError("Tack DLPack export does not support copy=True")
        versioned = max_version is not None and max_version[0] >= 1
        return array_to_dlpack(self, versioned=versioned)

    def __dlpack_device__(self):
        """Return (device_type, device_id) for the DLPack protocol."""
        from tack.lang.dlpack import dlpack_device
        return dlpack_device(self)

    @property
    def size(self) -> int:
        """Total number of elements in the array."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    def __len__(self) -> int:
        """Number of elements along the first dimension."""
        if not self.shape:
            raise TypeError("len() of a zero-dimensional array")
        return self.shape[0]

    def __getitem__(self, index):
        """Read one element from the host: ``a[i]``, ``a[i, j]``, or ``a[()]`` for a
        zero-dimensional array.

        An array of scalars gives a Python number; a vector or matrix array
        gives a NumPy array of the element. Negative indices count from
        the end, as in Python. This is for inspection and for host-side
        decisions: on backends without host-visible memory each read is a
        transfer of that element, so many elements are read with
        ``numpy()``. Arrays are not written this way; see ``tack.copyto``
        and ``fill``.
        """
        shape = self.shape
        indices = index if isinstance(index, tuple) else (index,)
        if len(indices) != len(shape) or not all(
                isinstance(i, (int, np.integer)) and not isinstance(i, bool) for i in indices):
            if not shape:
                wanted = "a[()]"
            elif len(shape) == 1:
                wanted = "one integer index"
            else:
                wanted = f"{len(shape)} integer indices"
            raise TypeError(
                f"an array of shape {shape} takes {wanted} on the host, not {index!r}; "
                f"to read many elements use numpy()")
        flat = 0
        for i, extent in zip(indices, shape, strict=True):
            i = int(i)
            if not -extent <= i < extent:
                raise IndexError(f"index {index!r} is out of range for an array of shape {shape}")
            flat = flat * extent + (i + extent if i < 0 else i)
        vector_shape = self._vector_shape()
        if vector_shape is None:
            return self._buffer.read_range(flat, 1)[0].item()
        element_shape = vector_shape[len(shape):]
        count = int(math.prod(element_shape))
        return self._buffer.read_range(flat * count, count).reshape(element_shape)

    def __setitem__(self, index, value):
        raise TypeError(
            "arrays are not written by element from the host; build the values in a NumPy "
            "array and call tack.copyto(), use fill(), or write them in a kernel")

    def __iter__(self):
        raise TypeError(
            "an array is not iterated by element from the host; iterate over numpy()")

    def _scalar_view(self):
        """A flat scalar view for storage-level copy, reductions and interop."""
        out = Array(self.scalar_dtype, (self.scalar_size,), self._buffer, self._writable,
                    self._device)
        hold = getattr(self, '_dlpack_hold', None)
        if hold is not None:
            out._dlpack_hold = hold
        return out

    def copy(self) -> 'Array':
        """Copy all values into independent storage on the same device."""
        from tack.algorithms.copy import copy as _copy
        out = empty_like(self)
        _copy(self._scalar_view(), out._scalar_view(), self.scalar_size)
        return out

    def astype(self, dtype) -> 'Array':
        """Copy with a new component type or a matching vector/matrix dtype.

        A scalar dtype changes the component type while preserving the element
        dimensions. An element dtype must have those same dimensions.
        """
        dtype = as_dtype(dtype)
        element_shape = self._vector_shape()
        if isinstance(dtype, ElementType):
            if element_shape is None or dtype.shape != element_shape[len(self._logical_shape):]:
                raise TypeError("astype cannot change an element's component dimensions")
        elif element_shape is not None:
            dtype = _element(dtype, self.dtype.shape)
        out = empty(self.shape, dtype, device=self._backend)
        from tack.algorithms.copy import copy as _copy
        _copy(self._scalar_view(), out._scalar_view(), self.scalar_size)
        return out

    def reshape(self, *shape) -> 'Array':
        """Share storage under a new shape, preserving the element type.

        Takes the shape as a tuple or as separate integers, as NumPy does;
        one dimension may be ``-1``, inferred from the size.
        """
        if len(shape) == 1 and not isinstance(shape[0], (int, np.integer)):
            shape = shape[0]
        new_shape = list(_array_shape(shape, allow_infer=True))
        if new_shape.count(-1) > 1:
            raise ValueError("reshape can infer at most one dimension")
        if -1 in new_shape:
            known = int(math.prod([n for n in new_shape if n != -1]))
            if known == 0 or self.size % known:
                raise ValueError(f"cannot reshape {self.shape} ({self.size} elements) "
                                 f"to {tuple(shape)}")
            new_shape[new_shape.index(-1)] = self.size // known
        new_shape = tuple(new_shape)
        new_size = int(math.prod(new_shape))
        if new_size != self.size:
            raise ValueError(
                f"Cannot reshape {self.shape} ({self.size} elements) "
                f"to {new_shape} ({new_size} elements)")
        reshaped = _array_view(self.dtype, new_shape, self._buffer, self._writable,
                               self._device)
        hold = getattr(self, '_dlpack_hold', None)
        if hold is not None:
            reshaped._dlpack_hold = hold
        return reshaped

    def __repr__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype})"


class SoaArray(Array):
    """A vector or matrix array stored as one scalar array per component.

    Made by `soa` from arrays that exist already (from another library,
    through ``from_dlpack`` or ``array_from_ptr``), or by a constructor given
    ``layout="soa"``. It is an `Array` like any other: a kernel reads
    ``a[i]`` as a vector, stores, updates and atomically updates it, and
    only the generated code differs, reading component ``c`` of element
    ``i`` from ``a.components[c][i]`` rather than from ``i * n + c`` of one
    buffer. A kernel compiled for one layout is not used for the other.

    With no single buffer, an SOA array is not exported whole (DLPack,
    ``export_memory``); its components are.
    """

    layout = "soa"

    def __init__(self, components, element_shape):
        first = components[0]
        super().__init__(first.scalar_dtype, (len(components) * first.size,), None,
                         all(c._writable for c in components), first._backend)
        self._components = tuple(components)
        self._vector_n = len(components)
        self._logical_shape = first.shape
        if len(element_shape) == 2:
            self._matrix_shape = tuple(element_shape)

    @property
    def components(self) -> tuple:
        """The scalar arrays holding each component, in order (row-major for
        a matrix): the arrays `soa` was given, or the ones a constructor made."""
        return self._components

    @property
    def _starts_zeroed(self) -> bool:
        return all(c._starts_zeroed for c in self._components)

    def to_device(self, device) -> 'Array':
        moved = tuple(c.to_device(device) for c in self._components)
        if all(m is c for m, c in zip(moved, self._components)):
            return self
        return SoaArray(moved, self.dtype.shape)

    def _check_writable(self):
        for c in self._components:
            c._check_writable()

    def _upload(self, host: np.ndarray):
        self._check_writable()
        host = np.asarray(host).reshape((*self.shape, self._vector_n))
        for k, c in enumerate(self._components):
            c._upload(host[..., k])

    def numpy(self) -> np.ndarray:
        """Copy the values into a new NumPy array, components as the trailing
        dimensions, exactly as for an interleaved array."""
        return np.stack([c.numpy() for c in self._components], axis=-1).reshape(
            self._host_shape)

    async def numpy_async(self) -> np.ndarray:
        import asyncio
        parts = await asyncio.gather(*(c.numpy_async() for c in self._components))
        return np.stack(parts, axis=-1).reshape(self._host_shape)

    def fill(self, value):
        self._check_writable()
        if np.ndim(value) == 0:
            for c in self._components:
                c.fill(value)
            return
        element = np.asarray(value)
        if element.shape != self.dtype.shape:
            raise ValueError(
                f"fill() value has shape {element.shape}; "
                f"the elements of this array have shape {self.dtype.shape}")
        for c, v in zip(self._components, element.reshape(-1), strict=True):
            c.fill(v.item())

    def _reduce(self, op: str):
        from tack.runtime.reductions import empty_reduction, reduce_numpy
        if self.size == 0:
            return empty_reduction(op)
        return reduce_numpy([c._reduce(op) for c in self._components], op)

    def __getitem__(self, index):
        values = [c[index] for c in self._components]
        return np.array(values, dtype=self.scalar_dtype.numpy_dtype).reshape(self.dtype.shape)

    def _scalar_view(self):
        raise TypeError(
            "an SOA array has no single storage to view; its components "
            "(a.components) are separate arrays")

    def copy(self) -> 'Array':
        return SoaArray(tuple(c.copy() for c in self._components), self.dtype.shape)

    def astype(self, dtype) -> 'Array':
        dtype = as_dtype(dtype)
        if isinstance(dtype, ElementType):
            if dtype.shape != self.dtype.shape:
                raise TypeError("astype cannot change an element's component dimensions")
            dtype = dtype.scalar_dtype
        return SoaArray(tuple(c.astype(dtype) for c in self._components), self.dtype.shape)

    def reshape(self, *shape) -> 'Array':
        return SoaArray(tuple(c.reshape(*shape) for c in self._components), self.dtype.shape)

    def _no_single_buffer(self, what):
        return BufferError(
            f"an SOA array has no single buffer for {what}; export its "
            f"components (a.components) one by one, or copy it to one buffer "
            f"with tack.array(a, layout='aos')")

    def export_memory(self) -> ExportedMemory:
        raise self._no_single_buffer("export_memory()")

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        raise self._no_single_buffer("DLPack")

    def __dlpack_device__(self):
        raise self._no_single_buffer("DLPack")

    def __repr__(self):
        return f"Array(shape={self.shape}, dtype={self.dtype}, layout='soa')"


def soa(*components, dtype=None) -> SoaArray:
    """A vector array whose components are the given arrays, without copying.

    ``tack.soa(x, y, z)`` is an array of ``vec3`` elements whose element
    ``i`` is ``(x[i], y[i], z[i])``: the structure-of-arrays layout, for
    data that arrives that way, such as a simulation's coordinate arrays
    or another library's tensors (``from_dlpack``, ``array_from_ptr``).
    Kernels read and write it as they do any vector array, and writes go
    to the component arrays. The components are arrays of scalars of one
    dtype and one shape, on one device. An element ``dtype`` of the same
    component type makes a matrix array instead, components row-major:
    ``tack.soa(a, b, c, d, dtype=tack.mat2())``.
    """
    if len(components) == 1 and isinstance(components[0], (list, tuple)):
        components = tuple(components[0])
    if not components:
        raise TypeError("soa() takes the component arrays, at least one")
    for c in components:
        if not isinstance(c, Array):
            raise TypeError(f"soa() takes Tack arrays, not {type(c).__name__}; make one "
                            f"with tack.array(), tack.from_dlpack() or tack.array_from_ptr()")
        if hasattr(c, '_vector_n'):
            raise TypeError(f"soa() takes arrays of scalars, one per component, "
                            f"not an array of {c.dtype}")
    first = components[0]
    for c in components[1:]:
        if c.scalar_dtype is not first.scalar_dtype or c.shape != first.shape:
            raise ValueError(
                f"soa() components must share a dtype and a shape; got {first.dtype} "
                f"{first.shape} and {c.dtype} {c.shape}")
    from tack.runtime.dispatch import same_device
    same_device("soa()", components)
    if dtype is None:
        element = vec(len(components), first.scalar_dtype)
    else:
        element = as_dtype(dtype)
        if not isinstance(element, ElementType) or element.size != len(components):
            raise TypeError(f"soa() was given {len(components)} components; dtype must be "
                            f"a vector or matrix of {len(components)}, not {element}")
        if element.scalar_dtype is not first.scalar_dtype:
            raise TypeError(f"soa() components are {first.scalar_dtype}; dtype {element} "
                            f"has {element.scalar_dtype} components")
    return SoaArray(components, element.shape)


def _check_layout(layout, dtype):
    if layout == "aos":
        return False
    if layout != "soa":
        raise ValueError(f"layout is 'aos' or 'soa', not {layout!r}")
    if not isinstance(dtype, ElementType):
        raise TypeError(f"layout='soa' stores each component of a vector or matrix element "
                        f"in its own array; the dtype {dtype} has no components")
    return True



def _element(scalar, shape):
    """The (cached, so comparable with ``is``) element type of ``shape``."""
    return mat(*shape, scalar) if len(shape) == 2 else vec(shape[0], scalar)


def _array_shape(shape, allow_infer=False):
    import operator
    if isinstance(shape, (int, np.integer)):
        shape = (shape,)
    try:
        dims = tuple(operator.index(n) for n in shape)
    except TypeError as exc:
        raise TypeError("array shape must contain integer dimensions") from exc
    if any(n < 0 and not (allow_infer and n == -1) for n in dims):
        raise ValueError("array dimensions must be nonnegative")
    return dims


def _array_view(dtype, shape, buffer, writable=True, device=None):
    if not isinstance(dtype, (ScalarType, ElementType)):
        raise TypeError("dtype must be a Tack scalar, vector or matrix type")
    scalar = dtype.scalar_dtype if isinstance(dtype, ElementType) else dtype
    storage = (int(math.prod(shape)) * dtype.size,) if isinstance(dtype, ElementType) else shape
    out = Array(scalar, storage, buffer, writable, device)
    if isinstance(dtype, ElementType):
        out._vector_n = dtype.size
        out._logical_shape = shape
        if len(dtype.shape) == 2:
            out._matrix_shape = dtype.shape
    return out


def _host_dtype(host: np.ndarray, values) -> ScalarType:
    """The dtype an array made from host ``values`` gets when none is given.

    A NumPy array keeps its own. Python numbers and sequences of them, which
    NumPy would make float64 or int64, become the default float type (f32
    unless `set_default_float` changed it; f64 is not available on every
    backend), and i32 unless a value needs i64.
    """
    if isinstance(values, np.ndarray) or isinstance(values, np.generic):
        return from_numpy_dtype(host.dtype)
    if host.dtype.kind == 'f':
        return get_default_float()
    if host.dtype.kind in 'iu':
        if host.size == 0:
            return i32
        low, high = int(host.min()), int(host.max())
        return i32 if integer_type_for_value(low) is i32 and integer_type_for_value(high) is i32 \
            else from_numpy_dtype(host.dtype)
    if host.dtype.kind == 'b':
        raise TypeError("Tack has no boolean arrays; pass dtype=tack.u8 or tack.i32")
    return from_numpy_dtype(host.dtype)


def _target(device):
    """The backend ``device`` names: a `Device`, a spec, a backend, or None for
    the default device."""
    from tack.runtime.backend import Backend
    from tack.runtime.dispatch import backend_of
    return device if isinstance(device, Backend) else backend_of(device)


def empty(shape, dtype=None, *, device=None, exportable=False, layout="aos") -> Array:
    """A new array of uninitialized scalar, vector or matrix elements.

    ``shape`` counts elements; ``dtype`` describes one element, by default the
    default float type (f32 unless `set_default_float` changed it). A vector
    array of shape ``(n,)`` and dtype ``vec3(f32)`` downloads as ``(n, 3)``.
    Initialize its values with a kernel, ``fill`` or ``tack.copyto`` before
    reading. ``device`` is a `tack.Device` or a spec such as ``"cuda"``; by
    default the default device. ``layout="soa"`` stores a vector or matrix
    array's components in one array each (see `SoaArray`).
    """
    backend = _target(device)
    shape = _array_shape(shape)
    dtype = get_default_float() if dtype is None else as_dtype(dtype)
    if _check_layout(layout, dtype):
        return SoaArray(tuple(empty(shape, dtype.scalar_dtype, device=backend,
                                    exportable=exportable) for _ in range(dtype.size)),
                        dtype.shape)
    scalar = dtype.scalar_dtype if isinstance(dtype, ElementType) else dtype
    _check_dtype(backend, scalar)
    storage = (int(math.prod(shape)) * dtype.size,) if isinstance(dtype, ElementType) else shape
    buffer = _allocate(backend, scalar, storage, exportable=exportable)
    return _array_view(dtype, shape, buffer, device=backend)


def _check_dtype(backend, scalar):
    """Refuse an array of a type the device's kernels cannot take (``f64`` on
    Metal, or on a Level Zero device without it) where it is made, rather than
    at the first kernel that is given it."""
    if scalar not in backend.supported_dtypes:
        supported = ", ".join(str(t) for t in sorted(backend.supported_dtypes,
                                                     key=lambda t: t.name))
        raise TypeError(
            f"{scalar} arrays are not supported on {backend.label}, whose kernels "
            f"take {supported}. Use a supported type (f32 for f64), or keep the "
            f"array on a device that has {scalar}."
            + (f" {scalar} is the default float type: tack.set_default_float(tack.f32) "
               f"makes arrays f32 by default." if scalar is get_default_float()
               and scalar is not f32 else ""))


def _allocate(backend, dtype, shape, **options):
    """``backend.allocate_array``, collecting Python's garbage and trying once more
    if the device is out of memory (``DeviceMemoryError``; NumPy's
    ``MemoryError`` on the CPU). A dropped array in a reference cycle -- a
    topology's cached locator holds a dataset of it -- keeps its memory until
    the cycle collector runs, and that collector counts objects, not device
    memory: a 16 GB GPU ran out with dropped meshes still on it. The wait
    lets frees a backend defers (Level Zero's, a device without memory
    pools) complete before the retry, and buffers the collection returned to
    a backend's cache (HIP's, Metal's) are freed (``release_memory``)."""
    try:
        return backend.allocate_array(dtype, shape, **options)
    except MemoryError:
        import gc

        gc.collect()
        backend.synchronize()
        backend.release_memory()        # what the collection gave back to a cache
        return backend.allocate_array(dtype, shape, **options)


def _empty_for_mpi(shape, dtype, device) -> Array:
    """``empty``, in memory a GPU-aware MPI reads and writes directly
    (``Backend.allocate_mpi_array``): what ``tack.communication`` receives
    into, and copies an array it cannot hand to MPI as it is into."""
    backend = _target(device)
    shape = _array_shape(shape)
    dtype = as_dtype(dtype)
    scalar = dtype.scalar_dtype if isinstance(dtype, ElementType) else dtype
    _check_dtype(backend, scalar)
    storage = (int(math.prod(shape)) * dtype.size,) if isinstance(dtype, ElementType) else shape
    return _array_view(dtype, shape, backend.allocate_mpi_array(scalar, storage),
                       device=backend)


def zeros(shape, dtype=None, *, device=None, layout="aos") -> Array:
    """A new array of zeros, filled on the device; ``dtype`` as `empty`'s."""
    a = empty(shape, dtype, device=device, layout=layout)
    if not a._starts_zeroed:
        a.fill(0)
    return a


def ones(shape, dtype=None, *, device=None, layout="aos") -> Array:
    """A new array of ones; ``dtype`` as `empty`'s."""
    a = empty(shape, dtype, device=device, layout=layout)
    a.fill(1)
    return a


def full(shape, fill_value, dtype=None, *, device=None, layout="aos") -> Array:
    """A new array with every element ``fill_value``.

    Without ``dtype``, a Python float gives the default float type (f32 unless
    `set_default_float` changed it) and an int i32 (i64 if it needs it). A sequence fills vector or matrix elements; it needs an
    element ``dtype``.
    """
    if dtype is None:
        dtype = _host_dtype(np.asarray(fill_value), fill_value)
    a = empty(shape, dtype, device=device, layout=layout)
    a.fill(fill_value)
    return a


def _like(prototype, dtype, device, layout):
    if not isinstance(prototype, (Array, np.ndarray)):
        raise TypeError("expected a Tack array or NumPy array as the prototype")
    if dtype is None:
        dtype = prototype.dtype if isinstance(prototype, Array) \
            else from_numpy_dtype(prototype.dtype)
    else:
        dtype = as_dtype(dtype)
        if (isinstance(dtype, ScalarType) and isinstance(prototype, Array)
                and isinstance(prototype.dtype, ElementType)):
            dtype = _element(dtype, prototype.dtype.shape)
    if device is None and isinstance(prototype, Array):
        device = prototype._backend
    if layout is None:
        layout = prototype.layout if isinstance(prototype, Array) else "aos"
        if not isinstance(dtype, ElementType):
            layout = "aos"
    return empty(prototype.shape, dtype, device=device, layout=layout)


def empty_like(prototype, dtype=None, *, device=None, layout=None) -> Array:
    """A new uninitialized array of ``prototype``'s shape and dtype.

    ``prototype`` is a Tack or NumPy array; nothing is read from it. A scalar
    ``dtype`` changes the component type of a vector or matrix prototype and
    keeps its element dimensions. A Tack prototype's device and layout are
    the defaults.
    """
    return _like(prototype, dtype, device, layout)


def zeros_like(prototype, dtype=None, *, device=None, layout=None) -> Array:
    """A new array of zeros of ``prototype``'s shape and dtype."""
    a = _like(prototype, dtype, device, layout)
    if not a._starts_zeroed:
        a.fill(0)
    return a


def ones_like(prototype, dtype=None, *, device=None, layout=None) -> Array:
    """A new array of ones of ``prototype``'s shape and dtype."""
    a = _like(prototype, dtype, device, layout)
    a.fill(1)
    return a


def full_like(prototype, fill_value, dtype=None, *, device=None, layout=None) -> Array:
    """A new array of ``prototype``'s shape and dtype, every element ``fill_value``."""
    a = _like(prototype, dtype, device, layout)
    a.fill(fill_value)
    return a


def array(obj, dtype=None, *, device=None, layout=None) -> Array:
    """A new array holding a copy of ``obj``: a NumPy array, a sequence or a Tack array.

    Without ``dtype``, all of ``obj``'s dimensions are array dimensions, a
    NumPy array keeps its dtype, and Python floats and ints become the default
    float type (f32 unless `set_default_float` changed it) and i32 (i64 if a
    value needs it). With an element dtype such as
    ``vec3(f32)``, the trailing dimensions are the elements' components.
    A Tack array is copied on its device unless ``device`` names another.
    ``layout`` is ``"aos"`` or ``"soa"`` for a vector or matrix array (see
    `SoaArray`); by default a Tack array keeps its own and other data is
    interleaved.
    """
    if isinstance(obj, Array):
        if layout is not None and layout != obj.layout:
            if device is not None:
                obj = obj.to_device(device)
            out = empty(obj.shape, obj.dtype if dtype is None else dtype,
                        device=obj._backend, layout=layout)
            copyto(out, obj)
            return out
        if device is not None and _target(device).memory_domain != obj._backend.memory_domain:
            moved = obj.to_device(device)
            return moved if dtype is None else moved.astype(dtype)
        return obj.copy() if dtype is None else obj.astype(dtype)
    host = np.asarray(obj)
    dtype = _host_dtype(host, obj) if dtype is None else as_dtype(dtype)
    shape = host.shape
    if isinstance(dtype, ElementType):
        rank = len(dtype.shape)
        if len(shape) < rank or shape[-rank:] != dtype.shape:
            raise ValueError(f"input shape {shape} must end with element shape {dtype.shape}")
        shape = shape[:-rank]
    out = empty(shape, dtype, device=device, layout=layout or "aos")
    out._upload(host.astype(out.scalar_dtype.numpy_dtype, copy=False))
    return out


def asarray(obj, dtype=None, *, device=None, layout=None) -> Array:
    """``obj`` itself if it is a Tack array of ``dtype`` (or any, without one)
    on ``device`` (or any, without one) in ``layout`` (or either, without one),
    else `array(obj, dtype, device=device, layout=layout)`."""
    if isinstance(obj, Array):
        if layout is not None and layout != obj.layout:
            return array(obj, dtype, device=device, layout=layout)
        if device is not None:
            obj = obj.to_device(device)
        if dtype is None or as_dtype(dtype) == obj.dtype:
            return obj
        return obj.astype(dtype)
    return array(obj, dtype, device=device, layout=layout)


def copyto(dst: Array, src):
    """Copy ``src`` into the existing array ``dst``, as ``np.copyto`` does.

    ``src`` is host data -- a NumPy array, a sequence or a scalar -- whose
    shape is ``dst.numpy()``'s or broadcasts to it, converted to ``dst``'s
    dtype; or a Tack array of ``dst``'s shape on ``dst``'s device, copied
    there (``src.to_device(dst.device)`` brings one from another device).

    It only writes into Tack arrays. `numpy` reads one into a new NumPy
    array; ``np.copyto`` copies between NumPy arrays.
    """
    if not isinstance(dst, Array):
        raise TypeError(f"copyto() writes into a Tack array, not {type(dst).__name__}")
    if isinstance(src, Array):
        if src.shape != dst.shape or src.scalar_size != dst.scalar_size:
            raise ValueError(f"copyto() from an array of shape {src.shape} and dtype "
                             f"{src.dtype} into one of shape {dst.shape} and dtype {dst.dtype}")
        from tack.runtime.dispatch import same_device
        same_device("copyto()", (dst, src))
        dst._check_writable()
        from tack.algorithms.copy import copy as _copy
        if src.layout == dst.layout == "aos":
            _copy(src._scalar_view(), dst._scalar_view(), dst.scalar_size)
            return
        # An SOA array has no one buffer: copy element by element, which
        # reads and writes either layout.
        if getattr(src.dtype, 'shape', None) != getattr(dst.dtype, 'shape', None):
            raise ValueError(f"copyto() from an array of {src.dtype} elements into one "
                             f"of {dst.dtype} elements")
        _copy(src.reshape(src.size), dst.reshape(dst.size), dst.size)
        return
    host = np.asarray(src, dtype=dst.scalar_dtype.numpy_dtype)
    expected = dst._host_shape
    if host.shape != expected:
        try:
            host = np.broadcast_to(host, expected)
        except ValueError:
            raise ValueError(f"copyto() into an array whose values have shape {expected}: "
                             f"cannot broadcast shape {host.shape}") from None
    dst._upload(host)


def arange(start, stop=None, step=1, dtype=None, *, device=None) -> Array:
    """Evenly spaced values in ``[start, stop)``, as ``np.arange``.

    Without ``dtype``, integer arguments give i32 (i64 if a value needs it)
    and any float argument gives the default float type (`get_default_float`).
    """
    if stop is None:
        start, stop = 0, start
    values = np.arange(start, stop, step)
    if dtype is None:
        dtype = _host_dtype(values, [start, stop, step])
        if values.size == 0 and values.dtype.kind in 'iu':
            dtype = i32 if all(integer_type_for_value(int(v)) is i32 for v in (start, stop)) else i64
    return array(values, dtype, device=device)


def concat(arrays) -> Array:
    """Join a sequence of arrays end to end into a new one-dimensional array.

    All arrays must have the same dtype and be on one device, where the copy
    runs.
    """
    arrays = list(arrays)
    if not arrays:
        raise ValueError("concat requires at least one array")
    dt = arrays[0].dtype
    for a in arrays[1:]:
        if a.dtype != dt:
            raise TypeError(
                f"concat: all arrays must have the same dtype, got {dt} and {a.dtype}")
    from tack.runtime.dispatch import same_device
    same_device("concat()", arrays)
    total = sum(a.size for a in arrays)
    result = empty((total,), dt, device=arrays[0]._backend)
    from tack.algorithms.copy import copy_with_offset
    offset = 0
    for a in arrays:
        copy_with_offset(a, result, offset, a.size)
        offset += a.size
    return result


def from_dlpack(x, *, device=None, copy: bool = False) -> Array:
    """A Tack array from a DLPack capsule or any object with ``__dlpack__``.

    Zero-copy: the array shares memory with the source, and the source is
    held alive for as long as the array lives. Works for device tensors as
    well as host ones. ``device`` must be able to wrap the tensor; by
    default the array goes on the default device when that can, else on the
    tensor's own device, as the array API's ``from_dlpack`` does.

    Wrap, not address: the Metal backend reads host memory perfectly well
    but cannot turn a host pointer into the MTLBuffer an array needs, so
    with Metal as the default a host tensor becomes a CPU array, and
    ``device="metal"`` refuses it and says so. Pass ``copy=True`` for an
    independent array instead.

    Usage:
        a = tack.from_dlpack(torch_tensor)
        a = tack.from_dlpack(cupy_array)
        a = tack.from_dlpack(numpy_array)
        a = tack.from_dlpack(vtk_capsule)
    """
    if copy:
        host = x if isinstance(x, np.ndarray) else np.from_dlpack(x)
        return array(host, device=device)

    from tack.lang.dlpack import dlpack_to_array
    return dlpack_to_array(x, device=device)


def memory_space(ptr, *, device=None) -> str:
    """Query the memory space of a pointer.

    Uses ``device`` (by default the default device) to determine where the
    pointer resides.

    Returns:
        The memory space's name, one of:

        - ``'cpu'``: unregistered host memory (regular malloc/new)
        - ``'cuda'``: CUDA device memory (cudaMalloc)
        - ``'cuda_pinned'``: CUDA pinned host memory (cudaMallocHost)
        - ``'cuda_managed'``: CUDA unified memory (cudaMallocManaged)
        - ``'hip'``: HIP device memory (hipMalloc)
        - ``'hip_pinned'``: HIP pinned host memory (hipHostMalloc)
        - ``'hip_managed'``: HIP unified memory (hipMallocManaged)
    """
    return _target(device).memory_space(ptr)


def array_from_ptr(ptr, shape, dtype, *, device=None, writable: bool = False) -> Array:
    """Wrap existing device memory as a Tack array without copying.

    Use this for interop with external libraries (pycuda, cupy, Catalyst)
    or when receiving a device pointer from a simulation framework.

    The array does NOT own the memory — Tack will not free it.
    Read-only by default; pass writable=True to enable writes.

    Raises ValueError if the pointer's memory space does not match the
    active backend (e.g. a CPU pointer with the CUDA backend), and
    TypeError if a CUDA, HIP or Level Zero pointer is not an address.

    Args:
        ptr: device pointer (integer) or backend-specific buffer object.
             - CPU: integer address or numpy array
             - Metal: MTLBuffer object (from PyObjC), or a NumPy array of
               host memory: a buffer Tack handed out, or memory on whole
               pages, wrapped in place (TypeError otherwise)
             - CUDA/HIP: device pointer as integer
             - Level Zero: device pointer as integer
        shape: the array's dimensions, counting elements
        dtype: scalar or element type (tack.f32, tack.vec3(), np.int32, ...)
        device: the device whose memory this is; by default the default device
        writable: if False (default), copyto() and fill() raise errors

    Returns:
        An Array wrapping the external memory.
    """
    from tack.runtime.kernel_utils import as_address

    shape = _array_shape(shape)
    dtype = as_dtype(dtype)
    if isinstance(dtype, ElementType):
        scalar = dtype.scalar_dtype
        storage = (int(math.prod(shape)) * dtype.size,)
    else:
        scalar, storage = dtype, shape
    backend = _target(device)
    _check_dtype(backend, scalar)

    # Validate the pointer's memory space against what this backend expects,
    # on backends that distinguish device memory; Metal MTLBuffer objects
    # and CPU numpy arrays are not addresses and need no check. Any form a
    # device pointer arrives in -- int, numpy integer, CUdeviceptr -- is
    # checked, not only a Python int.
    if backend.device_memory_spaces:
        if as_address(ptr) is None:
            raise TypeError(
                f"array_from_ptr() on the {backend.label} backend takes a "
                f"device address (an integer, or an object int() accepts), "
                f"not {type(ptr).__name__}.")
        space = backend.memory_space(ptr)
        if space not in backend.device_memory_spaces:
            raise ValueError(
                f"Pointer is in '{space}' memory but the device is "
                f"'{backend.label}'. array_from_ptr() requires a device "
                f"pointer. Use tack.array() to copy host data to the device.")

    buf = backend.wrap_ptr(ptr, scalar, storage)
    return _array_view(dtype, shape, buf, writable=writable, device=backend)


class Texture3D:
    """A 3D texture over a snapshot of an array, sampled trilinearly.

    Created via ``tack.texture3d(array, shape=(W, H, D))``.  In kernels,
    ``tex.sample(u, v, w)`` samples at normalized [0,1] coordinates using
    trilinear interpolation.

    The texture copies the array's data when it is created, and again on
    ``update()``; writes to the array in between do not reach it. That
    holds on every backend. Where the backend samples in hardware the copy
    is a texture image (a CUDA or HIP array, an MTLTexture, a Level Zero
    image); elsewhere it is a private array the generated code interpolates
    in software. Either way the texture owns it, so it is released with the
    texture and no other texture can be handed it.
    """

    def __init__(self, source_array: Array, shape_3d: tuple, interp: str = 'linear'):
        # Every hardware path builds a single-channel 32-bit float image,
        # so f64 data would be reinterpreted rather than converted.
        if source_array.dtype is not f32:
            raise ValueError(
                f"texture3d requires an f32 array, got {source_array.dtype}; "
                f"convert it first with array.astype(tack.f32)")
        # The generated code interpolates linearly on every backend; nothing
        # implements another mode.
        if interp != 'linear':
            raise ValueError(
                f"texture3d supports interp='linear' only, got {interp!r}")
        shape_3d = tuple(int(s) for s in shape_3d)
        W, H, D = shape_3d
        if W * H * D != source_array.size:
            raise ValueError(
                f"texture3d shape {shape_3d} holds {W * H * D} elements, but "
                f"the array has {source_array.size} elements")
        self.array = source_array
        self.shape_3d = shape_3d   # (W, H, D) logical 3D shape
        self.interp = interp

        # The texture lives where its array does, and kernels sampling it run
        # there (``dispatch.launch_backend`` reads ``_device``).
        backend = self._device = source_array._backend
        if backend.texture_in_hardware(shape_3d):
            self._storage = backend.create_texture_image(shape_3d)
        else:
            self._storage = Array(f32, (source_array.size,),
                                  _allocate(backend, f32, (source_array.size,)),
                                  device=backend)
        self.update()

    def update(self):
        """Copy the array's current data into the texture.

        Sampling after this sees the array as it is now, on every backend.
        """
        if isinstance(self._storage, Array):
            from tack.algorithms.copy import copy as _copy
            _copy(self.array.reshape((self.array.size,)), self._storage,
                  self.array.size)
        else:
            self._storage.upload(self.array)

    @property
    def dtype(self):
        return self.array.dtype

    def sample(self, u, v, w):
        """Sample at normalized coordinates. Only usable inside @tack.kernel."""
        raise RuntimeError("Texture3D.sample() can only be used inside a @tack.kernel")



# How a kernel names an SOA array's components: the parameter `a` becomes
# one parameter per component, in its place, and dispatch passes the
# component arrays there (`kernel_utils._flatten_soa`).
_SOA_PREFIX = "__soa"


def soa_param_names(name: str, n: int) -> tuple:
    """The kernel parameters standing for the ``n`` components of ``name``."""
    return tuple(f"{_SOA_PREFIX}{c}__{name}" for c in range(n))


def soa_component(param: str):
    """``(name, c)`` if ``param`` is component ``c`` of the SOA array ``name``, else None."""
    if not param.startswith(_SOA_PREFIX):
        return None
    c, sep, name = param[len(_SOA_PREFIX):].partition("__")
    return (name, int(c)) if sep and c.isdigit() else None

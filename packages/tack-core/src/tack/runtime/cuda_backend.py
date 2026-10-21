"""Tack CUDA backend — compiles kernels via NVRTC and dispatches on NVIDIA GPUs.

Pipeline:
    Tack IR → CUDA C source → NVRTC (PTX) → cuModuleLoad → cuLaunchKernel

Arrays are device-resident: ``tack.empty()`` allocates a device buffer via
``cuMemAlloc``.  Transfers are explicit:

    tack.copyto(a, arr)     # host → device (cuMemcpyHtoD)
    arr = a.numpy()         # device → host (cuMemcpyDtoH)

No per-dispatch copies — data stays on the GPU between kernel calls.
"""

import contextlib
import ctypes
import functools
import math
import os
import sys
import warnings

import numpy as np

from tack.codegen.reductions import array_reduction_source
from tack.lang import ir
from tack.lang.array import DeviceBuffer
from tack.lang.parallel_dims import launch_geometry
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import WORKGROUP_SIZE
from tack.runtime import device_fill
from tack.runtime.gpu_backend import GPUBackend
from tack.runtime.kernel_utils import (
    as_address,
    check_launch_size,
    new_kernel_cache,
)
from tack.runtime.nvrtc_version import old_nvrtc_message
from tack.runtime.reductions import REDUCTION_IDENTITIES, empty_reduction, reduce_numpy

_CUDA_SUPPORTED_DTYPES = frozenset({i8, u8, i16, u16, i32, u32, i64, u64, f32, f64})
from cuda.bindings import driver, nvrtc

from tack.codegen.cuda_gen import generate_cuda_source
from tack.codegen.identifiers import kernel_entry_name

# Shareable-handle types ExportableCUDABuffer may ask the driver for, in
# preference order, per platform. Each entry is (name, handle type, the device
# attribute that claims support).
#
# Windows deliberately does not list CU_MEM_HANDLE_TYPE_WIN32 -- the NT handle
# that Vulkan calls OPAQUE_WIN32 and would be the nicer thing to hand a
# consumer. Two measured reasons, on a GeForce RTX 5060 (WDDM, driver 616.56):
#
#   * cuMemCreate refuses it with CUDA_ERROR_INVALID_VALUE however
#     win32HandleMetaData is built -- NULL, an SDDL self-relative descriptor,
#     or the absolute descriptor with a real DACL that the CUDA samples
#     construct. The device attribute claims the type is supported anyway,
#     which is why _create_backing() trusts the allocation and not the
#     attribute.
#   * Asking cuda-python (13.3.1) for it is worse than an error: cuMemCreate
#     segfaults the interpreter whenever requestedHandleTypes is
#     CU_MEM_HANDLE_TYPE_WIN32 and win32HandleMetaData is non-NULL. The same
#     struct passed straight to nvcuda.dll through ctypes returns the error
#     code instead, so the fault is in the binding. There is no way to try NT
#     handles here and recover from a bad guess.
#
# WIN32_KMT is the legacy global handle (Vulkan's OPAQUE_WIN32_KMT). It needs
# no security attributes, and it works.
if sys.platform == "win32":
    _EXPORT_HANDLE_TYPES = [(
        "win32_kmt",
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_WIN32_KMT,
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED,
    )]
else:
    _EXPORT_HANDLE_TYPES = [(
        "posix_fd",
        driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
    )]

_NUMPY_DTYPE = {
    f32: np.float32,
    f64: np.float64,
    i32: np.int32,
    i64: np.int64,
    u32: np.uint32,
    u64: np.uint64,
}


_REDUCE_CUDA_SUM = array_reduction_source('cuda', 'sum')
_REDUCE_CUDA_MIN = array_reduction_source('cuda', 'min')
_REDUCE_CUDA_MAX = array_reduction_source('cuda', 'max')


def _check(err):
    """Check a CUDA driver or NVRTC result, raise on error."""
    if isinstance(err, tuple):
        err = err[0]
    if isinstance(err, driver.CUresult):
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"NVRTC error: {err}")


class _ContextToken:
    """Liveness shared by a CUDA context and every buffer allocated in it.

    Device pointers do not outlive their context. ``cuCtxDestroy`` invalidates
    every allocation made in it, and copying through one of those pointers
    afterwards faults inside the driver -- a SIGSEGV, not a CUresult we could
    check and report. So a buffer cannot ask whether its own pointer is still
    good; it has to be told. Buffers hold the token their context handed out
    and consult it before touching device memory.

    ``users`` counts the live CUDABackend objects sharing the context.
    A backend made beside the registry's one (directly, as tests do) shares
    the context, so two
    of them can overlap, and the context must survive until the last one
    goes. ``owned`` is False for a context an embedding application created:
    Tack adopts those but never destroys them, so their token never dies here.
    """

    __slots__ = ("alive", "device", "handle", "owned", "stream", "users")

    def __init__(self, handle, owned, device):
        self.handle = handle
        self.users = 1
        self.alive = True
        self.owned = owned
        self.device = device            # the device's ordinal
        # The context's queue: every launch, fill and copy Tack makes in the
        # context goes through this one stream, in order, and returns once
        # it is queued. Made on first use, in the context.
        self.stream = None

    def queue(self):
        if self.stream is None:
            with _BoundContext(self.handle):
                err, stream = driver.cuStreamCreate(0)   # ordered with the legacy stream
            _check(err)
            self.stream = stream
        return self.stream


# Every CUDA context Tack is currently aware of, keyed by handle. Contexts an
# embedding application created are in here too, marked ``owned=False``, so
# that buffers allocated in them still get a token to hold.
_CONTEXTS: dict[int, _ContextToken] = {}

# Whether every operation makes its context current for its duration
# (`_in_context`). The driver works in the thread's current context, and a
# backend's context is current on the thread that made it, as long as it is
# the only one: then nothing is switched and nothing is paid. Set for good
# once a backend's context is not current where it was made (a second
# device, or a context shared from another thread), since from then on some
# thread works in a context it did not make current.
_switching = False


def _in_context(token):
    """``token``'s context, current for a block once Tack switches contexts,
    and the thread's own after it."""
    if not _switching or token is None:
        return _NO_SWITCH
    return _BoundContext(token.handle)


_NO_SWITCH = contextlib.nullcontext()


def _current_context():
    """The context current on this thread, or None."""
    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or int(ctx) == 0:
        return None
    return ctx


def _current_device():
    """The ordinal of the device the current context is on."""
    err, dev = driver.cuCtxGetDevice()
    _check(err)
    return int(dev)


class _BoundContext:
    """Make a context current on this thread for the block, and put back
    whatever was current before. A worker thread has no current context
    until one is pushed."""

    __slots__ = ("_handle",)

    def __init__(self, handle):
        self._handle = handle

    def __enter__(self):
        _check(driver.cuCtxPushCurrent(self._handle))
        return self

    def __exit__(self, *exc):
        driver.cuCtxPopCurrent()


class _StreamMarker:
    """An event recorded on a context's stream after the work queued so far,
    waited on from any thread (`Backend.async_marker`). The event sleeps
    the waiting thread (blocking sync) and is destroyed by the wait."""

    __slots__ = ("_event", "_token")

    def __init__(self, token):
        self._token = token
        with _BoundContext(token.handle):
            flags = (driver.CUevent_flags.CU_EVENT_BLOCKING_SYNC
                     | driver.CUevent_flags.CU_EVENT_DISABLE_TIMING)
            err, self._event = driver.cuEventCreate(flags)
            _check(err)
            _check(driver.cuEventRecord(self._event, token.queue()))

    def wait(self):
        if not self._token.alive:
            raise RuntimeError(f"Cannot wait for this CUDA work: {_DEAD_CONTEXT_MSG}")
        with _BoundContext(self._token.handle):
            try:
                _check(driver.cuEventSynchronize(self._event))
            finally:
                driver.cuEventDestroy(self._event)


def _current_context_token():
    """Token for the context that is current right now, or None if untracked."""
    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or int(ctx) == 0:
        return None
    return _CONTEXTS.get(int(ctx))


_DEAD_CONTEXT_MSG = (
    "the CUDA context this array was allocated in has been destroyed, and "
    "every device pointer allocated in it died with the context. Tack keeps "
    "a context while any array of its device lives, so something outside "
    "Tack destroyed it; allocate the array again."
)


def _drain(stream):
    """Wait for the stream before a copy to pageable host memory.

    Such a copy waits for the stream inside the driver with the context
    held, so no other thread can launch meanwhile: on an H100 (driver 595)
    a launch from another thread took the whole 910 ms wait, with the copy
    queued (`cuMemcpyDtoHAsync`) or not. A thread in `cuStreamSynchronize`
    holds nothing a launch needs (511 launches at 43 us during the same
    wait), and an idle one costs 2 us, so a read waits here first. `None`
    is the legacy stream, which a synchronous copy waits for itself.
    """
    if stream is not None:
        _check(driver.cuStreamSynchronize(stream))


def _read_device_range(device_ptr, numpy_dtype, nbytes, start, count, stream=None):
    """Copy elements ``[start, start + count)`` of an allocation to the host.

    One copy of just those bytes, so ``array[i]`` on the host moves one
    element rather than the whole array, queued after the launches before
    it and waited for. The range is checked here because a device copy past
    the allocation is not caught by NumPy.
    """
    itemsize = numpy_dtype.itemsize
    if start < 0 or count < 0 or (start + count) * itemsize > nbytes:
        raise IndexError(
            f"elements [{start}, {start + count}) are outside a buffer of "
            f"{nbytes // itemsize} elements")
    out = np.empty(count, dtype=numpy_dtype)
    if count:
        if stream is None:
            _check(driver.cuMemcpyDtoH(out, int(device_ptr) + start * itemsize,
                                       count * itemsize))
        else:
            _drain(stream)
            _check(driver.cuMemcpyDtoHAsync(out, int(device_ptr) + start * itemsize,
                                            count * itemsize, stream))
            _check(driver.cuStreamSynchronize(stream))
    return out


class CUDABuffer(DeviceBuffer):
    """Device-resident buffer backed by a CUDA device pointer.

    Data lives on the GPU. Everything that touches it goes through the
    context's stream, in order with the launches: ``from_numpy`` and
    ``fill`` are queued and return at once, ``to_numpy`` and ``read_range``
    are queued and waited for. The memory is stream-ordered too
    (``cuMemAllocAsync``/``cuMemFreeAsync``), so an array Python drops while
    a queued launch still uses it is freed after that launch, and the
    driver's pool hands the memory back out without a device round trip.
    """

    backend_name = "cuda"
    # Memory handed outside Tack (DLPack, a wrapped pointer) may be read
    # with no Tack call between: a launch that uses it completes before
    # returning, as on Metal.
    _shared = False

    def __init__(self, numpy_dtype, shape, token=None):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(math.prod(shape)) * self._numpy_dtype.itemsize
        self._token = token = token or _current_context_token()
        stream = token.queue()
        with _in_context(token):
            # A zero-byte request is rejected; an empty array gets one byte
            # it never reads, as on Level Zero.
            err, self._device_ptr = driver.cuMemAllocAsync(max(self._nbytes, 1), stream)
            _check(err)
            _check(driver.cuMemsetD8Async(self._device_ptr, 0, self._nbytes, stream))

    @property
    def _stream(self):
        return self._token.queue()

    # Reading waits for the stream, so `Array.numpy_async` reads in a
    # worker thread, with the context made current there.
    waits_off_thread = True

    def thread_binding(self):
        token = getattr(self, "_token", None)
        if token is None:
            return super().thread_binding()
        self._live("read")
        return _BoundContext(token.handle)

    def synchronize(self):
        """Wait for every queued launch, fill and copy in this context."""
        self._live("wait for")
        with _in_context(self._token):
            _check(driver.cuStreamSynchronize(self._stream))

    def share(self):
        """Hand this memory outside Tack: wait for the queued work now, and
        have every later launch that uses it complete before returning."""
        self._shared = True
        self.synchronize()

    def _live(self, verb):
        """Refuse to touch device memory whose context is gone."""
        token = getattr(self, "_token", None)
        if token is not None and not token.alive:
            raise RuntimeError(f"Cannot {verb} this CUDA array: {_DEAD_CONTEXT_MSG}")

    @property
    def address(self) -> int:
        return int(self._device_ptr)

    @property
    def device_ptr(self):
        # Guarded because this is what a kernel launch reads: without the
        # check a dispatch against a stale array faults in the driver.
        self._live("run a kernel against")
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        # Queued after the launches before it, which may still read the old
        # contents. The host array is pageable, so the driver has staged it
        # before this returns and the array may go.
        self._live("write to")
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        with _in_context(self._token):
            _check(driver.cuMemcpyHtoDAsync(self._device_ptr, src, self._nbytes,
                                            self._stream))

    def to_numpy(self) -> np.ndarray:
        self._live("read")
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        # Through a flat view: cuda-python does not take a zero-dimensional
        # array as a host buffer (CUDA_ERROR_INVALID_VALUE, or a crash).
        stream = self._stream
        with _in_context(self._token):
            _drain(stream)
            _check(driver.cuMemcpyDtoHAsync(out.reshape(-1), self._device_ptr, self._nbytes,
                                            stream))
            _check(driver.cuStreamSynchronize(stream))
        return out

    def read_range(self, start: int, count: int) -> np.ndarray:
        self._live("read")
        with _in_context(self._token):
            return _read_device_range(self._device_ptr, self._numpy_dtype, self._nbytes,
                                      start, count, self._stream)

    def fill(self, value):
        self.fill_bytes(device_fill.scalar_pattern(value, self._numpy_dtype))

    def fill_bytes(self, pattern):
        self._live("write to")
        with _in_context(self._token):
            return self._fill_bytes(pattern)

    def _fill_bytes(self, pattern):
        base = int(self._device_ptr)
        stream = self._stream

        def native(pattern):
            byte = device_fill.uniform_byte(pattern)
            if byte is not None:
                _check(driver.cuMemsetD8Async(self._device_ptr, byte, self._nbytes, stream))
            elif len(pattern) == 2:
                _check(driver.cuMemsetD16Async(self._device_ptr,
                                               device_fill.as_unsigned(pattern),
                                               self._nbytes // 2, stream))
            elif len(pattern) == 4:
                _check(driver.cuMemsetD32Async(self._device_ptr,
                                               device_fill.as_unsigned(pattern),
                                               self._nbytes // 4, stream))
            else:
                return False
            return True

        device_fill.fill(
            self._nbytes, pattern, native,
            lambda offset, data: _check(driver.cuMemcpyHtoDAsync(
                base + offset, np.frombuffer(data, np.uint8), len(data), stream)),
            lambda dst, src, size: _check(driver.cuMemcpyDtoDAsync(
                base + dst, base + src, size, stream)))
        return True

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def export_memory(self):
        """Export as ExportedMemory. Lazily copies into exportable memory."""
        self._live("export")
        if not hasattr(self, '_export_buf'):
            self._export_buf = ExportableCUDABuffer(self._numpy_dtype, self._shape,
                                                    self._token)
            self.synchronize()
            with _in_context(self._token):
                _check(driver.cuMemcpyDtoD(
                    self._export_buf._device_ptr, self._device_ptr, self._nbytes))
        return self._export_buf.export_memory()

    def __del__(self):
        # Freeing into a destroyed context is the same fault as copying into
        # one, and the context took this allocation with it anyway.
        token = getattr(self, "_token", None)
        if token is not None and not token.alive:
            return
        if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
            try:
                # In stream order: after every launch queued so far, which
                # may still be using this memory.
                stream = token.queue()
                with _in_context(token):
                    driver.cuMemFreeAsync(self._device_ptr, stream)
            except Exception:
                pass


class ExportableCUDABuffer(DeviceBuffer):
    """Device buffer allocated via CUDA VMM so it can be shared with other APIs.

    Uses cuMemCreate/cuMemMap instead of cuMemAlloc so the underlying memory
    can be exported as an OS handle for cross-API sharing (e.g. Vulkan import
    via VK_KHR_external_memory_fd, or its Win32 equivalent).

    Which kind of handle that is depends on the platform, so the caller is
    told rather than left to assume: see ``ExportedMemory.handle_type`` and
    ``_EXPORT_HANDLE_TYPES``.
    """

    def __init__(self, numpy_dtype, shape, token=None):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(math.prod(shape)) * self._numpy_dtype.itemsize

        # Set before anything can fail, so __del__ can tell what was reached.
        self._mem_handle = None
        self._device_ptr = None
        self._exported_handle = None
        self._handle_type_name = None
        self._token = token = token or _current_context_token()
        with _in_context(token):
            self._map()

    def _map(self):
        err, self._cuda_device = driver.cuCtxGetDevice()
        _check(err)

        self._create_backing()

        err, self._device_ptr = driver.cuMemAddressReserve(
            self._alloc_size, self._granularity, 0, 0)
        _check(err)

        _check(driver.cuMemMap(
            self._device_ptr, self._alloc_size, 0, self._mem_handle, 0))

        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self._cuda_device
        access.flags = (
            driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        _check(driver.cuMemSetAccess(
            self._device_ptr, self._alloc_size, [access], 1))

        # Zero-initialise
        _check(driver.cuMemsetD8(self._device_ptr, 0, self._alloc_size))

    def _allocation_prop(self, handle_type):
        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._cuda_device
        prop.requestedHandleTypes = handle_type
        return prop

    def _create_backing(self):
        """Allocate physical memory with the first handle type that works.

        The device attribute is consulted first but is not taken as the
        answer: an RTX 5060 reports HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 1 and
        then fails the matching cuMemCreate. The allocation succeeding is the
        only real evidence, so every candidate is actually tried.
        """
        attempts = []
        for name, handle_type, attribute in _EXPORT_HANDLE_TYPES:
            err, supported = driver.cuDeviceGetAttribute(attribute, self._cuda_device)
            if err != driver.CUresult.CUDA_SUCCESS or not supported:
                attempts.append(f"{name}: device reports no support for this handle type")
                continue

            prop = self._allocation_prop(handle_type)
            err, granularity = driver.cuMemGetAllocationGranularity(
                prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
            if err != driver.CUresult.CUDA_SUCCESS:
                attempts.append(f"{name}: cuMemGetAllocationGranularity failed with {err}")
                continue

            alloc_size = ((max(self._nbytes, 1) + granularity - 1)
                          // granularity) * granularity
            err, mem_handle = driver.cuMemCreate(alloc_size, prop, 0)
            if err != driver.CUresult.CUDA_SUCCESS:
                attempts.append(f"{name}: cuMemCreate failed with {err}")
                continue

            self._handle_type_name = name
            self._handle_type = handle_type
            self._mem_handle = mem_handle
            self._granularity = granularity
            self._alloc_size = alloc_size
            return

        raise RuntimeError(
            "This CUDA device cannot allocate exportable memory on "
            f"{sys.platform}. Tried:\n"
            + "\n".join(f"  {a}" for a in attempts)
            + "\nExportable memory is only needed by Array.export_memory() for "
              "sharing with another API; ordinary allocations are unaffected."
        )

    @property
    def address(self) -> int:
        return int(self._device_ptr)

    @property
    def device_ptr(self):
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        with _in_context(self._token):
            _check(driver.cuMemcpyHtoD(self._device_ptr, src, self._nbytes))

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        stream = self._queued_stream
        with _in_context(self._token):
            _drain(stream)
            _check(driver.cuMemcpyDtoH(out.reshape(-1), self._device_ptr, self._nbytes))
        return out

    # A copy on the legacy stream waits for the context's stream, which is
    # made blocking: `Array.numpy_async` reads it in a worker thread.
    waits_off_thread = True

    @property
    def _queued_stream(self):
        """The context's stream, whose queued work a legacy-stream copy
        waits for; None when the context is untracked."""
        token = getattr(self, "_token", None)
        return token.queue() if token is not None and token.alive else None

    def thread_binding(self):
        token = getattr(self, "_token", None)
        return super().thread_binding() if token is None else _BoundContext(token.handle)

    def read_range(self, start: int, count: int) -> np.ndarray:
        stream = self._queued_stream
        with _in_context(self._token):
            _drain(stream)
            return _read_device_range(self._device_ptr, self._numpy_dtype, self._nbytes,
                                      start, count)

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def export_memory(self):
        """Export as ExportedMemory (handle + size + UUID). The handle is cached."""
        from tack.lang.array import ExportedMemory
        if self._exported_handle is None:
            err, handle = driver.cuMemExportToShareableHandle(
                self._mem_handle, self._handle_type, 0)
            _check(err)
            self._exported_handle = int(handle)
        err, uuid = driver.cuDeviceGetUuid(self._cuda_device)
        _check(err)
        return ExportedMemory(
            backend="cuda",
            size=self._nbytes,
            allocation_size=self._alloc_size,
            handle=self._exported_handle,
            handle_type=self._handle_type_name,
            device_uuid=bytes(uuid.bytes),
        )

    def __del__(self):
        try:
            # A POSIX fd is ours to close. A WIN32_KMT handle is not a kernel
            # handle at all -- it is a legacy global D3DKMT value, closing it
            # is not our job and CloseHandle on it would be a bug.
            if self._exported_handle is not None and self._handle_type_name == "posix_fd":
                os.close(self._exported_handle)
            self._exported_handle = None
            # The fd is ours whatever happened to the context, but the
            # mapping and the handle went down with it.
            token = getattr(self, "_token", None)
            if token is not None and not token.alive:
                return
            with _in_context(token):
                if self._device_ptr is not None:
                    driver.cuMemUnmap(self._device_ptr, self._alloc_size)
                    driver.cuMemAddressFree(self._device_ptr, self._alloc_size)
                if self._mem_handle is not None:
                    driver.cuMemRelease(self._mem_handle)
        except Exception:
            pass


def _libc_uselocale():
    """glibc's ``uselocale``, or None where there is none (Windows)."""
    try:
        uselocale = ctypes.CDLL(None).uselocale
    except (OSError, TypeError, AttributeError):
        return None
    uselocale.restype = ctypes.c_void_p
    uselocale.argtypes = [ctypes.c_void_p]
    return uselocale


_uselocale = _libc_uselocale()


def _compile_keeping_thread_locale(prog, opts):
    """``nvrtcCompileProgram``, with the calling thread's locale restored.

    NVRTC 12.0 switches the calling thread to the C locale with
    ``uselocale`` and leaves it there. ``setlocale`` still reports the
    process's locale, but Python's default text encoding follows the
    thread's, so every ``open``/``read_text``/``write_text`` without an
    encoding became ASCII after the first CUDA kernel compiled, and failed
    on any non-ASCII character. NVRTC 13.2 leaves the locale alone.
    """
    saved = _uselocale(None) if _uselocale is not None else None
    try:
        return nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    finally:
        if saved is not None:
            _uselocale(saved)


@functools.cache
def _warn_if_old_nvrtc():
    """Warn once per process, at the first compile, about an untested NVRTC."""
    err, major, minor = nvrtc.nvrtcVersion()
    _check(err)
    message = old_nvrtc_message(major, minor)
    if message:
        warnings.warn(message, stacklevel=3)


# Preserve NaNs, signed zeros, and expression grouping in every kernel.
# Adjacent multiply/add contraction is permitted by the language contract.
# Part of the disk cache's target: PTX compiled with other options is another
# artifact.
_NVRTC_OPTIONS = [b"--ftz=false", b"--prec-div=true", b"--prec-sqrt=true",
                  b"--fmad=true", b"--extra-device-vectorization"]


def _compile_ptx(cuda_source: str, func_name: str) -> bytes:
    """Compile CUDA C source to PTX via NVRTC."""
    _warn_if_old_nvrtc()
    src = cuda_source.encode("utf-8")
    err, prog = nvrtc.nvrtcCreateProgram(src, f"{func_name}.cu".encode(), 0, None, None)
    _check(err)

    # cuda-python marshals a list of bytes; a ctypes array can be misread.
    compile_result = _compile_keeping_thread_locale(prog, _NVRTC_OPTIONS)
    compile_err = compile_result[0] if isinstance(compile_result, tuple) else compile_result

    if compile_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(prog, log)
        nvrtc.nvrtcDestroyProgram(prog)
        raise RuntimeError(
            f"NVRTC compilation failed:\n{log.decode(errors='replace')}\n"
            f"Source:\n{cuda_source}"
        )

    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    _check(err)
    ptx = b" " * ptx_size
    _check(nvrtc.nvrtcGetPTX(prog, ptx))
    nvrtc.nvrtcDestroyProgram(prog)
    return ptx


def _load_kernel(ptx: bytes, kernel_name: str, param_types, param_is_array,
                 param_is_texture) -> "CompiledCUDAKernel":
    """A kernel from its PTX, loaded into the current context."""
    err, module = driver.cuModuleLoadData(ptx)
    _check(err)
    err, func = driver.cuModuleGetFunction(module, kernel_name.encode())
    _check(err)
    compiled = CompiledCUDAKernel(module, func, kernel_name, param_types,
                                  param_is_array, param_is_texture)
    compiled.ptx = ptx
    return compiled


_CUDA_CTYPES_MAP = {f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
                    u32: ctypes.c_uint, u64: ctypes.c_ulonglong}


class CUDATextureImage:
    """A CUDA 3D array and the texture object that samples it.

    Owned by one `Texture3D`, which uploads its array's data at creation and
    on ``update()``. Nothing else keeps a reference, so the array and the
    texture object are destroyed with the texture: no later texture can be
    served one that was made from someone else's memory.
    """

    def __init__(self, shape_3d, token=None):
        self._shape = shape_3d
        self._token = token = token or _current_context_token()
        with _in_context(token):
            self._create()

    def _create(self):
        W, H, D = self._shape
        # Create a CUDA array descriptor for a 3D float texture
        array_desc = driver.CUDA_ARRAY3D_DESCRIPTOR()
        array_desc.Width = W
        array_desc.Height = H
        array_desc.Depth = D
        array_desc.Format = driver.CUarray_format.CU_AD_FORMAT_FLOAT
        array_desc.NumChannels = 1
        array_desc.Flags = 0

        err, self._array = driver.cuArray3DCreate(array_desc)
        _check(err)

        # Create texture descriptor
        tex_desc = driver.CUDA_TEXTURE_DESC()
        tex_desc.addressMode = (
            driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
            driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
            driver.CUaddress_mode.CU_TR_ADDRESS_MODE_CLAMP,
        )
        tex_desc.filterMode = driver.CUfilter_mode.CU_TR_FILTER_MODE_LINEAR
        tex_desc.flags = driver.CU_TRSF_NORMALIZED_COORDINATES

        # Create resource descriptor
        res_desc = driver.CUDA_RESOURCE_DESC()
        res_desc.resType = driver.CUresourcetype.CU_RESOURCE_TYPE_ARRAY
        res_desc.res.array.hArray = self._array

        # Create resource view descriptor (default — full mip level 0)
        view_desc = driver.CUDA_RESOURCE_VIEW_DESC()
        view_desc.format = driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X32
        view_desc.width = W
        view_desc.height = H
        view_desc.depth = D

        err, self._tex_obj = driver.cuTexObjectCreate(res_desc, tex_desc, view_desc)
        _check(err)

    @property
    def handle(self) -> int:
        """The texture object a kernel launch passes."""
        return int(self._tex_obj)

    def upload(self, array):
        """Copy an array's device buffer into the array.

        The texture object reads the array, so it sees the new contents
        without being recreated.
        """
        W, H, D = self._shape
        copy_params = driver.CUDA_MEMCPY3D()
        # Source: device pointer, pitched linear memory
        copy_params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy_params.srcDevice = array._buffer.device_ptr
        copy_params.srcPitch = W * 4   # bytes per row
        copy_params.srcHeight = H
        # Destination: CUDA array
        copy_params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
        copy_params.dstArray = self._array
        # Extent
        copy_params.WidthInBytes = W * 4
        copy_params.Height = H
        copy_params.Depth = D

        stream = array._buffer._stream
        with _in_context(self._token):
            _check(driver.cuMemcpy3DAsync(copy_params, stream))

    def __del__(self):
        # Same rule as CUDABuffer: the context took these with it.
        token = getattr(self, "_token", None)
        if token is not None and not token.alive:
            return
        try:
            with _in_context(token):
                if hasattr(self, '_tex_obj'):
                    driver.cuTexObjectDestroy(self._tex_obj)
                if hasattr(self, '_array'):
                    driver.cuArrayDestroy(self._array)
        except Exception:
            pass


class CompiledCUDAKernel:
    """A compiled CUDA kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types, param_is_array,
                 param_is_texture=None):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_array = param_is_array
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self.ptx = None     # what NVRTC made, kept for the disk cache

    def __call__(self, kernel_args: list, loop_end: int, extents=(), limits=None,
                 stream=None):
        """Queue the CUDA kernel on ``stream`` and return.

        A multi-dimensional loop (``extents``, slowest first) launches a grid
        of its shape, or a flat one with ``__flat__`` set when an extent is
        past the device's grid limits (``limits``; see launch_geometry).
        """
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for arg, ptype, is_array, is_tex in zip(
                kernel_args, self._param_types, self._param_is_array,
                self._param_is_texture):
            if is_tex:
                # A CUDATextureImage; cudaTextureObject_t is unsigned long long
                arg_values.append(ctypes.c_ulonglong(arg.handle))
            elif is_array:
                arg_values.append(ctypes.c_void_p(int(arg._buffer.device_ptr)))
            else:
                ct = _CUDA_CTYPES_MAP[ptype]
                arg_values.append(ct(arg))
        arg_values.append(n_val)
        if extents:
            grid, block, flat = launch_geometry(extents, block_size=WORKGROUP_SIZE, **limits)
            arg_values.append(ctypes.c_int(int(flat)))
            arg_values.extend(ctypes.c_longlong(e) for e in extents)
        else:
            block = (WORKGROUP_SIZE, 1, 1)
            grid = ((loop_end + WORKGROUP_SIZE - 1) // WORKGROUP_SIZE, 1, 1)

        arg_ptrs = (ctypes.c_void_p * len(arg_values))()
        for i, val in enumerate(arg_values):
            arg_ptrs[i] = ctypes.addressof(val)

        _check(driver.cuLaunchKernel(
            self._func,
            *grid,
            *block,
            0, stream or 0,
            arg_ptrs, 0,
        ))


class CUDABackend(GPUBackend):
    """CUDA GPU backend — device-resident arrays, NVRTC compilation."""

    name = "cuda"
    display_name = "CUDA"
    supported_dtypes = _CUDA_SUPPORTED_DTYPES
    supports_device_reductions = True
    supports_workgroups = True
    device_memory_spaces = frozenset({"cuda", "cuda_pinned", "cuda_managed"})
    multi_device = True

    @classmethod
    def device_count(cls) -> int:
        _check(driver.cuInit(0))
        err, count = driver.cuDeviceGetCount()
        _check(err)
        return int(count)

    @classmethod
    def current_ordinal(cls) -> int:
        """The device of a context another library made current on this
        thread (an embedding framework's), else 0. A context Tack made and
        left current is not a choice the thread made."""
        ctx = _current_context()
        if ctx is None:
            return 0
        token = _CONTEXTS.get(int(ctx))
        if token is not None and token.owned:
            return 0
        return _current_device()

    def __init__(self, ordinal=None):
        _check(driver.cuInit(0))
        if ordinal is None:
            ordinal = self.current_ordinal()
        count = self.device_count()
        if not 0 <= ordinal < count:
            raise ValueError(f"cuda:{ordinal}: this machine has {count} CUDA "
                             f"device{'s' if count != 1 else ''}")
        self.ordinal = ordinal
        err, self._device = driver.cuDeviceGet(ordinal)
        _check(err)
        self._token = self._context_for(ordinal)
        self._context = self._token.handle
        self._owns_context = self._token.owned

        # A flat launch is one-dimensional, so gridDim.x bounds its size.
        # A multi-dimensional loop also needs the y/z grid limits (65535 on
        # NVIDIA parts) and the block limits (z is 64).
        def attribute(name):
            err, value = driver.cuDeviceGetAttribute(
                getattr(driver.CUdevice_attribute, f"CU_DEVICE_ATTRIBUTE_{name}"), self._device)
            _check(err)
            return int(value)
        max_blocks = attribute("MAX_GRID_DIM_X")
        self._max_launch = max_blocks * WORKGROUP_SIZE
        self._launch_limits = {
            "max_grid": (max_blocks, attribute("MAX_GRID_DIM_Y"), attribute("MAX_GRID_DIM_Z")),
            "max_block": (attribute("MAX_BLOCK_DIM_X"), attribute("MAX_BLOCK_DIM_Y"),
                          attribute("MAX_BLOCK_DIM_Z")),
        }

        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledCUDAKernel}
        # Keep freed memory in the driver's pool rather than returning it to
        # the device at every synchronization: a filter allocates and drops
        # dozens of temporaries, and the pool hands them back out at once.
        try:
            err, pool = driver.cuDeviceGetDefaultMemPool(self._device)
            _check(err)
            _check(driver.cuMemPoolSetAttribute(
                pool, driver.CUmemPool_attribute.CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                driver.cuuint64_t(2 ** 64 - 1)))
        except Exception:
            pass                            # the pool releases at syncs; still correct

    def _context_for(self, ordinal):
        """The token of the context this backend works in, on device ``ordinal``.

        Reuse an existing CUDA context if one is already active on the
        device (e.g. from a simulation framework like AMReX), or one Tack
        holds for it already; only create a new context when there is none.

        Whether the context may be destroyed is a property of the context,
        not of the backend that happens to hold it: a backend made beside the
        registry's one is a second backend on the context, so two
        CUDABackend objects routinely share one context for a moment. When
        that context is ours, both must agree that the *last* one out
        destroys it -- an adopter that recorded `_owns_context = False`
        would be left holding a destroyed context as soon as its creator
        was collected, and every later call would fail with
        CUDA_ERROR_INVALID_CONTEXT. Refcounting the context gets that
        right while leaving a foreign context untouched, which is the
        whole point of adopting one.
        """
        global _switching
        current = _current_context()
        token = None
        if current is not None and _current_device() == ordinal:
            token = _CONTEXTS.get(int(current))
            if token is None:
                # Nobody here created this one, so it belongs to the embedding
                # application: adopt it, but never destroy it.
                token = _CONTEXTS[int(current)] = _ContextToken(current, False, ordinal)
            else:
                token.users += 1
        else:
            token = next((t for t in _CONTEXTS.values()
                          if t.device == ordinal and t.alive), None)
            if token is not None:
                token.users += 1
            else:
                err, ctx = driver.cuCtxCreate(None, 0, self._device)
                _check(err)
                if current is not None:
                    # cuCtxCreate made it current: leave the thread's own.
                    driver.cuCtxPopCurrent()
                token = _CONTEXTS[int(ctx)] = _ContextToken(ctx, True, ordinal)
        # A context is current only on the thread that made or adopted it.
        # Once this one is not current here, or there are two, some thread
        # works in a context it did not make current: from now on every
        # operation makes its own current (`_in_context`).
        made_current = _current_context()
        if (made_current is None or int(made_current) != int(token.handle)
                or len(_CONTEXTS) > 1):
            _switching = True
        return token

    @property
    def _token_or_none(self):
        """The context's token; None for a backend built without one (the
        dispatch-path tests make one with ``__new__``)."""
        return getattr(self, "_token", None)

    @property
    def memory_domain(self) -> tuple:
        # A pointer belongs to the context it was allocated in, and two
        # backends on one device may hold different contexts (one adopted).
        token = self._token_or_none
        if token is None:
            return (self.name, self.ordinal)
        return (self.name, self.ordinal, int(token.handle))

    @property
    def _stream(self):
        """The context's queue; None for a backend built without a context
        (the dispatch-path tests make one with ``__new__``), which launches
        on the legacy stream."""
        token = getattr(self, "_token", None)
        return token.queue() if token is not None else None

    def synchronize(self):
        """Wait for every launch, fill and copy queued in this context. A
        launch's error surfaces here, at the next host access, rather than
        at its call."""
        stream = self._stream
        if stream is not None:
            with _in_context(self._token_or_none):
                _check(driver.cuStreamSynchronize(stream))

    def async_marker(self):
        """An event after the work queued so far in this context, for
        `tack.sync_async` to wait on in a worker thread."""
        token = getattr(self, "_token", None)
        if token is None:
            return None                      # the legacy stream: wait inline
        return _StreamMarker(token)

    def allocate_array(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> CUDABuffer:
        if exportable:
            return ExportableCUDABuffer(dtype.numpy_dtype, shape, self._token_or_none)
        return CUDABuffer(dtype.numpy_dtype, shape, self._token_or_none)

    def texture_in_hardware(self, shape_3d) -> bool:
        return True

    def create_texture_image(self, shape_3d) -> CUDATextureImage:
        return CUDATextureImage(shape_3d, self._token_or_none)

    def memory_space(self, ptr) -> str:
        """Query where a pointer resides: 'cpu', 'cuda', or 'cuda_managed'.

        Uses the CUDA driver API (cuPointerGetAttribute) via cuda-python
        bindings to classify the pointer.

        Returns:
            'cuda'         — device memory (cudaMalloc)
            'cuda_pinned'  — pinned host memory (cudaMallocHost)
            'cuda_managed' — unified memory (cudaMallocManaged)
            'cpu'          — unregistered host memory
        """
        addr = as_address(ptr)
        if addr is None:
            # Not an address at all. Asked and answered before the driver
            # is involved, so the API call below needs no `try` around it
            # and a mistaken one is a traceback rather than a quiet "cpu".
            return "cpu"

        with _in_context(self._token_or_none):
            err, mem_type = driver.cuPointerGetAttribute(
                driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, addr)
        if err != driver.CUresult.CUDA_SUCCESS:
            # The documented reply for memory the driver does not know --
            # CUDA_ERROR_INVALID_VALUE for an ordinary host allocation.
            # Verified on an RTX 4060 Ti, 2026-08-11: this branch answers
            # for every pointer and the old fallback never fired.
            return "cpu"
        # CU_MEMORYTYPE_HOST=1, CU_MEMORYTYPE_DEVICE=2,
        # CU_MEMORYTYPE_ARRAY=3, CU_MEMORYTYPE_UNIFIED=4
        return {1: "cuda_pinned", 2: "cuda", 4: "cuda_managed"}.get(
            int(mem_type), "cpu")

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing CUDA device pointer without allocating or copying."""
        buf = CUDABuffer.__new__(CUDABuffer)
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(math.prod(shape)) * buf._numpy_dtype.itemsize
        buf._device_ptr = ptr  # integer or CUdeviceptr
        buf._owned = False
        buf._token = self._token_or_none or _current_context_token()
        buf._shared = True      # someone else's memory: launches on it complete
        return buf

    _texture_image_type = CUDATextureImage

    def _launch(self, compiled, kernel_args, loop_end, extents):
        compiled(kernel_args, loop_end, extents, self._launch_limits, self._stream)

    def _device_scope(self):
        return _in_context(self._token_or_none)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledCUDAKernel:
        """Compile Tack IR → CUDA C → PTX → CUfunction."""
        kernel_name = kernel_entry_name(ir_func.name)
        cuda_source = generate_cuda_source(ir_func)
        ptx = _compile_ptx(cuda_source, kernel_name)
        param_types = [p.type_annotation for p in ir_func.params]
        param_is_array = [p._is_array for p in ir_func.params]
        param_is_texture = [p._is_texture for p in ir_func.params]
        return _load_kernel(ptx, kernel_name, param_types, param_is_array, param_is_texture)

    # ── The disk cache: PTX, loaded by the driver ──────────────────────
    # The driver compiles PTX for the device at load, so the stored artifact
    # is portable across devices of one architecture family, and the target
    # names what that compilation depends on: the device, the driver that
    # compiles it, and the NVRTC and options that made the PTX.

    _target = None

    def cache_target(self):
        if self._target is None:
            err, name = driver.cuDeviceGetName(128, self._device)
            _check(err)
            device_name = bytes(name).split(b"\0", 1)[0].decode(errors="replace")
            err, major = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self._device)
            _check(err)
            err, minor = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self._device)
            _check(err)
            err, driver_version = driver.cuDriverGetVersion()
            _check(err)
            err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
            _check(err)
            options = " ".join(o.decode() for o in _NVRTC_OPTIONS)
            self._target = (f"{device_name} sm_{major}{minor}; driver {driver_version}; "
                            f"nvrtc {nvrtc_major}.{nvrtc_minor}; {options}")
        return self._target

    def variant_artifact(self, payload):
        compiled, pack_info, _ = payload
        if compiled.ptx is None:
            raise ValueError("no PTX was kept for this kernel")
        return {"ptx": compiled.ptx, "name": compiled._func_name,
                "param_types": compiled._param_types,
                "param_is_array": compiled._param_is_array,
                "param_is_texture": compiled._param_is_texture,
                "pack_info": pack_info}

    def load_variant(self, ir_func, artifact, effective_args):
        from tack.runtime.kernel_utils import _create_pack_arrays
        compiled = _load_kernel(artifact["ptx"], artifact["name"], artifact["param_types"],
                                artifact["param_is_array"], artifact["param_is_texture"])
        pack_info = artifact["pack_info"]
        # The scalar pack buffers are device memory of this process, made
        # from this call's scalars as _build_variant makes them.
        pack_arrays = (_create_pack_arrays(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_arrays

    def reduce_array(self, array, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        with _in_context(self._token_or_none):
            return self._reduce_array(array, op)

    def _reduce_array(self, array, op: str) -> float:
        from tack.lang.types import f32
        if array.size == 0:
            return empty_reduction(op)
        if array.dtype is not f32:
            return reduce_numpy(array.numpy(), op)
        n = int(math.prod(array.shape))
        check_launch_size(f"Array {op}()", n, self._max_launch, self.label)

        if not hasattr(self, '_reduce_cache'):
            self._reduce_cache = {}
        if op not in self._reduce_cache:
            self._reduce_cache[op] = self._compile_reduce(op)

        func, module = self._reduce_cache[op]

        out_buf = CUDABuffer(np.float32, (1,), self._token_or_none)
        out_buf.from_numpy(np.array([REDUCTION_IDENTITIES[op]], dtype=np.float32))

        block_dim = 256
        grid_dim = (n + block_dim - 1) // block_dim

        # Dispatch
        in_ptr = ctypes.c_void_p(int(array._buffer.device_ptr))
        out_ptr = ctypes.c_void_p(int(out_buf.device_ptr))
        n_val = ctypes.c_longlong(n)
        args = (ctypes.c_void_p * 3)()
        args[0] = ctypes.addressof(in_ptr)
        args[1] = ctypes.addressof(out_ptr)
        args[2] = ctypes.addressof(n_val)

        _check(driver.cuLaunchKernel(
            func, grid_dim, 1, 1, block_dim, 1, 1,
            block_dim * 4, self._stream, args, 0))

        result = out_buf.to_numpy()         # queued after the launch, and waited for
        return float(result[0])

    def _compile_reduce(self, op: str):
        """Compile a reduction kernel for the given op."""
        _REDUCE_CUDA = {
            "sum": _REDUCE_CUDA_SUM,
            "min": _REDUCE_CUDA_MIN,
            "max": _REDUCE_CUDA_MAX,
        }
        func_names = {
            "sum": "reduce_sum_f32",
            "min": "reduce_min_f32",
            "max": "reduce_max_f32",
        }
        src = _REDUCE_CUDA[op]
        ptx = _compile_ptx(src, func_names[op])
        err, module = driver.cuModuleLoadData(ptx)
        _check(err)
        err, func = driver.cuModuleGetFunction(module, func_names[op].encode())
        _check(err)
        return func, module

    def __del__(self):
        token = getattr(self, "_token", None)
        if token is None:
            return
        try:
            token.users -= 1
            if token.users > 0:
                return
            _CONTEXTS.pop(int(token.handle), None)
            if token.stream is not None:
                with _in_context(token):
                    driver.cuStreamSynchronize(token.stream)
                    driver.cuStreamDestroy(token.stream)
                token.stream = None
            if token.owned:
                # Mark dead before destroying, so any buffer still holding
                # this token reports the problem instead of faulting.
                token.alive = False
                driver.cuCtxDestroy(token.handle)
        except Exception:
            pass

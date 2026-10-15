"""Tack HIP backend — compiles kernels via hipRTC and dispatches on AMD GPUs.

Pipeline:
    Tack IR → HIP C source → hipRTC (code object) → hipModuleLoad → hipLaunchKernel

Arrays are device-resident: ``tack.empty()`` allocates a device buffer via
``hipMallocAsync``.  Transfers are explicit:

    tack.copyto(a, arr)     # host → device, queued
    arr = a.numpy()         # device → host, queued and waited for

No per-dispatch copies — data stays on the GPU between kernel calls. Launches,
fills and copies go through one stream per device, in order, and return once
queued; the host waits where it reads an array or calls ``tack.sync()``.

Every wait is on an event recorded after the queued work, through ctypes
(`_StreamMarker`, `_runtime`), never ``hipStreamSynchronize``: hip-python's
wrappers hold the GIL while the runtime waits, which would stop every other
Python thread (and an ``asyncio`` loop awaiting `tack.sync_async`) for the
length of the kernel, and ROCm's stream synchronize holds the stream while it
waits, so no other thread could launch on it meanwhile.

Requires: hip-python (the ``[hip]`` extra)
"""

import ctypes
import inspect
import threading

import numpy as np

from tack.codegen.reductions import array_reduction_source
from tack.lang import ir
from tack.lang.array import DeviceBuffer
from tack.lang.parallel_dims import launch_geometry
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import WORKGROUP_SIZE
from tack.runtime import device_fill
from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    _get_launch,
    as_address,
    bind_textures,
    check_launch_size,
    new_kernel_cache,
    resolve_variant,
)
from tack.runtime.reductions import REDUCTION_IDENTITIES, empty_reduction, reduce_numpy

_HIP_SUPPORTED_DTYPES = frozenset({i8, u8, i16, u16, i32, u32, i64, u64, f32, f64})
import hip as _hip_python
from hip import hip, hiprtc

# hip-python 10 is a compatibility package over `rocm.bindings`: it has
# `VERSION` but no `LONG_VERSION` (7.x has both, equal) and no `chip`.
_HIP_PYTHON_VERSION = (getattr(_hip_python, "LONG_VERSION", None)
                       or getattr(_hip_python, "VERSION", "unknown"))

from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name

_REDUCE_HIP_SUM = array_reduction_source('hip', 'sum')
_REDUCE_HIP_MIN = array_reduction_source('hip', 'min')
_REDUCE_HIP_MAX = array_reduction_source('hip', 'max')

_NUMPY_DTYPE = {
    f32: np.float32,
    f64: np.float64,
    i32: np.int32,
    i64: np.int64,
    u32: np.uint32,
    u64: np.uint64,
}


class HIPError(RuntimeError):
    """A failed HIP runtime call; ``code`` is the `hipError_t`."""

    def __init__(self, code):
        super().__init__(f"HIP error: {code}")
        self.code = code


def _check_hip(result):
    """Check a HIP runtime result, raise on error (handles tuple returns)."""
    err = result[0] if isinstance(result, tuple) else result
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise HIPError(err)


_TAKES_OUTPUT = {}


def _call_with_output(fn, struct, *args):
    """Call a HIP function whose first C parameter is an output struct, and
    return `(error, struct)`. hip-python 7.x takes the struct as the first
    argument and returns the error; hip-python 10 allocates it and returns
    `(error, struct)`. Which one is read off the binding's signature."""
    takes = _TAKES_OUTPUT.get(fn.__name__)
    if takes is None:
        try:
            takes = len(inspect.signature(fn).parameters) > len(args)
        except (TypeError, ValueError):
            takes = True
        _TAKES_OUTPUT[fn.__name__] = takes
    if takes:
        out = struct()
        result = fn(out, *args)
        return (result[0] if isinstance(result, tuple) else result), out
    return fn(*args)


def _text(value):
    """A struct's character array: bytes in hip-python 7.x, str in 10."""
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _check_hiprtc(result):
    """Check a hipRTC result, raise on error (handles tuple returns)."""
    err = result[0] if isinstance(result, tuple) else result
    if isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(f"hipRTC error: {err}")


# The HIP runtime through ctypes, for the calls that wait. hip-python's
# wrappers call the runtime with the GIL held, so a thread waiting in
# hipEventSynchronize stopped every other Python thread: another thread got
# 6 turns during a 357 ms wait, and 353 when the same call went through
# ctypes, which releases the GIL. It is the library hip-python loaded, by
# the same name, or None if that fails, when the waits go through
# hip-python: correct, but holding the GIL.
#
# The waits are on events, not hipStreamSynchronize: on ROCm 7.1 a thread
# waiting in hipStreamSynchronize (even through ctypes) held the stream, and
# another thread's launch on it took the whole 350 ms wait; waiting on an
# event recorded after the same work, the other thread launched 309 times
# at 60 us each. A copy to pageable host memory waits inside hipMemcpyAsync
# the same way, so a read waits on an event before it copies. An idle event
# wait costs about 8 us (create, record, wait, destroy; 0.7 for the stream).
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_NO_RUNTIME = object()


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                _RUNTIME = _load_runtime() or _NO_RUNTIME
    return None if _RUNTIME is _NO_RUNTIME else _RUNTIME


def _runtime_library():
    """The HIP runtime library hip-python loaded, so the ctypes waits use
    the same one: `hip.chip.DLL` before hip-python 10, `rocm.bindings`'s
    lookup from 10 on, the soname if neither is there."""
    try:
        from hip import chip
    except ImportError:
        try:
            from rocm.bindings.util.paths import get_library_path
        except ImportError:
            return b"libamdhip64.so"
        return get_library_path("amdhip64")
    return getattr(chip, "DLL", b"libamdhip64.so")


def _load_runtime():
    name = _runtime_library()
    try:
        lib = ctypes.CDLL(name.decode() if isinstance(name, bytes) else name)
        lib.hipEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.hipEventSynchronize.restype = ctypes.c_int
    except (OSError, AttributeError):
        return None
    return lib


def _check_code(code):
    if code != 0:
        raise HIPError(hip.hipError_t(code))


def _wait_stream(stream):
    """Wait for everything queued on ``stream`` so far, on an event, with
    the GIL and the stream released: other threads launch meanwhile."""
    _StreamMarker(stream).wait()


def _wait_event(event):
    """Wait for ``event``, with the GIL released."""
    runtime = _runtime()
    if runtime is None:
        _check_hip(hip.hipEventSynchronize(event))
    else:
        _check_code(runtime.hipEventSynchronize(ctypes.c_void_p(int(event))))


class _BoundDevice:
    """Make a device current on this thread for the block, and put back
    whatever was current before: a worker thread starts on device 0."""

    __slots__ = ("_device", "_previous")

    def __init__(self, device):
        self._device = device

    def __enter__(self):
        err, self._previous = hip.hipGetDevice()
        _check_hip(err)
        if self._previous != self._device:
            _check_hip(hip.hipSetDevice(self._device))
        return self

    def __exit__(self, *exc):
        if self._previous != self._device:
            hip.hipSetDevice(self._previous)


class _StreamMarker:
    """An event recorded on a device's stream after the work queued so far,
    waited on from any thread (`Backend.async_marker`). The event sleeps
    the waiting thread (blocking sync), the wait releases the GIL, and the
    event is destroyed by the wait."""

    __slots__ = ("_event",)

    def __init__(self, stream):
        err, self._event = hip.hipEventCreateWithFlags(
            hip.hipEventBlockingSync | hip.hipEventDisableTiming)
        _check_hip(err)
        _check_hip(hip.hipEventRecord(self._event, stream))

    def wait(self):
        try:
            _wait_event(self._event)
        finally:
            hip.hipEventDestroy(self._event)


class _Queue:
    """A device's queue: every launch, fill and copy Tack makes on the device
    goes through this one stream, in order, and returns once it is queued.

    The stream is a blocking one, so it stays ordered with the null stream
    that synchronous HIP calls (and other libraries) use. Memory comes from
    the device's pool in stream order (``hipMallocAsync``/``hipFreeAsync``),
    so an array Python drops while a queued launch still uses it is freed
    after that launch; a device without pools allocates with ``hipMalloc``,
    whose ``hipFree`` waits for the device.
    """

    __slots__ = ("device", "pooled", "stream")

    def __init__(self, device):
        self.device = device
        err, self.stream = hip.hipStreamCreate()
        _check_hip(err)
        err, pools = hip.hipDeviceGetAttribute(
            hip.hipDeviceAttribute_t.hipDeviceAttributeMemoryPoolsSupported, device)
        _check_hip(err)
        self.pooled = bool(pools)
        if self.pooled:
            # Keep freed memory in the pool rather than returning it to the
            # device at every synchronization: a filter allocates and drops
            # dozens of temporaries, and the pool hands them back out at once.
            try:
                err, pool = hip.hipDeviceGetDefaultMemPool(device)
                _check_hip(err)
                _check_hip(hip.hipMemPoolSetAttribute(
                    pool, hip.hipMemPoolAttr.hipMemPoolAttrReleaseThreshold,
                    ctypes.c_uint64(2**64 - 1)))
            except Exception:
                pass                        # the pool releases at syncs; still correct


# One queue per device, made on first use, with the device current.
_QUEUES = {}
_QUEUES_LOCK = threading.Lock()


def _queue(device=None):
    """The queue of ``device``, or of the calling thread's current device."""
    if device is None:
        err, device = hip.hipGetDevice()
        _check_hip(err)
    queue = _QUEUES.get(device)
    if queue is None:
        with _QUEUES_LOCK:
            queue = _QUEUES.get(device)
            if queue is None:
                queue = _QUEUES[device] = _Queue(device)
    return queue


class HIPBuffer(DeviceBuffer):
    """Device-resident buffer backed by a HIP device pointer.

    Data lives on the GPU. Everything that touches it goes through its
    device's stream, in order with the launches: ``from_numpy`` and ``fill``
    are queued and return at once, ``to_numpy`` and ``read_range`` are
    queued and waited for. The memory is stream-ordered too, so an array
    Python drops while a queued launch still uses it is freed after that
    launch.

    A host write from pageable memory is safe to queue: HIP stages a small
    one before returning, and waits for the stream before a large one (on
    ROCm 7.1, 16 KB was staged and 4 MB waited), so the array may go once
    ``from_numpy`` returns.
    """

    backend_name = "hip"
    # Memory handed outside Tack (DLPack, a wrapped pointer) may be read
    # with no Tack call between: a launch that uses it completes before
    # returning, as on Metal and CUDA.
    _shared = False

    def __init__(self, numpy_dtype, shape, queue=None):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        self._queue = queue = queue or _queue()
        # hip-python returns None for a zero-byte request, which a launch
        # cannot pass as a pointer; an empty array gets one byte it never
        # reads, as on CUDA and Level Zero.
        if queue.pooled:
            err, self._device_ptr = hip.hipMallocAsync(max(self._nbytes, 1), queue.stream)
        else:
            err, self._device_ptr = hip.hipMalloc(max(self._nbytes, 1))
        _check_hip(err)
        _check_hip(hip.hipMemsetAsync(self._device_ptr, 0, self._nbytes, queue.stream))

    @property
    def _stream(self):
        return self._queue.stream

    def synchronize(self):
        """Wait for every queued launch, fill and copy on this device."""
        _wait_stream(self._stream)

    # Reading waits for the stream, so `Array.numpy_async` reads in a
    # worker thread, with the buffer's device made current there.
    waits_off_thread = True

    def thread_binding(self):
        return _BoundDevice(self._queue.device)

    def share(self):
        """Hand this memory outside Tack: wait for the queued work now, and
        have every later launch that uses it complete before returning."""
        self._shared = True
        self.synchronize()

    @property
    def address(self) -> int:
        return int(self._device_ptr)

    @property
    def device_ptr(self):
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        # Queued after the launches before it, which may still read the old
        # contents.
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        _check_hip(hip.hipMemcpyAsync(
            self._device_ptr, src, self._nbytes,
            hip.hipMemcpyKind.hipMemcpyHostToDevice, self._stream,
        ))

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        stream = self._stream
        # A copy to pageable memory waits for the stream inside the call,
        # holding the GIL and the stream: wait first, on an event.
        _wait_stream(stream)
        _check_hip(hip.hipMemcpyAsync(
            out, self._device_ptr, self._nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost, stream,
        ))
        _check_hip(hip.hipStreamSynchronize(stream))
        return out

    def read_range(self, start: int, count: int) -> np.ndarray:
        """Copy elements ``[start, start + count)`` to the host.

        One copy of just those bytes, so ``array[i]`` on the host moves one
        element rather than the whole array, queued after the launches
        before it and waited for. The range is checked here because a
        device copy past the allocation is not caught by NumPy.
        """
        itemsize = self._numpy_dtype.itemsize
        if start < 0 or count < 0 or (start + count) * itemsize > self._nbytes:
            raise IndexError(
                f"elements [{start}, {start + count}) are outside a buffer of "
                f"{self._nbytes // itemsize} elements")
        out = np.empty(count, dtype=self._numpy_dtype)
        if count:
            stream = self._stream
            _wait_stream(stream)                  # as in to_numpy
            _check_hip(hip.hipMemcpyAsync(
                out, self.address + start * itemsize, count * itemsize,
                hip.hipMemcpyKind.hipMemcpyDeviceToHost, stream,
            ))
            _check_hip(hip.hipStreamSynchronize(stream))
        return out

    def fill(self, value):
        self.fill_bytes(device_fill.scalar_pattern(value, self._numpy_dtype))

    def fill_bytes(self, pattern):
        base = self.address
        stream = self._stream

        def native(pattern):
            byte = device_fill.uniform_byte(pattern)
            if byte is not None:
                _check_hip(hip.hipMemsetD8Async(self._device_ptr, byte, self._nbytes, stream))
            elif len(pattern) == 2:
                _check_hip(hip.hipMemsetD16Async(self._device_ptr,
                                                 device_fill.as_unsigned(pattern),
                                                 self._nbytes // 2, stream))
            elif len(pattern) == 4:
                _check_hip(hip.hipMemsetD32Async(self._device_ptr,
                                                 device_fill.as_signed(pattern),
                                                 self._nbytes // 4, stream))
            else:
                return False
            return True

        device_fill.fill(
            self._nbytes, pattern, native,
            lambda offset, data: _check_hip(hip.hipMemcpyAsync(
                base + offset, np.frombuffer(bytearray(data), np.uint8), len(data),
                hip.hipMemcpyKind.hipMemcpyHostToDevice, stream)),
            lambda dst, src, size: _check_hip(hip.hipMemcpyAsync(
                base + dst, base + src, size, hip.hipMemcpyKind.hipMemcpyDeviceToDevice,
                stream)))
        return True

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __del__(self):
        if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
            try:
                if self._queue.pooled:
                    # In stream order: after every launch queued so far,
                    # which may still be using this memory.
                    hip.hipFreeAsync(self._device_ptr, self._stream)
                else:
                    hip.hipFree(self._device_ptr)
            except Exception:
                pass


# hipRTC's options. It compiles for the current device's architecture.
# Contraction stays inside one expression, as the language contract permits:
# hipRTC's default (-ffp-contract=fast) fused a product into a later
# statement while another read of the same variable kept it rounded, so on
# AMD GPUs `p - p` was the product's rounding error, not zero. Part of the
# disk cache's target, so that changing an option misses the code objects
# compiled without it.
_HIPRTC_OPTIONS = [b"-ffp-contract=on"]


def _compile_code_object(hip_source: str, func_name: str) -> bytes:
    """Compile HIP C source to a code object via hipRTC."""
    src = hip_source.encode("utf-8")
    err, prog = hiprtc.hiprtcCreateProgram(
        src, f"{func_name}.hip".encode(), 0, [], [],
    )
    _check_hiprtc(err)

    # Compile for the current device architecture
    compile_result = hiprtc.hiprtcCompileProgram(prog, len(_HIPRTC_OPTIONS),
                                                 _HIPRTC_OPTIONS)
    compile_err = compile_result[0] if isinstance(compile_result, tuple) else compile_result

    if compile_err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        err, log_size = hiprtc.hiprtcGetProgramLogSize(prog)
        log = bytearray(log_size)
        hiprtc.hiprtcGetProgramLog(prog, log)
        # NOTE: skip hiprtcDestroyProgram — segfaults in hip-python 7.1
        raise RuntimeError(
            f"hipRTC compilation failed:\n{log.decode(errors='replace')}\n"
            f"Source:\n{hip_source}"
        )

    err, code_size = hiprtc.hiprtcGetCodeSize(prog)
    _check_hiprtc(err)
    code = bytearray(code_size)
    _check_hiprtc(hiprtc.hiprtcGetCode(prog, code))
    # NOTE: skip hiprtcDestroyProgram — segfaults in hip-python 7.1
    return bytes(code)


def _load_kernel(code: bytes, kernel_name: str, param_types, param_is_array,
                 param_is_texture) -> "CompiledHIPKernel":
    """A kernel from its code object, loaded on the current device."""
    err, module = hip.hipModuleLoadData(code)
    _check_hip(err)
    err, func = hip.hipModuleGetFunction(module, kernel_name.encode())
    _check_hip(err)
    compiled = CompiledHIPKernel(module, func, kernel_name, param_types,
                                 param_is_array, param_is_texture)
    compiled.code_object = code
    return compiled


_HIP_CTYPES_MAP = {f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
                   u32: ctypes.c_uint, u64: ctypes.c_ulonglong}


def _set_handle(address, handle):
    """Store a hip-python handle object's pointer at `address`."""
    ctypes.c_void_p.from_address(address).value = handle.as_c_void_p().value


class HIPTextureImage:
    """A HIP 3D array and the texture object that samples it.

    Owned by one `Texture3D`, which uploads its array's data at creation and
    on ``update()``. Nothing else keeps a reference, so the array and the
    texture object are destroyed with the texture: no later texture can be
    served one that was made from someone else's memory.
    """

    def __init__(self, shape_3d):
        W, H, D = shape_3d
        self._shape = shape_3d

        # Create channel format descriptor: 1 channel, 32-bit float
        err, channel_desc = hip.hipCreateChannelDesc(
            32, 0, 0, 0, hip.hipChannelFormatKind.hipChannelFormatKindFloat)
        _check_hip(err)

        # Allocate 3D array
        err, self._array = hip.hipMalloc3DArray(
            channel_desc, hip.hipExtent(width=W, height=H, depth=D), 0)
        _check_hip(err)

        # hip-python 7.x leaves array handles and fixed-size arrays out of
        # its struct wrappers, so those are written through ctypes. Nested
        # structs are views into their parent, which locates them: the
        # `res.array` member holds only the handle, and addressMode[3] is
        # the texture descriptor's first member.
        res_desc = hip.hipResourceDesc()
        res_desc.resType = hip.hipResourceType.hipResourceTypeArray
        _set_handle(res_desc.res.array.as_c_void_p().value, self._array)

        tex_desc = hip.hipTextureDesc()
        address_mode = (ctypes.c_int * 3).from_address(
            tex_desc.as_c_void_p().value)
        address_mode[:] = [int(hip.hipTextureAddressMode.hipAddressModeClamp)] * 3
        tex_desc.filterMode = hip.hipTextureFilterMode.hipFilterModeLinear
        tex_desc.normalizedCoords = 1
        tex_desc.readMode = hip.hipTextureReadMode.hipReadModeElementType

        # Create texture object
        err, self._tex_obj = hip.hipCreateTextureObject(res_desc, tex_desc, None)
        _check_hip(err)

    @property
    def handle(self):
        """The texture object a kernel launch passes, as an address.

        `hipTextureObject_t` is a pointer, and the launch packs it as one.
        """
        return self._tex_obj.as_c_void_p().value

    def upload(self, array):
        """Copy an array's device buffer into the array.

        The texture object reads the array, so it sees the new contents
        without being recreated.
        """
        W, H, D = self._shape
        # Zero-initialized, so both positions start at the origin. Nested
        # structs are set in place (hip-python makes them read-only
        # attributes), and dstArray, which it leaves out, is the handle
        # just before dstPos.
        copy_params = hip.hipMemcpy3DParms()
        src = copy_params.srcPtr
        src.ptr = array._buffer.device_ptr
        src.pitch, src.xsize, src.ysize = W * 4, W, H
        _set_handle(copy_params.dstPos.as_c_void_p().value
                    - ctypes.sizeof(ctypes.c_void_p), self._array)
        ext = copy_params.extent
        ext.width, ext.height, ext.depth = W, H, D
        copy_params.kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice
        _check_hip(hip.hipMemcpy3DAsync(copy_params, array._buffer._stream))

    def __del__(self):
        try:
            if hasattr(self, '_tex_obj'):
                hip.hipDestroyTextureObject(self._tex_obj)
            if hasattr(self, '_array'):
                hip.hipFreeArray(self._array)
        except Exception:
            pass


class CompiledHIPKernel:
    """A compiled HIP kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types, param_is_array,
                 param_is_texture=None):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_array = param_is_array
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self.code_object = None     # what hipRTC made, kept for the disk cache

    def __call__(self, kernel_args: list, loop_end: int, extents=(), limits=None,
                 stream=None):
        """Queue the HIP kernel on ``stream`` and return.

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
                # A HIPTextureImage; hipTextureObject_t is unsigned long long
                arg_values.append(ctypes.c_ulonglong(arg.handle))
            elif is_array:
                arg_values.append(ctypes.c_void_p(int(arg._buffer.device_ptr)))
            else:
                ct = _HIP_CTYPES_MAP[ptype]
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

        # The arguments are copied when the launch is queued, so arg_values
        # may go when this returns.
        _check_hip(hip.hipModuleLaunchKernel(
            self._func,
            *grid,
            *block,
            0, stream,
            arg_ptrs, None,
        ))


class HIPBackend(Backend):
    """HIP GPU backend — device-resident arrays, hipRTC compilation."""

    name = "hip"
    display_name = "HIP"
    supported_dtypes = _HIP_SUPPORTED_DTYPES
    supports_device_reductions = True
    supports_workgroups = True
    device_memory_spaces = frozenset({"hip", "hip_pinned", "hip_managed"})


    def __init__(self):
        _check_hip(hip.hipInit(0))
        err, device = hip.hipGetDevice()
        _check_hip(err)
        self._device = device

        # Whether this device has texture/image hardware. CDNA parts
        # (gfx940/941/942 — MI300 and friends) have none: hipRTC marks
        # tex3D "unavailable: The image/texture API not supported on the
        # device" and refuses to compile. The runtime answers honestly via
        # hipDeviceAttributeImageSupport, so ask rather than assume, and
        # sample in software where the answer is no.
        #
        # gfx90a (MI200) answers yes but cannot filter a 3D texture
        # linearly: hipCreateTextureObject returns hipErrorNotSupported.
        # Point sampling works there, but Texture3D only offers trilinear,
        # so make one small texture and fall back if the device refuses it.
        self._has_image_support = (
            self._query_image_support() and self._linear_3d_textures_work())
        self._max_image_3d = (
            self._query_max_image_3d() if self._has_image_support else 0)
        self._max_launch = self._query_max_launch()
        self._launch_limits = self._query_launch_limits()
        self._queue = _queue(device)

        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledHIPKernel}

    @property
    def _stream(self):
        """The device's queue; None for a backend built without a device
        (the dispatch-path tests make one with ``__new__``), which launches
        on the null stream."""
        queue = getattr(self, "_queue", None)
        return queue.stream if queue is not None else None

    def synchronize(self):
        """Wait for every launch, fill and copy queued on this device. A
        launch's error surfaces here, at the next host access, rather than
        at its call."""
        stream = self._stream
        if stream is not None:
            _wait_stream(stream)

    def async_marker(self):
        """An event after the work queued so far on this device's stream,
        for `tack.sync_async` to wait on in a worker thread."""
        stream = self._stream
        if stream is None:
            return None                      # the null stream: wait inline
        return _StreamMarker(stream)

    def _query_image_support(self) -> bool:
        """True when the device exposes the texture/image API."""
        err, val = hip.hipDeviceGetAttribute(
            hip.hipDeviceAttribute_t.hipDeviceAttributeImageSupport,
            self._device)
        _check_hip(err)
        return bool(val)

    @staticmethod
    def _linear_3d_textures_work() -> bool:
        """True when a linearly filtered 3D texture object can be made."""
        try:
            HIPTextureImage((2, 2, 2))
        except HIPError as e:
            if e.code == hip.hipError_t.hipErrorNotSupported:
                return False
            raise
        return True

    def _query_max_launch(self) -> int:
        """Most iterations one one-dimensional launch can index.

        gridDim.x bounds the block count. On AMD GPUs the dispatch packet
        also stores the grid's total work-item count in 32 bits.
        """
        err, max_blocks = hip.hipDeviceGetAttribute(
            hip.hipDeviceAttribute_t.hipDeviceAttributeMaxGridDimX, self._device)
        _check_hip(err)
        max_blocks = min(int(max_blocks), (2**32 - 1) // WORKGROUP_SIZE)
        return max_blocks * WORKGROUP_SIZE

    def _query_launch_limits(self) -> dict:
        """Grid and block limits per dimension, for multi-dimensional loops.

        The dispatch packet stores each dimension's work-item count in 32
        bits, so the grid limits are capped to keep that below 2**32 at any
        block size up to a full workgroup.
        """
        def attribute(name):
            err, value = hip.hipDeviceGetAttribute(
                getattr(hip.hipDeviceAttribute_t, f"hipDeviceAttribute{name}"), self._device)
            _check_hip(err)
            return int(value)
        cap = (2**32 - 1) // WORKGROUP_SIZE
        return {
            "max_grid": tuple(min(attribute(f"MaxGridDim{a}"), cap) for a in "XYZ"),
            "max_block": tuple(attribute(f"MaxBlockDim{a}") for a in "XYZ"),
        }

    def _query_max_image_3d(self) -> int:
        """Smallest of the three max 3D texture extents, 0 if unreported.

        Only consulted on devices that claim image support; a device that
        reports a non-positive extent is treated as unbounded rather than
        as forbidding every texture, since the extent is advisory and the
        support flag above is the real gate.
        """
        dims = []
        for name in ("hipDeviceAttributeMaxTexture3DWidth",
                     "hipDeviceAttributeMaxTexture3DHeight",
                     "hipDeviceAttributeMaxTexture3DDepth"):
            err, val = hip.hipDeviceGetAttribute(
                getattr(hip.hipDeviceAttribute_t, name), self._device)
            _check_hip(err)
            dims.append(val)
        return min(dims) if all(d > 0 for d in dims) else 0

    def _store_texture_shapes(self, ir_func, effective_args):
        """Record Texture3D extents, falling back to software sampling.

        Mirrors the Level Zero backend: devices with no texture hardware,
        or textures past the device's 3D image limit, are sampled in
        software instead. That changes the generated code, so it is decided
        here — before the variant key is built — rather than at codegen time.
        """
        from tack.lang.array import Texture3D
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                if self.texture_in_hardware(arg.shape_3d):
                    param._texture_shape = arg.shape_3d
                else:
                    param._is_texture = False  # software fallback

    def texture_in_hardware(self, shape_3d) -> bool:
        max_dim = self._max_image_3d
        return self._has_image_support and (
            max_dim == 0 or all(s <= max_dim for s in shape_3d))

    def create_texture_image(self, shape_3d) -> HIPTextureImage:
        return HIPTextureImage(shape_3d)

    def allocate_array(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> HIPBuffer:
        return HIPBuffer(dtype.numpy_dtype, shape, getattr(self, "_queue", None))

    def memory_space(self, ptr) -> str:
        """Query where a pointer resides: 'cpu', 'hip', or 'hip_managed'.

        Uses hipPointerGetAttributes to classify the pointer.

        Returns:
            'hip'         — device memory (hipMalloc)
            'hip_pinned'  — pinned host memory (hipHostMalloc)
            'hip_managed' — unified memory (hipMallocManaged)
            'cpu'         — unregistered host memory

        This answered 'cpu' for every pointer until 2026-08-11, on the first
        machine that could run it. Two mistakes, both swallowed by a broad
        `except` that returned 'cpu': `hipPointerGetAttributes` takes the
        attribute struct as an out-parameter rather than returning it, and
        `hip.hipSuccess` does not exist (it is `hip.hipError_t.hipSuccess`).
        Since `array_from_ptr` validates against this, *every* attempt to
        wrap a HIP device pointer was rejected as host memory — which is the
        DLPack import path and the VTK device interop, both of them.
        """
        addr = as_address(ptr)
        if addr is None:
            # Not an address at all — Metal hands MTLBuffer objects around,
            # and callers pass whatever they have. The range half of that
            # question matters here too: `int()` accepts 1 << 200 happily
            # and the binding then refuses it from inside its own
            # marshalling, which used to be caught and reported as "cpu".
            return "cpu"

        # `attributes` is an out-parameter in hip-python 7.x, which
        # allocates nothing for you and returns only the error code; 10
        # returns it. Getting this wrong is how this function came to answer
        # "cpu" for every pointer it was ever given -- see the note above.
        err, attrs = _call_with_output(hip.hipPointerGetAttributes,
                                       hip.hipPointerAttribute_t,
                                       hip.hipDeviceptr_t(addr))
        if int(err) != int(hip.hipError_t.hipSuccess):
            # Documented outcome for a pointer HIP does not recognise, which
            # is what an ordinary host allocation is.
            return "cpu"

        # hipMemoryTypeUnregistered=0 (plain host memory), Host=1, Device=2,
        # Managed=3, Array=10, Unified=11. Anything not device-addressable
        # falls through to "cpu".
        return {1: "hip_pinned", 2: "hip", 3: "hip_managed",
                11: "hip_managed"}.get(int(attrs.type), "cpu")

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing HIP device pointer without allocating or copying."""
        buf = HIPBuffer.__new__(HIPBuffer)
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        # Store the address as an int. hip-python takes a NumPy integer
        # through the buffer protocol, as the address of the scalar's own
        # storage rather than the value it holds, so every copy through a
        # wrapped np.uint64 failed with hipErrorInvalidValue.
        buf._device_ptr = int(ptr)
        buf._owned = False
        buf._queue = getattr(self, "_queue", None) or _queue()
        buf._shared = True      # someone else's memory: launches on it complete
        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the HIP GPU.

        The IR passes and device compilation run only when this argument
        shape/type combination is new; see `resolve_variant`.
        """
        from tack.lang.array import Texture3D

        variant, effective_args = resolve_variant(
            self, kernel, args, kwargs,
            build=self._build_variant,
            store_texture_shapes=self._store_texture_shapes,
        )
        compiled, pack_info, pack_arrays = variant.payload

        # Loop range comes from the pre-packing IR: packing rewrites the
        # parameter list, and the range expression names the original args.
        kernel_args = [a.array if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end, extents = _get_launch(variant.ir, kernel_args)
        if loop_end <= 0:
            # range(0) runs nothing; do not ask the driver for an empty grid.
            return
        check_launch_size(f"Kernel '{kernel.name}'", loop_end, self._max_launch, self.label)

        # Textures bind their own snapshot, not the array they came from.
        kernel_args = bind_textures(effective_args)

        # Replace scalar args with the packed array buffers. The buffers
        # belong to the variant, so writing them and the launch that reads
        # them happen under its lock (see KernelVariant.dispatch_lock).
        with variant.dispatch_lock:
            if pack_info:
                from tack.lang.ir_pack_scalars import split_args
                from tack.runtime.kernel_utils import _update_pack_arrays
                _update_pack_arrays(pack_arrays, pack_info, effective_args)
                kept_args = split_args(effective_args, pack_info)
                kernel_args = bind_textures(kept_args) + pack_arrays

            compiled(kernel_args, loop_end, extents, self._launch_limits, self._stream)
        # Memory that can be read with no Tack call between -- a wrapped
        # pointer, an exported array -- and a texture, whose image could be
        # destroyed before a queued launch has read it: complete now.
        if any(getattr(getattr(a, '_buffer', None), '_shared', False)
               or isinstance(a, HIPTextureImage) for a in kernel_args):
            self.synchronize()

    def _build_variant(self, ir_func, effective_args):
        """Pack scalars, annotate, compile. Runs once per variant.

        Packing rewrites the parameter list, so it works on its own copy —
        the caller keeps `ir_func` for loop-range resolution.
        """
        from tack.lang.ir_pack_scalars import pack_scalars
        from tack.lang.ir_traversal import clone_ir
        from tack.lang.ir_type_annotate import annotate_types
        from tack.lang.ir_verify import verify_ir
        from tack.runtime.kernel_utils import _create_pack_arrays

        packed = clone_ir(ir_func)
        _, pack_info = pack_scalars(packed, effective_args)
        verify_ir(packed, 'packed')
        annotate_types(packed)
        verify_ir(packed, 'typed')
        compiled = self._compile_kernel(packed)
        pack_arrays = (_create_pack_arrays(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_arrays

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledHIPKernel:
        """Compile Tack IR → HIP C → code object → hipFunction."""
        kernel_name = kernel_entry_name(ir_func.name)
        hip_source = generate_hip_source(ir_func)
        code = _compile_code_object(hip_source, kernel_name)
        param_types = [p.type_annotation for p in ir_func.params]
        param_is_array = [getattr(p, '_is_array', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        return _load_kernel(code, kernel_name, param_types, param_is_array, param_is_texture)

    # ── The disk cache: code objects, loaded as they are ──────────────
    # A code object is machine code for one architecture and its feature
    # flags (gcnArchName, e.g. gfx90a:sramecc+:xnack-), so the target names
    # them, with the device, the ROCm runtime that loads the code, and the
    # hipRTC, hip-python and options that made it. The device also covers
    # the texture fallback, which changes the generated source: an MI300X
    # (gfx942, no texture hardware) and an MI210 have different targets.

    _target = None

    def cache_target(self):
        if self._target is None:
            err, props = _call_with_output(hip.hipGetDeviceProperties,
                                           hip.hipDeviceProp_t, self._device)
            _check_hip(err)
            device_name = _text(props.name)
            arch = _text(props.gcnArchName)
            err, runtime_version = hip.hipRuntimeGetVersion()
            _check_hip(err)
            err, driver_version = hip.hipDriverGetVersion()
            _check_hip(err)
            err, rtc_major, rtc_minor = hiprtc.hiprtcVersion()
            _check_hiprtc(err)
            options = " ".join(o.decode() for o in _HIPRTC_OPTIONS)
            self._target = (f"{device_name} {arch}; runtime {runtime_version}; "
                            f"driver {driver_version}; hiprtc {rtc_major}.{rtc_minor}; "
                            f"hip-python {_HIP_PYTHON_VERSION}; options [{options}]")
        return self._target

    def variant_artifact(self, payload):
        compiled, pack_info, _ = payload
        if compiled.code_object is None:
            raise ValueError("no code object was kept for this kernel")
        return {"code_object": compiled.code_object, "name": compiled._func_name,
                "param_types": compiled._param_types,
                "param_is_array": compiled._param_is_array,
                "param_is_texture": compiled._param_is_texture,
                "pack_info": pack_info}

    def load_variant(self, ir_func, artifact, effective_args):
        from tack.runtime.kernel_utils import _create_pack_arrays
        compiled = _load_kernel(artifact["code_object"], artifact["name"],
                                artifact["param_types"], artifact["param_is_array"],
                                artifact["param_is_texture"])
        pack_info = artifact["pack_info"]
        # The scalar pack buffers are device memory of this process, made
        # from this call's scalars as _build_variant makes them.
        pack_arrays = (_create_pack_arrays(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_arrays

    def reduce_array(self, array, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        from tack.lang.types import f32
        if array.size == 0:
            return empty_reduction(op)
        if array.dtype is not f32:
            return reduce_numpy(array.numpy(), op)
        n = int(np.prod(array.shape))
        check_launch_size(f"Array {op}()", n, self._max_launch, self.label)

        if not hasattr(self, '_reduce_cache'):
            self._reduce_cache = {}
        if op not in self._reduce_cache:
            self._reduce_cache[op] = self._compile_reduce(op)

        func, module = self._reduce_cache[op]

        out_buf = HIPBuffer(np.float32, (1,), getattr(self, "_queue", None))
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

        _check_hip(hip.hipModuleLaunchKernel(
            func, grid_dim, 1, 1, block_dim, 1, 1,
            block_dim * 4, self._stream, args, 0))

        result = out_buf.to_numpy()         # queued after the launch, and waited for
        return float(result[0])

    def _compile_reduce(self, op: str):
        """Compile a HIP reduction kernel for the given op."""
        # HIP device code uses the same syntax as CUDA
        _REDUCE_SRC = {
            "sum": _REDUCE_HIP_SUM,
            "min": _REDUCE_HIP_MIN,
            "max": _REDUCE_HIP_MAX,
        }
        func_names = {
            "sum": "reduce_sum_f32",
            "min": "reduce_min_f32",
            "max": "reduce_max_f32",
        }
        src = _REDUCE_SRC[op]
        code = _compile_code_object(src, func_names[op])
        err, module = hip.hipModuleLoadData(code)
        _check_hip(err)
        err, func = hip.hipModuleGetFunction(module, func_names[op].encode())
        _check_hip(err)
        return func, module

    def __del__(self):
        pass  # HIP context is managed by the runtime

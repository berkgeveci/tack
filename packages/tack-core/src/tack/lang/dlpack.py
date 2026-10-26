"""DLPack support for Tack arrays.

Implements the DLPack protocol (__dlpack__, __dlpack_device__) for
zero-copy tensor exchange with PyTorch, CuPy, JAX, NumPy, VTK, etc.

DLPack spec: https://dmlc.github.io/dlpack/latest/
Python protocol: https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
"""

import ctypes
import itertools

from tack.lang.types import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

# DLPack device types
kDLCPU = 1
kDLCUDA = 2
kDLCUDAManaged = 13
kDLROCM = 10
kDLCUDAHost = 3
kDLROCMHost = 11
kDLMetal = 8
kDLOneAPI = 14

# DLPack data type codes
kDLFloat = 2
kDLInt = 0
kDLUInt = 1

# Map Tack types to DLPack (code, bits, lanes).
# Every dtype a backend accepts as an array belongs here — the narrow ints
# were missing, so tack.i8 and friends round-tripped through numpy fine but
# raised "DLPack does not support dtype" on export.
_DTYPE_TO_DLPACK = {
    f32: (kDLFloat, 32, 1),
    f64: (kDLFloat, 64, 1),
    i8:  (kDLInt, 8, 1),
    i16: (kDLInt, 16, 1),
    i32: (kDLInt, 32, 1),
    i64: (kDLInt, 64, 1),
    u8:  (kDLUInt, 8, 1),
    u16: (kDLUInt, 16, 1),
    u32: (kDLUInt, 32, 1),
    u64: (kDLUInt, 64, 1),
}

# Map DLPack to Tack types
_DLPACK_TO_DTYPE = {
    (kDLFloat, 32): f32,
    (kDLFloat, 64): f64,
    (kDLInt, 8): i8,
    (kDLInt, 16): i16,
    (kDLInt, 32): i32,
    (kDLInt, 64): i64,
    (kDLUInt, 8): u8,
    (kDLUInt, 16): u16,
    (kDLUInt, 32): u32,
    (kDLUInt, 64): u64,
}


# --- DLPack C structures ---

class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", ctypes.c_int32),
        ("device_id", ctypes.c_int32),
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


# DLPack v1.0 adds a versioned managed tensor, which is the only form that
# can say a buffer is read-only. Both directions need it: NumPy will only
# export a read-only array through it, and it is the only way tack can
# say that an array over external memory must not be written.
#
# Note the member order differs from the unversioned struct -- dl_tensor
# moves to the end. Reading one as the other silently misreads every
# array, so the two are never used interchangeably.
class DLPackVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32)]


class DLManagedTensorVersioned(ctypes.Structure):
    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("flags", ctypes.c_uint64),
        ("dl_tensor", DLTensor),
    ]


DLPACK_FLAG_BITMASK_READ_ONLY = 1 << 0


# ── Export lifetime ──────────────────────────────────────────────────
#
# An exported capsule points at the array's memory, so everything backing
# it has to outlive the consumer: the array itself, the DLManagedTensor,
# and the shape/stride arrays. They are pinned here and released when the
# consumer is done.
#
# The key is what makes this work. DLPack hands the deleter the
# DLManagedTensor, not the context, so the producer's bookkeeping key has
# to travel inside it — that is what `manager_ctx` is for. Keying the
# table by anything the deleter cannot recover means nothing is ever
# released and every export leaks, on the device as well as the host.

_prevent_gc = {}

# Starts at 1: 0 round-trips through c_void_p as None and would be
# indistinguishable from "no context".
_ctx_keys = itertools.count(1)

def _c_api(name, restype, *argtypes):
    """``name`` from Python's C API, with a prototype of Tack's own.

    ``ctypes.pythonapi.<name>`` is one object for the whole process, so a
    prototype set on it re-types every other library's calls, and theirs
    Tack's. Warp sets PyCapsule_IsValid's first argument to py_object where
    Tack used c_void_p: imported after Warp, Tack made Warp's capsule
    destructor raise ArgumentError; imported before it, Warp's prototype made
    Tack's destructor find no capsule, and every export dropped unconsumed
    stayed pinned. A prototype of one's own binds the same symbol."""
    return ctypes.PYFUNCTYPE(restype, *argtypes)((name, ctypes.pythonapi))


_PyCapsule_New = _c_api("PyCapsule_New", ctypes.py_object,
                        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)

# Raw pointers rather than py_object throughout: these run during capsule
# teardown, where touching the object's refcount risks resurrecting it.
_PyCapsule_IsValid = _c_api("PyCapsule_IsValid", ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p)
_PyCapsule_GetPointer = _c_api("PyCapsule_GetPointer", ctypes.c_void_p,
                               ctypes.c_void_p, ctypes.c_char_p)
_Py_IncRef = _c_api("Py_IncRef", None, ctypes.py_object)


def _release(managed_ptr, struct, _cast=ctypes.cast, _pointer=ctypes.POINTER,
             _pins=_prevent_gc):
    """Unpin whatever was retained for the export at `managed_ptr`.

    `struct` says which layout to read, since manager_ctx sits at a
    different offset in the two.

    Reached from `_capsule_destructor`, so it too takes what it needs as
    default arguments rather than from module globals; see there.
    """
    if not managed_ptr:
        return
    managed = _cast(managed_ptr, _pointer(struct)).contents
    _pins.pop(managed.manager_ctx, None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter(managed_ptr):
    """Called by the consumer once it has finished with the tensor."""
    _release(managed_ptr, DLManagedTensor)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter_versioned(managed_ptr):
    """As above, for a v1.0 tensor -- a separate function because the
    deleter is handed only the pointer and cannot tell the two apart."""
    _release(managed_ptr, DLManagedTensorVersioned)


def _destroy_capsule(capsule_ptr, _is_valid=_PyCapsule_IsValid,
                     _get_pointer=_PyCapsule_GetPointer, _release=_release,
                     _legacy=DLManagedTensor,
                     _versioned=DLManagedTensorVersioned):
    """Release an export that no consumer ever adopted.

    A consumer that takes the tensor renames the capsule to
    "used_dltensor" and becomes responsible for calling the deleter. If a
    capsule is collected still named "dltensor" nobody adopted it, and
    without this the pinned objects would never be freed.

    Everything it needs is bound as a default argument. A capsule can
    outlive this module -- a consumer that failed part-way may still hold
    one when the interpreter exits -- and by then the module's globals are
    gone, so a plain global lookup here raises NameError. The thunk that
    calls it is kept alive for the same reason; see below.
    """
    if not capsule_ptr:
        return
    if _is_valid(capsule_ptr, b"dltensor"):
        _release(_get_pointer(capsule_ptr, b"dltensor"), _legacy)
    elif _is_valid(capsule_ptr, b"dltensor_versioned"):
        _release(_get_pointer(capsule_ptr, b"dltensor_versioned"), _versioned)


_capsule_destructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(_destroy_capsule)

# C holds these thunks only as raw pointers: the capsule its destructor, a
# consumer the deleter. Each stays callable for as long as a capsule or tensor
# lives, which can be past this module's teardown -- VTK keeps the capsules it
# adopts in a module-level table that is cleared after this one. Clearing the
# module's globals freed the thunk, and the capsule then called freed memory:
# 39_vtk_interop.py segfaulted at exit. One reference each, never released.
for _thunk in (_dlpack_deleter, _dlpack_deleter_versioned, _capsule_destructor):
    _Py_IncRef(_thunk)
del _thunk


# ── Keeping a consumer's error ───────────────────────────────────────
#
# A consumer that refuses a tensor sets its error and then drops the capsule,
# so the capsule's destructor runs with that error pending -- and so may the
# deleter, if a consumer gives up after adopting the tensor. A ctypes thunk
# cannot keep it: whatever Python the thunk runs, even a function that only
# appends to a list, meets the pending error (a C-API call raises it, and a
# call that returns finds "a result with an exception set"), ctypes reports
# that as unraisable and clears it, and the consumer returns NULL with no
# error set. np.from_dlpack of a CUDA array raised "SystemError:
# <built-in function from_dlpack> returned NULL without setting an
# exception" in place of NumPy's "Unsupported device in DLTensor."
# (mustafar-linux, Tack's CUDA, HIP and Level Zero arrays alike).
#
# So C is handed native functions that set the error aside, call the thunk
# and put the error back, compiled with llvmlite where it is installed (the
# cpu extra); without it the thunks are handed over as before.

_DELETER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_natives = None         # id(thunk) -> its native wrapper, made at the first export


def _natively(thunk):
    """``thunk`` behind a native function that keeps a pending error; ``thunk``
    itself without llvmlite."""
    global _natives
    if _natives is None:
        _natives = _wrap_natively((_dlpack_deleter, _dlpack_deleter_versioned,
                                   _capsule_destructor))
    return _natives.get(id(thunk), thunk)


def _wrap_natively(thunks):
    """``{id(thunk): native wrapper}`` for ctypes thunks taking one pointer (a
    function pointer is not hashable; the thunks live as long as the process,
    see above), or ``{}`` without llvmlite. The wrappers' code is never freed
    either."""
    try:
        import llvmlite.binding as llvm
        import llvmlite.ir as ll
    except ImportError:
        return {}
    import sys

    def c_api(name):
        return ctypes.cast(getattr(ctypes.pythonapi, name), ctypes.c_void_p).value

    pointer = ll.IntType(8).as_pointer()
    address = ll.IntType(8 * ctypes.sizeof(ctypes.c_void_p))
    void = ll.VoidType()

    def call(builder, function, result, args):
        ftype = ll.FunctionType(result, [a.type for a in args])
        return builder.call(builder.inttoptr(ll.Constant(address, function),
                                             ftype.as_pointer()), args)

    module = ll.Module("tack_dlpack")
    module.triple = llvm.get_process_triple()
    for k, thunk in enumerate(thunks):
        function = ll.Function(module, ll.FunctionType(void, [pointer]), f"tack_dlpack_{k}")
        b = ll.IRBuilder(function.append_basic_block())
        target = ctypes.cast(thunk, ctypes.c_void_p).value
        # The GIL first: a capsule's destructor runs with it, but Tack's own
        # import calls the deleter through ctypes, which releases it.
        gil = call(b, c_api("PyGILState_Ensure"), ll.IntType(32), [])
        if sys.version_info >= (3, 12):
            error = call(b, c_api("PyErr_GetRaisedException"), pointer, [])
            call(b, target, void, [function.args[0]])
            call(b, c_api("PyErr_SetRaisedException"), void, [error])
        else:
            slots = [b.alloca(pointer) for _ in range(3)]
            call(b, c_api("PyErr_Fetch"), void, slots)
            call(b, target, void, [function.args[0]])
            call(b, c_api("PyErr_Restore"), void, [b.load(s) for s in slots])
        call(b, c_api("PyGILState_Release"), void, [gil])
        b.ret_void()

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(str(module)), machine)
    engine.finalize_object()
    _Py_IncRef(engine)
    return {id(thunk): _DELETER(engine.get_function_address(f"tack_dlpack_{k}"))
            for k, thunk in enumerate(thunks)}


def _get_device_info(array):
    """Determine DLPack device type and data pointer for an array."""
    buf = array._buffer
    cls_name = type(buf).__name__

    if cls_name == "NumpyBuffer":
        # CPU: numpy array data pointer
        data_ptr = buf._data.ctypes.data
        return kDLCPU, 0, data_ptr

    if cls_name == "MetalBuffer":
        # Metal: shared memory — expose as CPU since it's unified memory
        # and the numpy view points into the same physical memory. The consumer
        # reads it at once, and may keep reading it after later launches.
        buf.share()
        data_ptr = buf._view.ctypes.data
        return kDLCPU, 0, data_ptr

    if cls_name == "CUDABuffer":
        # The consumer reads it at once, on its own stream, and may keep
        # reading it after later launches: complete the queued ones now and
        # every later one that uses it.
        buf.share()
        data_ptr = int(buf._device_ptr)
        return kDLCUDA, array._backend.ordinal, data_ptr

    if cls_name == "HIPBuffer":
        # As for CUDA: complete the queued launches now, and every later
        # one that uses it.
        buf.share()
        data_ptr = int(buf._device_ptr)
        return kDLROCM, array._backend.ordinal, data_ptr

    if cls_name == "L0Buffer":
        # As for CUDABuffer: the consumer reads it at once, on its own
        # queue, and may keep reading it after later launches.
        buf.share()
        data_ptr = buf._device_ptr.value if hasattr(buf._device_ptr, 'value') else int(buf._device_ptr)
        return kDLOneAPI, array._backend.ordinal, data_ptr

    if cls_name == "WarpBuffer":
        # As for CUDABuffer: complete the queued launches now, and every
        # later one that uses it.
        buf.share()
        device = array._backend.wp_device
        if device.is_cuda:
            return kDLCUDA, device.ordinal, buf.address
        return kDLCPU, 0, buf.address

    raise RuntimeError(f"DLPack not supported for buffer type: {cls_name}")


def array_to_dlpack(array, versioned=False):
    """Create a DLPack capsule from a Tack array.

    With `versioned`, produces a v1.0 DLManagedTensorVersioned in a
    'dltensor_versioned' capsule; otherwise the legacy DLManagedTensor in
    a 'dltensor' one. Only the versioned form can carry the read-only
    flag, so a non-writable array exported the legacy way arrives as
    writable -- the legacy protocol simply has no way to say otherwise.
    """
    device_type, device_id, data_ptr = _get_device_info(array)

    dl_dtype = _DTYPE_TO_DLPACK.get(array.scalar_dtype)
    if dl_dtype is None:
        raise TypeError(f"DLPack does not support dtype: {array.dtype}")
    code, bits, lanes = dl_dtype

    shape = array._host_shape
    ndim = len(shape)
    shape_arr = (ctypes.c_int64 * ndim)(*shape)

    # Contiguous C-order strides (in elements, not bytes)
    strides_arr = (ctypes.c_int64 * ndim)()
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides_arr[i] = stride
        stride *= shape[i]

    if versioned:
        managed = DLManagedTensorVersioned()
        managed.version = DLPackVersion(major=1, minor=0)
        managed.flags = 0 if array._writable else DLPACK_FLAG_BITMASK_READ_ONLY
        capsule_name = b"dltensor_versioned"
        deleter = _dlpack_deleter_versioned
    else:
        managed = DLManagedTensor()
        capsule_name = b"dltensor"
        deleter = _dlpack_deleter

    managed.dl_tensor.data = ctypes.c_void_p(data_ptr)
    managed.dl_tensor.device = DLDevice(device_type=device_type, device_id=device_id)
    managed.dl_tensor.ndim = ndim
    managed.dl_tensor.dtype = DLDataType(code=code, bits=bits, lanes=lanes)
    managed.dl_tensor.shape = shape_arr
    managed.dl_tensor.strides = strides_arr
    managed.dl_tensor.byte_offset = 0

    # The key travels in manager_ctx, which is the only thing the deleter
    # gets back. Pin everything the capsule points at under that key.
    key = next(_ctx_keys)
    managed.manager_ctx = ctypes.c_void_p(key)
    managed.deleter = _natively(deleter)
    _prevent_gc[key] = (array, managed, shape_arr, strides_arr)

    # addressof, not byref: byref yields a temporary whose lifetime is not
    # tied to the capsule. `managed` itself is kept alive by _prevent_gc.
    return _PyCapsule_New(
        ctypes.addressof(managed), capsule_name,
        ctypes.cast(_natively(_capsule_destructor), ctypes.c_void_p),
    )


def dlpack_device(array):
    """Return (device_type, device_id) tuple for DLPack protocol."""
    device_type, device_id, _ = _get_device_info(array)
    return (device_type, device_id)


# ── Import ───────────────────────────────────────────────────────────
#
# The mirror of the export side. A consumer that adopts a tensor takes on
# calling its deleter, so an imported array has to hold the capsule and
# release it when the array dies -- otherwise the producer frees memory
# the array is still pointing at.

_PyCapsule_SetName = _c_api("PyCapsule_SetName", ctypes.c_int, ctypes.py_object, ctypes.c_char_p)

# DLPack device type -> the memory spaces a backend reports for it. An
# imported tensor has to land on a backend that can actually address it.
#
# kDLCPU listed "metal" once, on the reasoning that Apple silicon memory is
# unified so a host pointer is reachable from the GPU. Reachable is not the
# same as wrappable: a MetalBuffer needs an MTLBuffer, and turning a host
# pointer into one without copying needs `newBufferWithBytesNoCopy`, which
# wants a page-aligned address. Host allocations are page-aligned only by
# accident of size -- measured on an M1 Max, a 4000-byte numpy array is and
# a 256-byte one is not -- so supporting it would work or fail depending on
# how big the array happened to be. The entry never worked in any case:
# `dlpack_to_array` hands `wrap_ptr` an integer address and Metal's expects
# an MTLBuffer object, so every such import raised AttributeError.
#
# kDLMetal is absent for a related reason. Its `data` is an opaque
# id<MTLBuffer> handle, not an address, and `byte_offset` locates the
# tensor inside that buffer. Wrapping one needs the handle turned back
# into a PyObjC object and an offset carried through MetalBuffer and the
# argument-buffer binding, neither of which exists; listing it sent the
# handle to `wrap_ptr` as an integer, which refuses it with a TypeError.
_DEVICE_BACKENDS = {
    kDLCPU: ("cpu",),
    kDLCUDAHost: ("cpu",),
    kDLROCMHost: ("cpu",),
    kDLCUDA: ("cuda",),
    kDLCUDAManaged: ("cuda",),
    kDLROCM: ("hip",),
    kDLOneAPI: ("level_zero",),
}


def _tensor_device(kind, device_id):
    """The backend of the Tack device a DLPack tensor's memory is on: ``kind``
    is the backend name `_DEVICE_BACKENDS` gives, ``device_id`` the
    tensor's device number (meaningless for host memory)."""
    from tack.runtime.dispatch import device
    spec = kind if kind == "cpu" else f"{kind}:{device_id}"
    return device(spec)._backend


def _request_capsule(source):
    """Get a capsule from *source*, preferring the versioned protocol.

    Asking for v1.0 first matters: NumPy refuses to export a read-only array
    over the legacy protocol, because that protocol has no way to say so. A
    producer that predates v1.0 raises TypeError on max_version, so fall back.
    """
    if not hasattr(source, "__dlpack__"):
        return source
    try:
        return source.__dlpack__(max_version=(1, 0))
    except TypeError:
        return source.__dlpack__()


class _CapsuleHold:
    """Keeps an adopted DLPack tensor alive, and releases it exactly once.

    Attached to the imported array, so the producer is told the memory is
    free the moment the array is collected -- and not before.
    """

    __slots__ = ("_capsule", "_managed_ptr", "_released", "_struct")

    def __init__(self, capsule, managed_ptr, struct=None):
        self._capsule = capsule
        self._managed_ptr = managed_ptr
        self._struct = struct or DLManagedTensor
        self._released = False

    def release(self):
        if self._released:
            return
        self._released = True
        managed = ctypes.cast(
            self._managed_ptr, ctypes.POINTER(self._struct)).contents
        if managed.deleter:
            managed.deleter(self._managed_ptr)
        self._capsule = None

    def __del__(self):
        try:
            self.release()
        except Exception:
            # Interpreter teardown can pull ctypes out from under us; a
            # failed release at exit is not worth an unraisable traceback.
            pass


def dlpack_to_array(source, writable=True, device=None):
    """Wrap a DLPack tensor as a Tack array, without copying.

    `source` may be a capsule or anything implementing ``__dlpack__`` -- a
    CuPy array, a PyTorch tensor, a VTK array via
    ``vtkmodules.util.dlpack_support``.

    The tensor is held for as long as the array lives, so the source may be
    dropped immediately. The array does not copy, so the two share memory
    and a write through either is visible to the other.

    ``device`` is where the array goes. By default it is the default
    device when that can wrap the tensor, else the tensor's own device, as
    the array API's ``from_dlpack`` keeps an array on its device: CUDA
    memory imported with the CPU as default is a CUDA array, not a copy.

    Raises RuntimeError if ``device`` is given and cannot wrap the tensor --
    CUDA memory imported onto the CPU device. Metal refuses host tensors for
    a different reason, and says which: it can read that memory, it just
    cannot make an MTLBuffer out of it.
    """
    from tack.lang.array import _target, array_from_ptr
    from tack.runtime.dispatch import get_backend

    capsule = _request_capsule(source)

    capsule_ptr = ctypes.cast(id(capsule), ctypes.c_void_p)
    if _PyCapsule_IsValid(capsule_ptr, b"dltensor_versioned"):
        name, used_name = b"dltensor_versioned", b"used_dltensor_versioned"
        struct = DLManagedTensorVersioned
    elif _PyCapsule_IsValid(capsule_ptr, b"dltensor"):
        name, used_name = b"dltensor", b"used_dltensor"
        struct = DLManagedTensor
    else:
        raise ValueError(
            "expected an unconsumed DLPack capsule; this one has already "
            "been taken by another consumer")

    managed_ptr = _PyCapsule_GetPointer(capsule_ptr, name)
    managed = ctypes.cast(managed_ptr, ctypes.POINTER(struct)).contents
    tensor = managed.dl_tensor

    if struct is DLManagedTensorVersioned and \
            managed.flags & DLPACK_FLAG_BITMASK_READ_ONLY:
        # The producer says this memory must not be written. Honour it rather
        # than handing back an array whose writes go somewhere unexpected.
        writable = False

    if not tensor.data:
        raise ValueError("the tensor has a null data pointer")

    dtype = _DLPACK_TO_DTYPE.get((tensor.dtype.code, tensor.dtype.bits))
    if dtype is None:
        raise TypeError(
            f"no Tack dtype for DLPack (code={tensor.dtype.code}, "
            f"bits={tensor.dtype.bits})")
    if tensor.dtype.lanes != 1:
        raise TypeError(
            f"vectorized DLPack dtypes are not supported (lanes={tensor.dtype.lanes})")

    shape = tuple(tensor.shape[i] for i in range(tensor.ndim))

    # Only C-contiguous tensors can be wrapped. Anything else would be read
    # with the wrong stride, which produces plausible-looking wrong numbers.
    if tensor.strides:
        expected, stride = [0] * tensor.ndim, 1
        for i in range(tensor.ndim - 1, -1, -1):
            expected[i] = stride
            stride *= shape[i]
        actual = [tensor.strides[i] for i in range(tensor.ndim)]
        if actual != expected:
            raise ValueError(
                f"only C-contiguous tensors can be wrapped without copying; "
                f"strides {tuple(actual)} != {tuple(expected)}")

    allowed = _DEVICE_BACKENDS.get(tensor.device.device_type)
    if tensor.device.device_type == kDLMetal:
        raise ValueError(
            "Metal DLPack tensors (kDLMetal) cannot be imported: Tack has "
            "no way yet to wrap the MTLBuffer handle they carry. Move the "
            "tensor to host memory in the producing library, then import "
            "it with copy=True.")
    if allowed is None:
        raise ValueError(
            f"unsupported DLPack device type {tensor.device.device_type}")
    if device is not None:
        backend = _target(device)
    else:
        backend = get_backend()
        if backend.name not in allowed or (
                backend.multi_device and backend.ordinal != tensor.device.device_id):
            backend = _tensor_device(allowed[0], tensor.device.device_id)
    if backend.name not in allowed:
        message = (
            f"the tensor is on a device the '{backend.name}' device cannot "
            f"wrap without copying (DLPack device type "
            f"{tensor.device.device_type}). Import it on its own device with "
            f"tack.from_dlpack(x, device='{allowed[0]}'), or pass copy=True.")
        if backend.dlpack_refusal_note:
            message += f" {backend.dlpack_refusal_note}"
        raise RuntimeError(message)

    array = array_from_ptr(
        (tensor.data or 0) + tensor.byte_offset, shape, dtype, device=backend,
        writable=writable)

    # Adopt the capsule: rename it so nobody else releases it, and attach the
    # hold to the array so the producer is told when the array is gone.
    _PyCapsule_SetName(capsule, used_name)
    array._dlpack_hold = _CapsuleHold(capsule, managed_ptr, struct)
    return array

"""DLPack support for Tack fields.

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
# Every dtype a backend accepts as a field belongs here — the narrow ints
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
# say that a field over external memory must not be written.
#
# Note the field order differs from the unversioned struct -- dl_tensor
# moves to the end. Reading one as the other silently misreads every
# field, so the two are never used interchangeably.
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
# An exported capsule points at the field's memory, so everything backing
# it has to outlive the consumer: the field itself, the DLManagedTensor,
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

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

# Raw pointers rather than py_object throughout: these run during capsule
# teardown, where touching the object's refcount risks resurrecting it.
_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.restype = ctypes.c_int
_PyCapsule_IsValid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


def _release(managed_ptr, struct):
    """Unpin whatever was retained for the export at `managed_ptr`.

    `struct` says which layout to read, since manager_ctx sits at a
    different offset in the two.
    """
    if not managed_ptr:
        return
    managed = ctypes.cast(managed_ptr, ctypes.POINTER(struct)).contents
    _prevent_gc.pop(managed.manager_ctx, None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter(managed_ptr):
    """Called by the consumer once it has finished with the tensor."""
    _release(managed_ptr, DLManagedTensor)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter_versioned(managed_ptr):
    """As above, for a v1.0 tensor -- a separate function because the
    deleter is handed only the pointer and cannot tell the two apart."""
    _release(managed_ptr, DLManagedTensorVersioned)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _capsule_destructor(capsule_ptr):
    """Release an export that no consumer ever adopted.

    A consumer that takes the tensor renames the capsule to
    "used_dltensor" and becomes responsible for calling the deleter. If a
    capsule is collected still named "dltensor" nobody adopted it, and
    without this the pinned objects would never be freed.
    """
    if not capsule_ptr:
        return
    if _PyCapsule_IsValid(capsule_ptr, b"dltensor"):
        _release(_PyCapsule_GetPointer(capsule_ptr, b"dltensor"),
                 DLManagedTensor)
    elif _PyCapsule_IsValid(capsule_ptr, b"dltensor_versioned"):
        _release(_PyCapsule_GetPointer(capsule_ptr, b"dltensor_versioned"),
                 DLManagedTensorVersioned)


def _get_device_info(field):
    """Determine DLPack device type and data pointer for a field."""
    buf = field._buffer
    cls_name = type(buf).__name__

    if cls_name == "NumpyBuffer":
        # CPU: numpy array data pointer
        data_ptr = buf._data.ctypes.data
        return kDLCPU, 0, data_ptr

    if cls_name == "MetalBuffer":
        # Metal: shared memory — expose as CPU since it's unified memory
        # and the numpy view points into the same physical memory
        data_ptr = buf._view.ctypes.data
        return kDLCPU, 0, data_ptr

    if cls_name == "CUDABuffer":
        data_ptr = int(buf._device_ptr)
        return kDLCUDA, 0, data_ptr

    if cls_name == "HIPBuffer":
        data_ptr = int(buf._device_ptr)
        return kDLROCM, 0, data_ptr

    if cls_name == "L0Buffer":
        data_ptr = buf._device_ptr.value if hasattr(buf._device_ptr, 'value') else int(buf._device_ptr)
        return kDLOneAPI, 0, data_ptr

    raise RuntimeError(f"DLPack not supported for buffer type: {cls_name}")


def field_to_dlpack(field, versioned=False):
    """Create a DLPack capsule from a Tack field.

    With `versioned`, produces a v1.0 DLManagedTensorVersioned in a
    'dltensor_versioned' capsule; otherwise the legacy DLManagedTensor in
    a 'dltensor' one. Only the versioned form can carry the read-only
    flag, so a non-writable field exported the legacy way arrives as
    writable -- the legacy protocol simply has no way to say otherwise.
    """
    device_type, device_id, data_ptr = _get_device_info(field)

    dl_dtype = _DTYPE_TO_DLPACK.get(field.dtype)
    if dl_dtype is None:
        raise TypeError(f"DLPack does not support dtype: {field.dtype}")
    code, bits, lanes = dl_dtype

    ndim = len(field.shape)
    shape_arr = (ctypes.c_int64 * ndim)(*field.shape)

    # Contiguous C-order strides (in elements, not bytes)
    strides_arr = (ctypes.c_int64 * ndim)()
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides_arr[i] = stride
        stride *= field.shape[i]

    if versioned:
        managed = DLManagedTensorVersioned()
        managed.version = DLPackVersion(major=1, minor=0)
        managed.flags = 0 if field._writable else DLPACK_FLAG_BITMASK_READ_ONLY
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
    managed.deleter = deleter
    _prevent_gc[key] = (field, managed, shape_arr, strides_arr)

    # addressof, not byref: byref yields a temporary whose lifetime is not
    # tied to the capsule. `managed` itself is kept alive by _prevent_gc.
    return _PyCapsule_New(
        ctypes.addressof(managed), capsule_name,
        ctypes.cast(_capsule_destructor, ctypes.c_void_p),
    )


def dlpack_device(field):
    """Return (device_type, device_id) tuple for DLPack protocol."""
    device_type, device_id, _ = _get_device_info(field)
    return (device_type, device_id)


# ── Import ───────────────────────────────────────────────────────────
#
# The mirror of the export side. A consumer that adopts a tensor takes on
# calling its deleter, so an imported field has to hold the capsule and
# release it when the field dies -- otherwise the producer frees memory
# the field is still pointing at.

_PyCapsule_SetName = ctypes.pythonapi.PyCapsule_SetName
_PyCapsule_SetName.restype = ctypes.c_int
_PyCapsule_SetName.argtypes = [ctypes.py_object, ctypes.c_char_p]

# DLPack device type -> the memory spaces a backend reports for it. An
# imported tensor has to land on a backend that can actually address it.
_DEVICE_BACKENDS = {
    kDLCPU: ("cpu", "metal"),
    kDLCUDAHost: ("cpu",),
    kDLROCMHost: ("cpu",),
    kDLMetal: ("metal",),
    kDLCUDA: ("cuda",),
    kDLCUDAManaged: ("cuda",),
    kDLROCM: ("hip",),
    kDLOneAPI: ("level_zero",),
}


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

    Attached to the imported field, so the producer is told the memory is
    free the moment the field is collected -- and not before.
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


def dlpack_to_field(source, writable=True):
    """Wrap a DLPack tensor as a Tack field, without copying.

    `source` may be a capsule or anything implementing ``__dlpack__`` -- a
    CuPy array, a PyTorch tensor, a VTK array via
    ``vtkmodules.util.dlpack_support``.

    The tensor is held for as long as the field lives, so the source may be
    dropped immediately. The field does not copy, so the two share memory
    and a write through either is visible to the other.

    Raises RuntimeError if the tensor lives somewhere the active backend
    cannot address -- importing CUDA memory while running on the CPU
    backend is a mistake, not something to paper over with a copy.
    """
    from tack.lang.field import field_from_ptr
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
        # than handing back a field whose writes go somewhere unexpected.
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

    backend = get_backend()
    allowed = _DEVICE_BACKENDS.get(tensor.device.device_type)
    if allowed is None:
        raise ValueError(
            f"unsupported DLPack device type {tensor.device.device_type}")
    if backend.name not in allowed:
        raise RuntimeError(
            f"the tensor is on a device the '{backend.name}' backend cannot "
            f"address (DLPack device type {tensor.device.device_type}). "
            f"Initialize a backend from {allowed}, or copy the data yourself.")

    field = field_from_ptr(
        (tensor.data or 0) + tensor.byte_offset, dtype, shape, writable=writable)

    # Adopt the capsule: rename it so nobody else releases it, and attach the
    # hold to the field so the producer is told when the field is gone.
    _PyCapsule_SetName(capsule, used_name)
    field._dlpack_hold = _CapsuleHold(capsule, managed_ptr, struct)
    return field

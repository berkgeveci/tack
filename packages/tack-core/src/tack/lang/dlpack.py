"""DLPack support for Tack fields.

Implements the DLPack protocol (__dlpack__, __dlpack_device__) for
zero-copy tensor exchange with PyTorch, CuPy, JAX, NumPy, VTK, etc.

DLPack spec: https://dmlc.github.io/dlpack/latest/
Python protocol: https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
"""

import ctypes
import itertools

import numpy as np

from tack.lang.types import f32, f64, i8, i16, i32, i64, u8, u16, u32, u64

# DLPack device types
kDLCPU = 1
kDLCUDA = 2
kDLCUDAManaged = 13
kDLROCM = 10
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


def _release(managed_ptr):
    """Unpin whatever was retained for the export at `managed_ptr`."""
    if not managed_ptr:
        return
    managed = ctypes.cast(managed_ptr,
                          ctypes.POINTER(DLManagedTensor)).contents
    _prevent_gc.pop(managed.manager_ctx, None)


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter(managed_ptr):
    """Called by the consumer once it has finished with the tensor."""
    _release(managed_ptr)


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
        _release(_PyCapsule_GetPointer(capsule_ptr, b"dltensor"))


def _get_device_info(field):
    """Determine DLPack device type and data pointer for a field."""
    buf = field._buffer
    cls_name = type(buf).__name__

    if cls_name == "NumpyBuffer":
        # CPU: numpy array data pointer
        data_ptr = buf._data.ctypes.data
        return kDLCPU, 0, data_ptr

    elif cls_name == "MetalBuffer":
        # Metal: shared memory — expose as CPU since it's unified memory
        # and the numpy view points into the same physical memory
        data_ptr = buf._view.ctypes.data
        return kDLCPU, 0, data_ptr

    elif cls_name == "CUDABuffer":
        data_ptr = int(buf._device_ptr)
        return kDLCUDA, 0, data_ptr

    elif cls_name == "HIPBuffer":
        data_ptr = int(buf._device_ptr)
        return kDLROCM, 0, data_ptr

    elif cls_name == "L0Buffer":
        data_ptr = buf._device_ptr.value if hasattr(buf._device_ptr, 'value') else int(buf._device_ptr)
        return kDLOneAPI, 0, data_ptr

    else:
        raise RuntimeError(f"DLPack not supported for buffer type: {cls_name}")


def field_to_dlpack(field):
    """Create a DLPack capsule from a Tack field.

    Returns a PyCapsule with name 'dltensor' containing a DLManagedTensor.
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

    managed = DLManagedTensor()
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
    managed.deleter = _dlpack_deleter
    _prevent_gc[key] = (field, managed, shape_arr, strides_arr)

    # addressof, not byref: byref yields a temporary whose lifetime is not
    # tied to the capsule. `managed` itself is kept alive by _prevent_gc.
    return _PyCapsule_New(
        ctypes.addressof(managed), b"dltensor",
        ctypes.cast(_capsule_destructor, ctypes.c_void_p),
    )


def dlpack_device(field):
    """Return (device_type, device_id) tuple for DLPack protocol."""
    device_type, device_id, _ = _get_device_info(field)
    return (device_type, device_id)

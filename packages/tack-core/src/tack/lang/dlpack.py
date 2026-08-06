"""DLPack support for Tack fields.

Implements the DLPack protocol (__dlpack__, __dlpack_device__) for
zero-copy tensor exchange with PyTorch, CuPy, JAX, NumPy, VTK, etc.

DLPack spec: https://dmlc.github.io/dlpack/latest/
Python protocol: https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
"""

import ctypes
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


# Prevent GC of the DLManagedTensor and its dependencies
_prevent_gc = {}
_capsule_counter = 0


@ctypes.CFUNCTYPE(None, ctypes.c_void_p)
def _dlpack_deleter(ptr):
    """Called when the consumer releases the DLPack capsule."""
    # Remove from the prevent-GC set
    _prevent_gc.pop(id(ptr), None)


class DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
    ]


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
    global _capsule_counter

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
    managed.manager_ctx = None
    managed.deleter = _dlpack_deleter

    # Prevent GC of the field, managed tensor, shape/strides arrays
    key = _capsule_counter
    _capsule_counter += 1
    _prevent_gc[key] = (field, managed, shape_arr, strides_arr)

    # Create PyCapsule
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = ctypes.py_object
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    capsule = PyCapsule_New(ctypes.byref(managed), b"dltensor", None)
    return capsule


def dlpack_device(field):
    """Return (device_type, device_id) tuple for DLPack protocol."""
    device_type, device_id, _ = _get_device_info(field)
    return (device_type, device_id)

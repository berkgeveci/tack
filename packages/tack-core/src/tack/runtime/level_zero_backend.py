"""Tack Level Zero backend — compiles kernels via ocloc and dispatches on Intel GPUs.

Pipeline:
    Tack IR → OpenCL C source → libocloc (SPIR-V) → zeModuleCreate → zeKernelCreate
           → zeCommandListAppendLaunchKernel

Arrays are device-resident: ``tack.empty()`` allocates a device buffer via
``zeMemAllocDevice``.  Transfers are explicit:

    tack.copyto(a, arr)     # host → device (zeCommandListAppendMemoryCopy)
    arr = a.numpy()         # device → host (zeCommandListAppendMemoryCopy)

No per-dispatch copies — data stays on the GPU between kernel calls.

Requires: libze_loader.so (Level Zero runtime), libocloc.so (Intel offline compiler).
No Python packages needed — uses ctypes directly.
"""

import ctypes
import ctypes.util
import math
import threading

import numpy as np

from tack.codegen.reductions import array_reduction_source
from tack.lang import ir
from tack.lang.array import DeviceBuffer
from tack.lang.parallel_dims import launch_geometry
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import (
    WORKGROUP_SIZE,
    check_workgroup_launch,
    requires_full_workgroups,
)
from tack.runtime import device_fill
from tack.runtime.gpu_backend import GPUBackend
from tack.runtime.kernel_utils import (
    as_address,
    check_launch_size,
    new_kernel_cache,
)
from tack.runtime.reductions import REDUCTION_IDENTITIES, empty_reduction, reduce_numpy

_L0_SUPPORTED_DTYPES = frozenset({i8, u8, i16, u16, i32, u32, i64, u64, f32, f64})
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.opencl_gen import generate_opencl_source

# ---------------------------------------------------------------------------
# Reduce kernel sources (OpenCL C)
# ---------------------------------------------------------------------------
_REDUCE_OCL_SUM = array_reduction_source('opencl', 'sum')
_REDUCE_OCL_MIN = array_reduction_source('opencl', 'min')
_REDUCE_OCL_MAX = array_reduction_source('opencl', 'max')
# ---------------------------------------------------------------------------
# Numpy dtype mapping
# ---------------------------------------------------------------------------
_NUMPY_DTYPE = {
    f32: np.float32, f64: np.float64,
    i32: np.int32, i64: np.int64,
    u32: np.uint32, u64: np.uint64,
}

# ---------------------------------------------------------------------------
# Level Zero constants
# ---------------------------------------------------------------------------
ZE_RESULT_SUCCESS = 0

# Structure types
ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES = 0x01
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0x0d
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x0e
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0x0f
ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b
ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES = 0x17
ZE_MEMORY_TYPE_DEVICE = 2
ZE_MEMORY_TYPE_SHARED = 3
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x03
ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x04
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x06
ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES = 0x07
ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT = 0x1000f

# Module format
ZE_MODULE_FORMAT_IL_SPIRV = 0
ZE_MODULE_FORMAT_NATIVE = 1

# Command queue mode and flags
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1
ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS = 2
ZE_RESULT_ERROR_UNSUPPORTED_FEATURE = 0x78000003

# Freeing memory after the work queued so far (ZE_extension_memory_free_policies)
ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC = 0x1000a
ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE = 0x2

# Events the host waits on (`_ListMarker`)
ZE_STRUCTURE_TYPE_EVENT_POOL_DESC = 0x10
ZE_STRUCTURE_TYPE_EVENT_DESC = 0x11
ZE_EVENT_POOL_FLAG_HOST_VISIBLE = 0x1
ZE_EVENT_SCOPE_FLAG_HOST = 0x4

# Command queue group property flags
ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE = 0x1

# Image constants
ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES = 0x05
ZE_STRUCTURE_TYPE_IMAGE_DESC = 0x19
ZE_IMAGE_TYPE_3D = 4
ZE_IMAGE_FORMAT_LAYOUT_32 = 2       # single 32-bit channel
ZE_IMAGE_FORMAT_TYPE_FLOAT = 1
ZE_IMAGE_FORMAT_SWIZZLE_R = 0
ZE_IMAGE_FORMAT_SWIZZLE_0 = 4
ZE_IMAGE_FORMAT_SWIZZLE_1 = 5

# Handle types (opaque pointers)
ze_driver_handle_t = ctypes.c_void_p
ze_device_handle_t = ctypes.c_void_p
ze_context_handle_t = ctypes.c_void_p
ze_command_queue_handle_t = ctypes.c_void_p
ze_command_list_handle_t = ctypes.c_void_p
ze_module_handle_t = ctypes.c_void_p
ze_module_build_log_handle_t = ctypes.c_void_p
ze_kernel_handle_t = ctypes.c_void_p
ze_event_handle_t = ctypes.c_void_p
ze_event_pool_handle_t = ctypes.c_void_p

ZE_MAX_DEVICE_NAME = 256
ZE_MAX_DEVICE_UUID_SIZE = 16
ZE_MAX_DRIVER_UUID_SIZE = 16


# ---------------------------------------------------------------------------
# Level Zero structures
# ---------------------------------------------------------------------------
class ze_context_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ze_command_queue_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("ordinal", ctypes.c_uint32),
        ("index", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("priority", ctypes.c_uint32),
    ]


class ze_command_list_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("commandQueueGroupOrdinal", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class ze_module_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("format", ctypes.c_uint32),
        ("inputSize", ctypes.c_size_t),
        ("pInputModule", ctypes.c_void_p),
        ("pBuildFlags", ctypes.c_char_p),
        ("pConstants", ctypes.c_void_p),
    ]


class ze_memory_free_ext_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("freePolicy", ctypes.c_uint32),
    ]


class ze_event_pool_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
    ]


class ze_event_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("index", ctypes.c_uint32),
        ("signal", ctypes.c_uint32),
        ("wait", ctypes.c_uint32),
    ]


class ze_driver_extension_properties_t(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("version", ctypes.c_uint32),
    ]


class ze_kernel_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pKernelName", ctypes.c_char_p),
    ]


class ze_group_count_t(ctypes.Structure):
    _fields_ = [
        ("groupCountX", ctypes.c_uint32),
        ("groupCountY", ctypes.c_uint32),
        ("groupCountZ", ctypes.c_uint32),
    ]


class ze_memory_allocation_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
        ("pageSize", ctypes.c_uint64),
    ]


class ze_device_mem_alloc_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("ordinal", ctypes.c_uint32),
    ]


class ze_host_mem_alloc_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
    ]


class ze_driver_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("uuid", ctypes.c_uint8 * ZE_MAX_DRIVER_UUID_SIZE),
        ("driverVersion", ctypes.c_uint32),
    ]


class ze_device_ip_version_ext_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("ipVersion", ctypes.c_uint32),
    ]


class ze_device_uuid_t(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint8 * ZE_MAX_DEVICE_UUID_SIZE)]


class ze_device_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("type", ctypes.c_uint32),
        ("vendorId", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("subdeviceId", ctypes.c_uint32),
        ("coreClockRate", ctypes.c_uint32),
        ("maxMemAllocSize", ctypes.c_uint64),
        ("maxHardwareContexts", ctypes.c_uint32),
        ("maxCommandQueuePriority", ctypes.c_uint32),
        ("numThreadsPerEU", ctypes.c_uint32),
        ("physicalEUSimdWidth", ctypes.c_uint32),
        ("numEUsPerSubslice", ctypes.c_uint32),
        ("numSubslicesPerSlice", ctypes.c_uint32),
        ("numSlices", ctypes.c_uint32),
        ("timerResolution", ctypes.c_uint64),
        ("timestampValidBits", ctypes.c_uint32),
        ("kernelTimestampValidBits", ctypes.c_uint32),
        ("uuid", ze_device_uuid_t),
        ("name", ctypes.c_char * ZE_MAX_DEVICE_NAME),
    ]


class ze_device_compute_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("maxTotalGroupSize", ctypes.c_uint32),
        ("maxGroupSizeX", ctypes.c_uint32),
        ("maxGroupSizeY", ctypes.c_uint32),
        ("maxGroupSizeZ", ctypes.c_uint32),
        ("maxGroupCountX", ctypes.c_uint32),
        ("maxGroupCountY", ctypes.c_uint32),
        ("maxGroupCountZ", ctypes.c_uint32),
        ("maxSharedLocalMemory", ctypes.c_uint32),
        ("numSubGroupSizes", ctypes.c_uint32),
        # subGroupSizes is uint32_t[8] in the spec
        ("subGroupSizes", ctypes.c_uint32 * 8),
    ]


class ze_device_module_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("spirvVersionSupported", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("fp16flags", ctypes.c_uint32),
        ("fp32flags", ctypes.c_uint32),
        ("fp64flags", ctypes.c_uint32),
        ("maxArgumentsSize", ctypes.c_uint32),
        ("printfBufferSize", ctypes.c_uint32),
        ("nativeKernelSupported", ze_device_uuid_t),
    ]


class ze_command_queue_group_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("maxMemoryFillPatternSize", ctypes.c_size_t),
        ("numQueues", ctypes.c_uint32),
    ]


class ze_device_image_properties_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("maxImageDims1D", ctypes.c_uint32),
        ("maxImageDims2D", ctypes.c_uint32),
        ("maxImageDims3D", ctypes.c_uint32),
        ("maxImageBufferSize", ctypes.c_uint64),
        ("maxImageArraySlices", ctypes.c_uint32),
        ("maxSamplers", ctypes.c_uint32),
        ("maxReadImageArgs", ctypes.c_uint32),
        ("maxWriteImageArgs", ctypes.c_uint32),
    ]


class ze_image_format_t(ctypes.Structure):
    _fields_ = [
        ("layout", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("z", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
    ]


class ze_image_desc_t(ctypes.Structure):
    _fields_ = [
        ("stype", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("format", ze_image_format_t),
        ("width", ctypes.c_uint64),
        ("height", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("arraylevels", ctypes.c_uint32),
        ("miplevels", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------
_ze = None
_ocloc = None


def _load_level_zero():
    """Load the Level Zero loader shared library."""
    names = ["libze_loader.so.1", "libze_loader.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    path = ctypes.util.find_library("ze_loader")
    if path:
        return ctypes.CDLL(path)
    raise RuntimeError(
        "Could not find Level Zero library (libze_loader.so). "
        "Install the Level Zero runtime.")


def _load_ocloc():
    """Load the Intel offline compiler shared library."""
    names = ["libocloc.so"]
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    path = ctypes.util.find_library("ocloc")
    if path:
        return ctypes.CDLL(path)
    raise RuntimeError(
        "Could not find libocloc.so. "
        "Install the Intel compute runtime (intel-opencl-icd).")


def _setup_argtypes(ze):
    """Set argtypes/restype for Level Zero functions."""
    P = ctypes.c_void_p
    U32 = ctypes.c_uint32

    ze.zeInit.argtypes = [U32]
    ze.zeInit.restype = ctypes.c_int32

    ze.zeDriverGet.argtypes = [ctypes.POINTER(U32), P]
    ze.zeDriverGet.restype = ctypes.c_int32

    ze.zeDriverGetExtensionProperties.argtypes = [P, ctypes.POINTER(U32), P]
    ze.zeDriverGetExtensionProperties.restype = ctypes.c_int32

    ze.zeDriverGetProperties.argtypes = [P, P]
    ze.zeDriverGetProperties.restype = ctypes.c_int32

    ze.zeDeviceGet.argtypes = [P, ctypes.POINTER(U32), P]
    ze.zeDeviceGet.restype = ctypes.c_int32

    ze.zeDeviceGetProperties.argtypes = [P, P]
    ze.zeDeviceGetProperties.restype = ctypes.c_int32

    ze.zeDeviceGetComputeProperties.argtypes = [P, P]
    ze.zeDeviceGetComputeProperties.restype = ctypes.c_int32

    ze.zeDeviceGetImageProperties.argtypes = [P, P]
    ze.zeDeviceGetImageProperties.restype = ctypes.c_int32

    ze.zeDeviceGetCommandQueueGroupProperties.argtypes = [P, ctypes.POINTER(U32), P]
    ze.zeDeviceGetCommandQueueGroupProperties.restype = ctypes.c_int32

    ze.zeContextCreate.argtypes = [P, P, P]
    ze.zeContextCreate.restype = ctypes.c_int32

    ze.zeContextDestroy.argtypes = [P]
    ze.zeContextDestroy.restype = ctypes.c_int32

    ze.zeCommandQueueCreate.argtypes = [P, P, P, P]
    ze.zeCommandQueueCreate.restype = ctypes.c_int32

    ze.zeCommandQueueDestroy.argtypes = [P]
    ze.zeCommandQueueDestroy.restype = ctypes.c_int32

    ze.zeCommandQueueExecuteCommandLists.argtypes = [P, U32, P, P]
    ze.zeCommandQueueExecuteCommandLists.restype = ctypes.c_int32

    ze.zeCommandQueueSynchronize.argtypes = [P, ctypes.c_uint64]
    ze.zeCommandQueueSynchronize.restype = ctypes.c_int32

    ze.zeCommandListCreate.argtypes = [P, P, P, P]
    ze.zeCommandListCreate.restype = ctypes.c_int32

    ze.zeCommandListDestroy.argtypes = [P]
    ze.zeCommandListDestroy.restype = ctypes.c_int32

    ze.zeCommandListClose.argtypes = [P]
    ze.zeCommandListClose.restype = ctypes.c_int32

    ze.zeCommandListReset.argtypes = [P]
    ze.zeCommandListReset.restype = ctypes.c_int32

    ze.zeCommandListAppendMemoryCopy.argtypes = [P, P, P, ctypes.c_size_t, P, U32, P]
    ze.zeCommandListAppendMemoryCopy.restype = ctypes.c_int32

    ze.zeCommandListAppendMemoryFill.argtypes = [P, P, P, ctypes.c_size_t, ctypes.c_size_t, P,
                                                 U32, P]
    ze.zeCommandListAppendMemoryFill.restype = ctypes.c_int32

    ze.zeCommandListAppendLaunchKernel.argtypes = [P, P, P, P, U32, P]
    ze.zeCommandListAppendLaunchKernel.restype = ctypes.c_int32

    ze.zeCommandListAppendBarrier.argtypes = [P, P, U32, P]
    ze.zeCommandListAppendBarrier.restype = ctypes.c_int32

    ze.zeMemAllocDevice.argtypes = [P, P, ctypes.c_size_t, ctypes.c_size_t, P, P]
    ze.zeMemAllocDevice.restype = ctypes.c_int32

    ze.zeMemAllocHost.argtypes = [P, P, ctypes.c_size_t, ctypes.c_size_t, P]
    ze.zeMemAllocHost.restype = ctypes.c_int32

    ze.zeMemFree.argtypes = [P, P]
    ze.zeMemFree.restype = ctypes.c_int32

    ze.zeMemGetAllocProperties.argtypes = [P, P, P, P]
    ze.zeMemGetAllocProperties.restype = ctypes.c_int32

    ze.zeCommandListAppendBarrier.argtypes = [P, P, U32, P]
    ze.zeCommandListAppendBarrier.restype = ctypes.c_int32

    ze.zeCommandListHostSynchronize.argtypes = [P, ctypes.c_uint64]
    ze.zeCommandListHostSynchronize.restype = ctypes.c_int32

    ze.zeMemFreeExt.argtypes = [P, P, P]
    ze.zeMemFreeExt.restype = ctypes.c_int32

    ze.zeEventPoolCreate.argtypes = [P, P, U32, P, P]
    ze.zeEventPoolCreate.restype = ctypes.c_int32

    ze.zeEventCreate.argtypes = [P, P, P]
    ze.zeEventCreate.restype = ctypes.c_int32

    ze.zeEventHostSynchronize.argtypes = [P, ctypes.c_uint64]
    ze.zeEventHostSynchronize.restype = ctypes.c_int32

    ze.zeEventHostReset.argtypes = [P]
    ze.zeEventHostReset.restype = ctypes.c_int32

    ze.zeModuleCreate.argtypes = [P, P, P, P, P]
    ze.zeModuleCreate.restype = ctypes.c_int32

    ze.zeModuleGetNativeBinary.argtypes = [P, ctypes.POINTER(ctypes.c_size_t), P]
    ze.zeModuleGetNativeBinary.restype = ctypes.c_int32

    ze.zeModuleDestroy.argtypes = [P]
    ze.zeModuleDestroy.restype = ctypes.c_int32

    ze.zeModuleBuildLogGetString.argtypes = [P, ctypes.POINTER(ctypes.c_size_t), P]
    ze.zeModuleBuildLogGetString.restype = ctypes.c_int32

    ze.zeModuleBuildLogDestroy.argtypes = [P]
    ze.zeModuleBuildLogDestroy.restype = ctypes.c_int32

    ze.zeKernelCreate.argtypes = [P, P, P]
    ze.zeKernelCreate.restype = ctypes.c_int32

    ze.zeKernelDestroy.argtypes = [P]
    ze.zeKernelDestroy.restype = ctypes.c_int32

    ze.zeKernelSetGroupSize.argtypes = [P, U32, U32, U32]
    ze.zeKernelSetGroupSize.restype = ctypes.c_int32

    ze.zeKernelSetArgumentValue.argtypes = [P, U32, ctypes.c_size_t, P]
    ze.zeKernelSetArgumentValue.restype = ctypes.c_int32

    ze.zeImageCreate.argtypes = [P, P, P, P]
    ze.zeImageCreate.restype = ctypes.c_int32

    ze.zeImageDestroy.argtypes = [P]
    ze.zeImageDestroy.restype = ctypes.c_int32

    ze.zeCommandListAppendImageCopyFromMemory.argtypes = [P, P, P, P, P, U32, P]
    ze.zeCommandListAppendImageCopyFromMemory.restype = ctypes.c_int32


def _setup_ocloc_argtypes(ocloc):
    """Set argtypes/restype for ocloc functions."""
    ocloc.oclocInvoke.restype = ctypes.c_int
    ocloc.oclocInvoke.argtypes = [
        ctypes.c_uint,                     # numArgs
        ctypes.POINTER(ctypes.c_char_p),   # argv
        ctypes.c_uint32,                   # numSources
        ctypes.c_void_p,                   # dataSources (uint8_t**)
        ctypes.c_void_p,                   # lenSources (uint64_t*)
        ctypes.c_void_p,                   # nameSources (char**)
        ctypes.c_uint32,                   # numInputHeaders
        ctypes.c_void_p,                   # dataInputHeaders
        ctypes.c_void_p,                   # lenInputHeaders
        ctypes.c_void_p,                   # nameInputHeaders
        ctypes.POINTER(ctypes.c_uint32),   # numOutputs
        ctypes.POINTER(ctypes.c_void_p),   # dataOutputs
        ctypes.POINTER(ctypes.c_void_p),   # lenOutputs
        ctypes.POINTER(ctypes.c_void_p),   # nameOutputs
    ]
    ocloc.oclocFreeOutput.restype = ctypes.c_int


def _get_ze():
    global _ze
    if _ze is None:
        _ze = _load_level_zero()
        _setup_argtypes(_ze)
    return _ze


def _get_ocloc():
    global _ocloc
    if _ocloc is None:
        _ocloc = _load_ocloc()
        _setup_ocloc_argtypes(_ocloc)
    return _ocloc


def _extensions(ze, driver) -> set:
    """The names of the extensions a driver supports."""
    count = ctypes.c_uint32(0)
    if ze.zeDriverGetExtensionProperties(driver, ctypes.byref(count), None) != ZE_RESULT_SUCCESS:
        return set()
    props = (ze_driver_extension_properties_t * count.value)()
    if ze.zeDriverGetExtensionProperties(driver, ctypes.byref(count), props) != ZE_RESULT_SUCCESS:
        return set()
    return {p.name.decode(errors="replace") for p in props}


def _check_ze(result, msg="Level Zero"):
    """Check a Level Zero result, raise on error."""
    if result != ZE_RESULT_SUCCESS:
        raise RuntimeError(f"{msg} failed with ze_result_t 0x{result:08x}")
    return result


# ---------------------------------------------------------------------------
# In-process SPIR-V compilation via libocloc
# ---------------------------------------------------------------------------
def _extract_ocloc_log(num_outputs, data_outputs, len_outputs, name_outputs):
    """Extract stdout.log text from ocloc output arrays."""
    if not data_outputs or num_outputs.value == 0:
        return ""
    # Interpret output arrays as raw pointers
    n = num_outputs.value
    data_arr = ctypes.cast(data_outputs, ctypes.POINTER(ctypes.c_void_p))
    len_arr = ctypes.cast(len_outputs, ctypes.POINTER(ctypes.c_uint64))
    name_arr = ctypes.cast(name_outputs, ctypes.POINTER(ctypes.c_char_p))
    for i in range(n):
        name = name_arr[i]
        if name and b"stdout.log" in name:
            log_len = len_arr[i]
            if log_len > 0 and data_arr[i]:
                return ctypes.string_at(data_arr[i], log_len).decode(errors="replace")
    return ""


def _extract_ocloc_spv(num_outputs, data_outputs, len_outputs, name_outputs):
    """Extract .spv binary from ocloc output arrays."""
    if not data_outputs or num_outputs.value == 0:
        return None
    n = num_outputs.value
    data_arr = ctypes.cast(data_outputs, ctypes.POINTER(ctypes.c_void_p))
    len_arr = ctypes.cast(len_outputs, ctypes.POINTER(ctypes.c_uint64))
    name_arr = ctypes.cast(name_outputs, ctypes.POINTER(ctypes.c_char_p))
    for i in range(n):
        name = name_arr[i]
        if name and name.endswith(b".spv"):
            spv_len = len_arr[i]
            if spv_len > 0 and data_arr[i]:
                return ctypes.string_at(data_arr[i], spv_len)
    return None


def _free_ocloc_output(num_outputs, data_outputs, len_outputs, name_outputs):
    """Free ocloc output arrays."""
    ocloc = _get_ocloc()
    ocloc.oclocFreeOutput(
        ctypes.byref(num_outputs), ctypes.byref(data_outputs),
        ctypes.byref(len_outputs), ctypes.byref(name_outputs))


# What ocloc is given besides the device, and the flags zeModuleCreate
# finalizes the SPIR-V with. Part of the disk cache's target: SPIR-V
# compiled, or a module built, with other options is another artifact.
_OCLOC_OPTIONS = (b"-spv_only", b"-options", b"-cl-std=CL2.0")
_MODULE_BUILD_FLAGS = b"-cl-std=CL2.0"


def _ocloc_version() -> str:
    """The driver version of the libocloc loaded, e.g. ``25.05.32567.17``."""
    ocloc = _get_ocloc()
    # Unlike compile's, a query's argv starts with the program's name.
    args_list = [b"ocloc", b"query", b"OCL_DRIVER_VERSION"]
    argv = (ctypes.c_char_p * len(args_list))(*args_list)
    num_outputs = ctypes.c_uint32(0)
    data_outputs = ctypes.c_void_p()
    len_outputs = ctypes.c_void_p()
    name_outputs = ctypes.c_void_p()
    result = ocloc.oclocInvoke(
        len(args_list), argv, 0, None, None, None, 0, None, None, None,
        ctypes.byref(num_outputs), ctypes.byref(data_outputs),
        ctypes.byref(len_outputs), ctypes.byref(name_outputs))
    log = _extract_ocloc_log(num_outputs, data_outputs, len_outputs, name_outputs)
    _free_ocloc_output(num_outputs, data_outputs, len_outputs, name_outputs)
    if result != 0:
        raise RuntimeError(f"ocloc query failed (error {result}):\n{log}")
    return log.strip("\x00").strip()


def _compile_to_spirv(opencl_source: str, device_id: int) -> bytes:
    """Compile OpenCL C source to SPIR-V binary using libocloc in-process."""
    ocloc = _get_ocloc()

    # Null-terminate source and include the terminator in the length —
    # ocloc treats in-memory sources as C strings internally.
    src = opencl_source.encode("utf-8") + b"\x00"

    device_str = f"0x{device_id:04x}"
    args_list = [
        b"compile",
        b"-device", device_str.encode(),
        *_OCLOC_OPTIONS,
        b"-file", b"kernel.cl",
    ]
    argc = len(args_list)
    argv = (ctypes.c_char_p * argc)(*args_list)

    # In-memory source: pack the pointer, length, and name into a contiguous
    # ctypes struct so the three parallel arrays share stable addresses.
    src_np = np.frombuffer(src, dtype=np.uint8).copy()

    class _OclocSrc(ctypes.Structure):
        _fields_ = [
            ("data_ptr", ctypes.c_void_p),
            ("length", ctypes.c_uint64),
            ("name", ctypes.c_char_p),
        ]

    src_desc = _OclocSrc(
        data_ptr=src_np.ctypes.data,
        length=len(src),
        name=b"kernel.cl",
    )

    # Output placeholders
    num_outputs = ctypes.c_uint32(0)
    data_outputs = ctypes.c_void_p()
    len_outputs = ctypes.c_void_p()
    name_outputs = ctypes.c_void_p()

    base = ctypes.addressof(src_desc)
    result = ocloc.oclocInvoke(
        argc, argv,
        1,
        base,            # &data_ptr  = uint8_t**
        base + 8,        # &length    = uint64_t*
        base + 16,       # &name      = char**
        0, None, None, None,
        ctypes.byref(num_outputs),
        ctypes.byref(data_outputs),
        ctypes.byref(len_outputs),
        ctypes.byref(name_outputs),
    )

    if result != 0:
        log = _extract_ocloc_log(num_outputs, data_outputs, len_outputs, name_outputs)
        _free_ocloc_output(num_outputs, data_outputs, len_outputs, name_outputs)
        raise RuntimeError(
            f"ocloc compilation failed (error {result}):\n{log}\n"
            f"Source:\n{opencl_source}")

    # Find the .spv output
    spirv_bytes = _extract_ocloc_spv(num_outputs, data_outputs, len_outputs, name_outputs)
    _free_ocloc_output(num_outputs, data_outputs, len_outputs, name_outputs)

    if spirv_bytes is None:
        raise RuntimeError(
            f"ocloc produced no .spv output.\nSource:\n{opencl_source}")

    return spirv_bytes


def _create_module(backend, data: bytes, module_format, opencl_source=None):
    """A module of the backend's context, from SPIR-V or a native binary.

    ``opencl_source``, when known, goes into the error a failed build raises.
    """
    ze = _get_ze()
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    module_desc = ze_module_desc_t(
        stype=ZE_STRUCTURE_TYPE_MODULE_DESC, pNext=None,
        format=module_format,
        inputSize=len(data),
        pInputModule=ctypes.cast(buf, ctypes.c_void_p),
        pBuildFlags=_MODULE_BUILD_FLAGS,
        pConstants=None)

    module = ze_module_handle_t()
    build_log = ze_module_build_log_handle_t()
    result = ze.zeModuleCreate(
        backend._context, backend._device, ctypes.byref(module_desc),
        ctypes.byref(module), ctypes.byref(build_log))

    if result != ZE_RESULT_SUCCESS:
        # Extract build log
        log_text = ""
        if build_log:
            log_size = ctypes.c_size_t(0)
            ze.zeModuleBuildLogGetString(build_log, ctypes.byref(log_size), None)
            if log_size.value > 0:
                log_buf = ctypes.create_string_buffer(log_size.value)
                ze.zeModuleBuildLogGetString(build_log, ctypes.byref(log_size), log_buf)
                log_text = log_buf.value.decode(errors="replace")
            ze.zeModuleBuildLogDestroy(build_log)
        source = f"Source:\n{opencl_source}" if opencl_source is not None else ""
        raise RuntimeError(
            f"zeModuleCreate failed (0x{result:08x}):\n{log_text}\n{source}")

    if build_log:
        ze.zeModuleBuildLogDestroy(build_log)
    return module


def _native_binary(module) -> bytes:
    """The device code the driver made for a module, loadable as ZE_MODULE_FORMAT_NATIVE."""
    ze = _get_ze()
    size = ctypes.c_size_t(0)
    _check_ze(ze.zeModuleGetNativeBinary(module, ctypes.byref(size), None),
               "zeModuleGetNativeBinary (size)")
    buf = (ctypes.c_uint8 * size.value)()
    _check_ze(ze.zeModuleGetNativeBinary(module, ctypes.byref(size), buf),
               "zeModuleGetNativeBinary")
    return bytes(buf)


def _load_kernel(backend, spirv: bytes, kernel_name: str, param_types, param_is_array,
                 param_is_texture, workgroup_size, requires_full_workgroups=False,
                 opencl_source=None, native=None) -> "CompiledL0Kernel":
    """A kernel from its SPIR-V, or from its native binary when one is given.

    A native binary skips IGC's finalization of the SPIR-V, about 50 ms a
    kernel when the driver's own cache does not have it. One the driver
    refuses -- another stepping, another IGC -- falls back to the SPIR-V.
    """
    ze = _get_ze()
    module = None
    if native is not None:
        try:
            module = _create_module(backend, native, ZE_MODULE_FORMAT_NATIVE)
        except RuntimeError:
            from tack.runtime import disk_cache
            disk_cache._count("native_refused")
    if module is None:
        module = _create_module(backend, spirv, ZE_MODULE_FORMAT_IL_SPIRV, opencl_source)

    # Create kernel
    kernel_desc = ze_kernel_desc_t(
        stype=ZE_STRUCTURE_TYPE_KERNEL_DESC, pNext=None,
        flags=0, pKernelName=kernel_name.encode())
    kernel = ze_kernel_handle_t()
    _check_ze(ze.zeKernelCreate(module, ctypes.byref(kernel_desc),
                                 ctypes.byref(kernel)),
               "zeKernelCreate")

    compiled = CompiledL0Kernel(module, kernel, kernel_name,
                                param_types, param_is_array, workgroup_size,
                                param_is_texture,
                                requires_full_workgroups=requires_full_workgroups)
    compiled.spirv = spirv
    return compiled


# ---------------------------------------------------------------------------
# L0Buffer
# ---------------------------------------------------------------------------
def _release(obj):
    """Give back a held object that came from a pool (a `_StagingBlock`);
    anything else is simply dropped."""
    release = getattr(obj, "release", None)
    if release is not None:
        release()


class _StagingBlock:
    """A block of USM host memory from `_HostStaging`, given back with
    `release` once the queue has passed the copy that uses it."""

    __slots__ = ("_staging", "capacity", "ptr")

    def __init__(self, ptr, capacity, staging):
        self.ptr = ptr
        self.capacity = capacity
        self._staging = staging

    def array(self, dtype, count):
        """The first ``count`` elements of the block as a NumPy array."""
        return np.frombuffer((ctypes.c_char * self.capacity).from_address(self.ptr),
                             dtype=dtype, count=count)

    def release(self):
        self._staging.give(self)


class _HostStaging:
    """USM host memory for copies between the host and the device.

    Intel's driver (25.05, Max 1100) does not return from appending a small
    copy to or from pageable memory until the queue has drained -- to the
    device up to 4 MiB, from it up to 1 KiB -- so a launch whose scalar pack
    was uploaded behind a long kernel waited for that kernel. From USM host
    memory the append returns at once. Copies up to `WRITE_LIMIT` and
    `READ_LIMIT`, a margin above those sizes, go through a block here; larger
    ones use pageable memory directly, saving a host copy. Blocks are powers
    of two from 256 bytes, kept for reuse up to `_KEEP_BYTES` in all.

    The free lists are guarded by the backend's `_launch_lock`, not a lock
    of their own: an array collected while a thread held a second lock would
    wait for the launch lock in `_free`, and a thread holding the launch
    lock while releasing blocks would wait for the second.
    """

    WRITE_LIMIT = 16 << 20
    READ_LIMIT = 64 << 10
    _MIN = 256
    _KEEP_BYTES = 64 << 20

    def __init__(self, context, lock):
        self._context = context
        self._free = {}
        self._kept = 0
        self._lock = lock

    def take(self, nbytes) -> _StagingBlock:
        capacity = max(self._MIN, 1 << (max(nbytes, 1) - 1).bit_length())
        with self._lock:
            blocks = self._free.get(capacity)
            if blocks:
                self._kept -= capacity
                return blocks.pop()
        desc = ze_host_mem_alloc_desc_t(
            stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, pNext=None, flags=0)
        ptr = ctypes.c_void_p()
        _check_ze(_get_ze().zeMemAllocHost(self._context, ctypes.byref(desc),
                                           capacity, 64, ctypes.byref(ptr)),
                   "zeMemAllocHost (staging)")
        return _StagingBlock(ptr.value, capacity, self)

    def give(self, block):
        with self._lock:
            if self._kept + block.capacity <= self._KEEP_BYTES:
                self._free.setdefault(block.capacity, []).append(block)
                self._kept += block.capacity
                return
        _get_ze().zeMemFree(self._context, block.ptr)


class L0Buffer(DeviceBuffer):
    """Device-resident buffer backed by a Level Zero device allocation.

    Data lives on the GPU. Everything that touches it goes through the
    backend's queue, in order with the launches: ``from_numpy`` and ``fill``
    are queued and return at once, ``to_numpy`` and ``read_range`` are
    queued and waited for, on an event after the copy and outside the
    backend's lock, so other threads go on launching meanwhile. Freeing is
    deferred by the driver until the work queued so far is done, so an array
    Python drops while a queued launch still uses it outlives that launch.
    """

    backend_name = "level_zero"
    # Memory handed outside Tack (DLPack, a wrapped pointer) may be read
    # with no Tack call between: a launch that uses it completes before
    # returning, as on Metal and CUDA.
    _shared = False

    def __init__(self, backend, numpy_dtype, shape):
        self._backend = backend
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(math.prod(shape)) * self._numpy_dtype.itemsize

        ze = _get_ze()
        alloc_size = max(self._nbytes, 1)

        # Allocate device memory
        alloc_desc = ze_device_mem_alloc_desc_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
            pNext=None, flags=0, ordinal=0)
        self._device_ptr = ctypes.c_void_p()
        _check_ze(ze.zeMemAllocDevice(
            backend._context, ctypes.byref(alloc_desc),
            alloc_size, 0, backend._device,
            ctypes.byref(self._device_ptr)),
            "zeMemAllocDevice")

        # Zero-initialize, on the device
        self.fill_bytes(b"\0")

    @property
    def address(self) -> int:
        return int(self._device_ptr.value)

    @property
    def device_ptr(self):
        return self._device_ptr

    def synchronize(self):
        """Wait for every launch, fill and copy queued on the backend."""
        self._backend.synchronize()

    def share(self):
        """Hand this memory outside Tack: wait for the queued work now, and
        have every later launch that uses it complete before returning."""
        self._shared = True
        self.synchronize()

    @property
    def waits_off_thread(self) -> bool:
        """Whether `Array.numpy_async` reads in a worker thread: where
        launches are queued, reading waits for them. Level Zero needs no
        per-thread binding."""
        return self._backend._queued

    def _copy_to_device(self, arr: np.ndarray):
        """Queue a copy of a numpy array to the device.

        The copy runs later, so it reads a private copy of ``arr``, which
        the caller may change or drop at once; the backend holds it until
        the queue has passed it. Up to `_HostStaging.WRITE_LIMIT` the copy
        is in USM host memory, so appending does not wait for the queue.
        """
        ze = _get_ze()
        backend = self._backend
        if self._nbytes > _HostStaging.WRITE_LIMIT:
            src = np.array(arr, dtype=self._numpy_dtype, order="C", copy=True)
            with backend._launch_lock:
                backend._ordered(ze.zeCommandListAppendMemoryCopy(
                    backend._queue,
                    self._device_ptr, src.ctypes.data, self._nbytes,
                    None, 0, None), "zeCommandListAppendMemoryCopy (H2D)")
                backend._hold(src, self._nbytes)
            return
        count = self._nbytes // self._numpy_dtype.itemsize
        block = backend._staging.take(self._nbytes)
        try:
            block.array(self._numpy_dtype, count)[...] = np.asarray(arr).reshape(-1)
        except BaseException:
            block.release()
            raise
        with backend._launch_lock:
            try:
                backend._ordered(ze.zeCommandListAppendMemoryCopy(
                    backend._queue,
                    self._device_ptr, block.ptr, self._nbytes,
                    None, 0, None), "zeCommandListAppendMemoryCopy (H2D)")
            except BaseException:
                block.release()
                raise
            backend._hold(block, self._nbytes)

    def _read_into(self, out_ptr, device_ptr, nbytes, what):
        """Copy ``nbytes`` from the device to host memory at ``out_ptr``,
        after the work queued before it, waiting on a marker outside the
        backend's lock: through a staging block up to
        `_HostStaging.READ_LIMIT`, directly above it."""
        ze = _get_ze()
        backend = self._backend
        if nbytes > _HostStaging.READ_LIMIT:
            with backend._launch_lock:
                backend._ordered(ze.zeCommandListAppendMemoryCopy(
                    backend._queue, out_ptr, device_ptr, nbytes, None, 0, None), what)
                marker = backend._mark()
            if marker is not None:
                marker.wait()
            return
        block = backend._staging.take(nbytes)
        try:
            with backend._launch_lock:
                backend._ordered(ze.zeCommandListAppendMemoryCopy(
                    backend._queue, block.ptr, device_ptr, nbytes,
                    None, 0, None), what)
                marker = backend._mark()
            if marker is not None:
                marker.wait()
            ctypes.memmove(out_ptr, block.ptr, nbytes)
        finally:
            block.release()

    def _copy_from_device(self, out: np.ndarray):
        """Copy device → numpy array, after the work queued before it."""
        self._read_into(out.ctypes.data, self._device_ptr, self._nbytes,
                        "zeCommandListAppendMemoryCopy (D2H)")

    def from_numpy(self, arr: np.ndarray):
        self._copy_to_device(arr)

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        self._copy_from_device(out)
        return out

    def read_range(self, start: int, count: int) -> np.ndarray:
        """Copy elements ``[start, start + count)`` to the host.

        One copy of just those bytes, so ``array[i]`` on the host moves one
        element rather than the whole array. The range is checked here
        because a device copy past the allocation is not caught by NumPy.
        """
        itemsize = self._numpy_dtype.itemsize
        if start < 0 or count < 0 or (start + count) * itemsize > self._nbytes:
            raise IndexError(
                f"elements [{start}, {start + count}) are outside a buffer of "
                f"{self._nbytes // itemsize} elements")
        out = np.empty(count, dtype=self._numpy_dtype)
        if count:
            self._read_into(out.ctypes.data, self.address + start * itemsize,
                            count * itemsize, "zeCommandListAppendMemoryCopy (D2H range)")
        return out

    def fill(self, value):
        self.fill_bytes(device_fill.scalar_pattern(value, self._numpy_dtype))

    def fill_bytes(self, pattern):
        ze = _get_ze()
        backend, base = self._backend, self.address

        def native(pattern):
            # The compute group's fill takes a power-of-two pattern up to its
            # maximum. The fill runs later: the backend holds the pattern.
            size = len(pattern)
            if size & (size - 1) or size > backend._max_fill_pattern:
                return False
            buffer = ctypes.create_string_buffer(pattern, size)
            with backend._launch_lock:
                backend._ordered(ze.zeCommandListAppendMemoryFill(
                    backend._queue, self._device_ptr, buffer, size, self._nbytes,
                    None, 0, None), "zeCommandListAppendMemoryFill")
                backend._hold(buffer, size)
            return True

        def write(offset, data):
            source = backend._staging.take(len(data))
            ctypes.memmove(source.ptr, data, len(data))
            with backend._launch_lock:
                try:
                    backend._ordered(ze.zeCommandListAppendMemoryCopy(
                        backend._queue, base + offset, source.ptr, len(data), None, 0, None),
                        "zeCommandListAppendMemoryCopy (H2D pattern)")
                except BaseException:
                    source.release()
                    raise
                backend._hold(source, len(data))

        def copy(dst, src, size):
            with backend._launch_lock:
                backend._ordered(ze.zeCommandListAppendMemoryCopy(
                    backend._queue, base + dst, base + src, size, None, 0, None),
                    "zeCommandListAppendMemoryCopy (D2D)")

        device_fill.fill(self._nbytes, pattern, native, write, copy)
        return True

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __del__(self):
        if hasattr(self, '_device_ptr') and self._device_ptr and getattr(self, '_owned', True):
            try:
                # After every launch queued so far, which may still use it.
                self._backend._free(self._device_ptr)
            except Exception:
                pass


class _ListMarker:
    """A point on the backend's queue after the work appended so far: a
    barrier that signals a host-visible event, which any thread can wait on
    (`Backend.async_marker`). The wait does not hold the backend's lock, so
    other threads go on launching; it hands the event back and releases the
    host memory held for the copies and fills before the mark."""

    __slots__ = ("_backend", "_event", "_held_upto")

    def __init__(self, backend):
        """Append the mark. Called under the backend's `_launch_lock`."""
        self._backend = backend
        self._event = backend._take_event()
        try:
            _check_ze(_get_ze().zeCommandListAppendBarrier(
                backend._queue, self._event, 0, None),
                "zeCommandListAppendBarrier (marker)")
        except BaseException:
            backend._events.append(self._event)
            raise
        self._held_upto = backend._held_released + len(backend._held)

    def wait(self):
        event = self._event
        if event is None:
            return
        self._event = None
        backend = self._backend
        try:
            _check_ze(_get_ze().zeEventHostSynchronize(event, 0xFFFFFFFFFFFFFFFF),
                       "zeEventHostSynchronize")
        finally:
            with backend._launch_lock:
                _get_ze().zeEventHostReset(event)
                backend._events.append(event)
                backend._release_held(self._held_upto)


# ---------------------------------------------------------------------------
# CompiledL0Kernel
# ---------------------------------------------------------------------------
_L0_CTYPES_MAP = {
    f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
    u32: ctypes.c_uint, u64: ctypes.c_ulonglong,
}


class L0TextureImage:
    """A read-only Level Zero 3D image, sampled through a hardware sampler.

    Owned by one `Texture3D`, which uploads its array's data at creation and
    on ``update()``. Nothing else keeps a reference, so the image is
    destroyed with the texture: no later texture can be served one that was
    made from someone else's memory.
    """

    def __init__(self, shape_3d, backend):
        ze = _get_ze()
        W, H, D = shape_3d
        self._shape = shape_3d
        self._backend = backend
        fmt = ze_image_format_t(
            layout=ZE_IMAGE_FORMAT_LAYOUT_32,
            type=ZE_IMAGE_FORMAT_TYPE_FLOAT,
            x=ZE_IMAGE_FORMAT_SWIZZLE_R,
            y=ZE_IMAGE_FORMAT_SWIZZLE_0,
            z=ZE_IMAGE_FORMAT_SWIZZLE_0,
            w=ZE_IMAGE_FORMAT_SWIZZLE_1,
        )
        desc = ze_image_desc_t(
            stype=ZE_STRUCTURE_TYPE_IMAGE_DESC,
            pNext=None,
            flags=0,  # read-only (no ZE_IMAGE_FLAG_KERNEL_WRITE)
            type=ZE_IMAGE_TYPE_3D,
            format=fmt,
            width=W,
            height=H,
            depth=D,
            arraylevels=0,
            miplevels=0,
        )
        image = ctypes.c_void_p()
        _check_ze(ze.zeImageCreate(
            backend._context, backend._device,
            ctypes.byref(desc), ctypes.byref(image)),
            "zeImageCreate")
        self.handle = image

    def upload(self, array):
        """Copy an array's device buffer into the image."""
        ze = _get_ze()
        backend = self._backend
        W, H, D = self._shape

        # Copy data: device → host → image
        # Direct device→image copy can fail with OOM on the host staging path,
        # so we stage through a host allocation explicitly.
        nbytes = W * H * D * 4  # R32F
        host_desc = ze_host_mem_alloc_desc_t(
            stype=ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, pNext=None, flags=0)
        host_ptr = ctypes.c_void_p()
        _check_ze(ze.zeMemAllocHost(
            backend._context, ctypes.byref(host_desc),
            nbytes, 0, ctypes.byref(host_ptr)),
            "zeMemAllocHost (image staging)")

        # Device → host staging buffer → image, then wait for both before
        # freeing the staging memory they use. Uploads are rare; launches
        # are not. The queue is shared, so the appends hold its lock.
        with backend._launch_lock:
            backend._ordered(ze.zeCommandListAppendMemoryCopy(
                backend._queue, host_ptr,
                array._buffer.device_ptr, nbytes,
                None, 0, None), "zeCommandListAppendMemoryCopy (D2H staging)")
            backend._ordered(ze.zeCommandListAppendImageCopyFromMemory(
                backend._queue, self.handle,
                host_ptr, None,
                None, 0, None), "zeCommandListAppendImageCopyFromMemory")
            backend.synchronize()
        ze.zeMemFree(backend._context, host_ptr)

    def __del__(self):
        if getattr(self, 'handle', None):
            try:
                _get_ze().zeImageDestroy(self.handle)
            except Exception:
                pass


class CompiledL0Kernel:
    """A compiled Level Zero kernel ready for dispatch."""

    def __init__(self, module, kernel, func_name, param_types, param_is_array,
                 workgroup_size, param_is_texture=None, *,
                 requires_full_workgroups=False):
        self._requires_full_workgroups = requires_full_workgroups
        if requires_full_workgroups:
            check_workgroup_launch(func_name, 0, backend_label='Level Zero',
                                   workgroup_size=workgroup_size)
        self._module = module
        self._kernel = kernel
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_array = param_is_array
        self._workgroup_size = workgroup_size
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self.spirv = None   # what ocloc made, kept for the disk cache

    def __call__(self, kernel_args: list, loop_end: int, backend, extents=()):
        """Dispatch the Level Zero kernel.

        A multi-dimensional loop (``extents``, slowest first) launches groups
        in its shape, or flat ones with ``__flat__`` set when an extent is
        past the device's group-count limits (see launch_geometry).
        """
        if self._requires_full_workgroups:
            check_workgroup_launch(self._func_name, loop_end, backend_label='Level Zero',
                                   workgroup_size=self._workgroup_size)
        ze = _get_ze()
        kernel = self._kernel

        # Set kernel arguments
        arg_idx = 0
        for arg, ptype, is_array, is_tex in zip(
                kernel_args, self._param_types, self._param_is_array,
                self._param_is_texture):
            if is_tex:
                # Bind an L0TextureImage as image3d_t
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, arg_idx, ctypes.sizeof(ctypes.c_void_p),
                    ctypes.byref(arg.handle)),
                    f"zeKernelSetArgumentValue (image arg {arg_idx})")
            elif is_array:
                ptr = arg._buffer.device_ptr
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, arg_idx, ctypes.sizeof(ctypes.c_void_p),
                    ctypes.byref(ptr)),
                    f"zeKernelSetArgumentValue (array arg {arg_idx})")
            else:
                ct = _L0_CTYPES_MAP[ptype]
                val = ct(arg)
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, arg_idx, ctypes.sizeof(val),
                    ctypes.byref(val)),
                    f"zeKernelSetArgumentValue (scalar arg {arg_idx})")
            arg_idx += 1

        # Set __n__ parameter (loop count)
        n_val = ctypes.c_longlong(loop_end)
        _check_ze(ze.zeKernelSetArgumentValue(
            kernel, arg_idx, ctypes.sizeof(n_val),
            ctypes.byref(n_val)),
            "zeKernelSetArgumentValue (__n__)")

        wg = self._workgroup_size
        if extents:
            props = backend._compute_props
            grid, block, flat = launch_geometry(
                extents, block_size=wg,
                max_grid=(props.maxGroupCountX, props.maxGroupCountY, props.maxGroupCountZ),
                max_block=(props.maxGroupSizeX, props.maxGroupSizeY, props.maxGroupSizeZ))
            values = [ctypes.c_int(int(flat))] + [ctypes.c_longlong(e) for e in extents]
            for k, value in enumerate(values, start=arg_idx + 1):
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, k, ctypes.sizeof(value), ctypes.byref(value)),
                    f"zeKernelSetArgumentValue (launch arg {k})")
        else:
            block = (wg, 1, 1)
            grid = ((loop_end + wg - 1) // wg, 1, 1)

        # Set workgroup size
        _check_ze(ze.zeKernelSetGroupSize(kernel, *block),
                   "zeKernelSetGroupSize")

        group_count = ze_group_count_t(
            groupCountX=grid[0],
            groupCountY=grid[1],
            groupCountZ=grid[2])

        # Queue the launch and return. The kernel's arguments and group size
        # are captured as it is appended, so the next launch of this kernel
        # may set its own at once.
        backend._ordered(ze.zeCommandListAppendLaunchKernel(
            backend._queue, kernel, ctypes.byref(group_count),
            None, 0, None), "zeCommandListAppendLaunchKernel")


def _driver_devices(ze, driver=None):
    """A driver (the first, unless given) and its devices, in the order that
    numbers them: ``level_zero:N`` is the Nth."""
    if driver is None:
        count = ctypes.c_uint32(0)
        _check_ze(ze.zeDriverGet(ctypes.byref(count), None), "zeDriverGet (count)")
        if count.value == 0:
            raise RuntimeError("No Level Zero drivers found")
        drivers = (ze_driver_handle_t * count.value)()
        _check_ze(ze.zeDriverGet(ctypes.byref(count), drivers), "zeDriverGet")
        driver = drivers[0]
    count = ctypes.c_uint32(0)
    _check_ze(ze.zeDeviceGet(driver, ctypes.byref(count), None), "zeDeviceGet (count)")
    devices = (ze_device_handle_t * count.value)()
    if count.value:
        _check_ze(ze.zeDeviceGet(driver, ctypes.byref(count), devices), "zeDeviceGet")
    return driver, [int(d or 0) for d in devices]


# ---------------------------------------------------------------------------
# LevelZeroBackend
# ---------------------------------------------------------------------------
class LevelZeroBackend(GPUBackend):
    """Level Zero GPU backend — device-resident arrays, ocloc compilation."""

    name = "level_zero"
    display_name = "Level Zero"
    # supported_dtypes is set per device in __init__: f64 only when the
    # device reports it.
    supported_dtypes = _L0_SUPPORTED_DTYPES
    supports_device_reductions = True
    supports_workgroups = True
    device_options = frozenset({"external_context"})

    @property
    def memory_domain(self) -> tuple:
        # A USM pointer is valid only in the context that allocated it, and
        # two backends (one adopting another library's context) can differ.
        return (self.name, self.ordinal, self._context.value)
    device_memory_spaces = frozenset({"level_zero"})
    multi_device = True

    @classmethod
    def device_count(cls) -> int:
        ze = _get_ze()
        _check_ze(ze.zeInit(0), "zeInit")
        return len(_driver_devices(ze)[1])

    def __init__(self, external_context=None, ordinal=None):
        ze = _get_ze()

        # Initialize Level Zero
        _check_ze(ze.zeInit(0), "zeInit")

        # A USM pointer means something only inside the context it was
        # allocated from, and nothing that exchanges pointers -- DLPack, a
        # raw address -- carries one. Sharing device memory with another
        # library therefore means allocating in *its* context, so all three
        # handles are taken from it: a driver or device handle obtained
        # separately need not be the one that context was created against.
        self._owns_context = external_context is None
        if external_context is None:
            self._init_own_driver_and_device(ze, 0 if ordinal is None else ordinal)
        else:
            missing = [k for k in ("driver", "device", "context")
                       if not external_context.get(k)]
            if missing:
                raise ValueError(
                    "external_context needs non-null 'driver', 'device' and "
                    f"'context' handles; missing or null: {', '.join(missing)}")
            self._driver = int(external_context["driver"])
            self._device = int(external_context["device"])
            # The context's device numbers this one, among its driver's.
            devices = _driver_devices(ze, self._driver)[1]
            found = devices.index(self._device) if self._device in devices else 0
            if ordinal is not None and ordinal != found:
                raise ValueError(
                    f"level_zero:{ordinal}: the external_context's device is "
                    f"level_zero:{found}; name it, or write 'level_zero'")
            self.ordinal = found

        self._query_device(ze)

        if self._owns_context:
            ctx_desc = ze_context_desc_t(
                stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0)
            self._context = ze_context_handle_t()
            _check_ze(ze.zeContextCreate(self._driver, ctypes.byref(ctx_desc),
                                          ctypes.byref(self._context)),
                       "zeContextCreate")
        else:
            self._context = ze_context_handle_t(int(external_context["context"]))

        self._create_queues(ze)
        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledL0Kernel}

    @property
    def shares_external_context(self) -> bool:
        """Whether this backend allocates in a context another library owns."""
        return not self._owns_context

    def memory_space(self, ptr) -> str:
        """Classify an address by asking the context what it allocated there.

        'level_zero' for USM device or shared memory, 'cpu' otherwise. An
        address the driver does not recognise -- ordinary host memory -- is
        a successful query of type UNKNOWN, so a failed query is a fault and
        raises rather than being reported as host memory (see D9 in
        `hip_backend.memory_space`).

        Intel's driver answers for pointers from any context on the device,
        so this cannot tell whether `ptr` was allocated in *this* context.
        """
        addr = as_address(ptr)
        if addr is None:
            return "cpu"
        props = ze_memory_allocation_properties_t(
            stype=ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES, pNext=None)
        _check_ze(_get_ze().zeMemGetAllocProperties(
            self._context, ctypes.c_void_p(addr), ctypes.byref(props), None),
            "zeMemGetAllocProperties")
        if props.type in (ZE_MEMORY_TYPE_DEVICE, ZE_MEMORY_TYPE_SHARED):
            return "level_zero"
        return "cpu"

    def _init_own_driver_and_device(self, ze, ordinal):
        # The first driver's devices; each backend makes its own context.
        self._driver, devices = _driver_devices(ze)
        if not devices:
            raise RuntimeError("No Level Zero devices found")
        if not 0 <= ordinal < len(devices):
            raise ValueError(f"level_zero:{ordinal}: the first Level Zero driver has "
                             f"{len(devices)} "
                             f"device{'s' if len(devices) != 1 else ''}")
        self.ordinal = ordinal
        self._device = devices[ordinal]

    def _query_device(self, ze):
        # Get device properties (for device ID and name)
        self._dev_props = ze_device_properties_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, pNext=None)
        _check_ze(ze.zeDeviceGetProperties(self._device, ctypes.byref(self._dev_props)),
                   "zeDeviceGetProperties")
        self._device_id = self._dev_props.deviceId
        self._device_name = self._dev_props.name.decode(errors="replace").rstrip("\x00")

        # Get compute properties (for max workgroup size)
        self._compute_props = ze_device_compute_properties_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES, pNext=None)
        _check_ze(ze.zeDeviceGetComputeProperties(
            self._device, ctypes.byref(self._compute_props)),
            "zeDeviceGetComputeProperties")
        # The group count is a uint32 that ctypes would truncate rather than
        # reject. Kernels use the workgroup size _compile_kernel chooses;
        # native reductions run only where that is 256.
        self._max_launch = self._compute_props.maxGroupCountX * min(
            WORKGROUP_SIZE, self._compute_props.maxGroupSizeX,
            self._compute_props.maxTotalGroupSize)

        # Get image properties (for max 3D texture dimensions and sampler support)
        self._image_props = ze_device_image_properties_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES, pNext=None)
        _check_ze(ze.zeDeviceGetImageProperties(
            self._device, ctypes.byref(self._image_props)),
            "zeDeviceGetImageProperties")
        self._max_image_3d = self._image_props.maxImageDims3D
        # Xe-HPC (Ponte Vecchio) has no hardware texture/sampler units —
        # maxSamplers == 0 means filtered image reads would be driver-emulated.
        self._has_hw_sampler = self._image_props.maxSamplers > 0

        # Get module properties (for fp64 support detection)
        self._module_props = ze_device_module_properties_t(
            stype=ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES, pNext=None)
        _check_ze(ze.zeDeviceGetModuleProperties(
            self._device, ctypes.byref(self._module_props)),
            "zeDeviceGetModuleProperties")
        # Device-dependent, so this shadows the class attribute rather than
        # replacing it. supports_f64 derives from it; there is no second flag
        # to keep in step.
        self.supported_dtypes = (
            _L0_SUPPORTED_DTYPES if self._module_props.fp64flags != 0
            else _L0_SUPPORTED_DTYPES - {f64})

        # Find compute queue group ordinal
        qg_count = ctypes.c_uint32(0)
        _check_ze(ze.zeDeviceGetCommandQueueGroupProperties(
            self._device, ctypes.byref(qg_count), None),
            "zeDeviceGetCommandQueueGroupProperties (count)")
        qg_props = (ze_command_queue_group_properties_t * qg_count.value)()
        for i in range(qg_count.value):
            qg_props[i].stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES
            qg_props[i].pNext = None
        _check_ze(ze.zeDeviceGetCommandQueueGroupProperties(
            self._device, ctypes.byref(qg_count), qg_props),
            "zeDeviceGetCommandQueueGroupProperties")

        self._compute_ordinal = None
        for i in range(qg_count.value):
            if qg_props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE:
                self._compute_ordinal = i
                break
        if self._compute_ordinal is None:
            raise RuntimeError("No compute queue group found on Level Zero device")
        self._max_fill_pattern = qg_props[self._compute_ordinal].maxMemoryFillPatternSize

    def _create_queues(self, ze):
        # Level Zero command lists and kernel argument state may not be used
        # from two threads at once, and the queue below is shared by every
        # dispatch, reduction and copy on this backend. Reentrant because a
        # dispatch copies its scalar pack into place while holding it.
        self._launch_lock = threading.RLock()

        # The backend's queue: one asynchronous immediate command list. Every
        # launch, fill and copy is appended to it, followed by a barrier, and
        # returns once queued; the host waits (`synchronize`) where it reads
        # an array. The barriers keep the commands in program order. Not the
        # in-order flag (ZE_COMMAND_QUEUE_FLAG_IN_ORDER): on driver
        # 25.05.32567 it cost about 100 us of device time per kernel, a
        # barrier about 5. Asynchronous where the driver has
        # zeCommandListHostSynchronize (API 1.6); otherwise synchronous,
        # every append completing at once.
        self._queue = ze_command_list_handle_t()
        result = ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
        if hasattr(ze, "zeCommandListHostSynchronize"):
            desc = ze_command_queue_desc_t(
                stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, pNext=None,
                ordinal=self._compute_ordinal, index=0, flags=0,
                mode=ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, priority=0)
            result = ze.zeCommandListCreateImmediate(
                self._context, self._device, ctypes.byref(desc),
                ctypes.byref(self._queue))
        self._queued = result == ZE_RESULT_SUCCESS
        if not self._queued:
            desc = ze_command_queue_desc_t(
                stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, pNext=None,
                ordinal=self._compute_ordinal, index=0, flags=0,
                mode=ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, priority=0)
            _check_ze(ze.zeCommandListCreateImmediate(
                self._context, self._device, ctypes.byref(desc),
                ctypes.byref(self._queue)),
                "zeCommandListCreateImmediate")

        # Host memory queued copies and fills read, held until the queue has
        # passed them, as (object, bytes). Waiting releases it all, a marker
        # what was held before it; past these limits a hold waits first, so
        # a loop that never reads holds a bounded amount. `_held_released`
        # counts the holds released so far, the position of `_held[0]`.
        self._held = []
        self._held_bytes = 0
        self._held_released = 0

        # USM host memory for copies between the host and the device.
        self._staging = _HostStaging(self._context, self._launch_lock)

        # Host-visible events for markers (`_ListMarker`), reused once
        # waited for; a pool of `_EVENT_POOL_SIZE` more when they run out.
        self._events = []
        self._event_pools = []

        # A freed array may still be read by a queued launch. With the
        # driver's deferred free, it frees the memory once the work queued
        # so far is done; without it, freeing waits for the queue.
        self._deferred_free = "ZE_extension_memory_free_policies" in _extensions(
            ze, self._driver)

    _HOLD_COUNT = 1024
    _HOLD_BYTES = 256 << 20
    _EVENT_POOL_SIZE = 64

    def _ordered(self, result, what):
        """Check an append to the queue, then put a barrier after it, so
        nothing appended later starts before it completes. Called under
        `_launch_lock`."""
        _check_ze(result, what)
        if self._queued:
            _check_ze(_get_ze().zeCommandListAppendBarrier(self._queue, None, 0, None),
                       "zeCommandListAppendBarrier")

    def _hold(self, obj, nbytes):
        """Keep host memory a queued copy or fill reads until it has run.
        Called under `_launch_lock`, after the append."""
        if not self._queued:                    # the copy is done
            _release(obj)
            return
        self._held.append((obj, nbytes))
        self._held_bytes += nbytes
        if len(self._held) > self._HOLD_COUNT or self._held_bytes > self._HOLD_BYTES:
            self.synchronize()

    def _release_held(self, upto):
        """Drop the holds before position ``upto`` (counted from the first
        ever made), which the queue has passed. Called under `_launch_lock`."""
        count = upto - self._held_released
        if count <= 0:
            return
        released = self._held[:count]
        del self._held[:count]
        self._held_released = upto
        self._held_bytes -= sum(nbytes for _, nbytes in released)
        for obj, _ in released:
            _release(obj)

    def synchronize(self):
        """Wait for every launch, fill and copy queued on this backend. A
        launch's error surfaces here, at the next host access, rather than
        at its call."""
        with self._launch_lock:
            if self._queued:
                _check_ze(_get_ze().zeCommandListHostSynchronize(
                    self._queue, 0xFFFFFFFFFFFFFFFF),
                    "zeCommandListHostSynchronize")
            self._release_held(self._held_released + len(self._held))

    def async_marker(self):
        """A mark after the work queued so far, for `tack.sync_async` to
        wait on in a worker thread; None on a synchronous list, whose work
        is already done."""
        return self._mark()

    def _mark(self):
        """`async_marker`, which reads wait on outside the lock; theirs, so
        that replacing the public hook cannot leave a read unwaited."""
        if not self._queued:
            return None
        with self._launch_lock:
            return _ListMarker(self)

    def _take_event(self):
        """A reset host-visible event, from a new pool if every one is in
        use. Called under `_launch_lock`."""
        if not self._events:
            ze = _get_ze()
            desc = ze_event_pool_desc_t(
                stype=ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, pNext=None,
                flags=ZE_EVENT_POOL_FLAG_HOST_VISIBLE, count=self._EVENT_POOL_SIZE)
            devices = (ze_device_handle_t * 1)(getattr(self._device, "value", self._device))
            pool = ze_event_pool_handle_t()
            _check_ze(ze.zeEventPoolCreate(self._context, ctypes.byref(desc), 1, devices,
                                            ctypes.byref(pool)), "zeEventPoolCreate")
            self._event_pools.append(pool)
            for index in range(self._EVENT_POOL_SIZE):
                event_desc = ze_event_desc_t(
                    stype=ZE_STRUCTURE_TYPE_EVENT_DESC, pNext=None, index=index,
                    signal=ZE_EVENT_SCOPE_FLAG_HOST, wait=ZE_EVENT_SCOPE_FLAG_HOST)
                event = ze_event_handle_t()
                _check_ze(ze.zeEventCreate(pool, ctypes.byref(event_desc),
                                            ctypes.byref(event)), "zeEventCreate")
                self._events.append(event)
        return self._events.pop()

    def _free(self, ptr):
        """Free device memory after the work queued so far, which may use it.

        Under `_launch_lock`: the deferred free looks at what the queue has
        yet to run, and two threads freeing at once, or one freeing while
        another appends, crashed the driver (25.05; four threads launching
        and dropping arrays, within a few hundred frees). Reentrant, so an
        array collected while its thread holds the lock is freed in turn.
        """
        ze = _get_ze()
        with self._launch_lock:
            if self._deferred_free:
                desc = ze_memory_free_ext_desc_t(
                    stype=ZE_STRUCTURE_TYPE_MEMORY_FREE_EXT_DESC, pNext=None,
                    freePolicy=ZE_DRIVER_MEMORY_FREE_POLICY_EXT_FLAG_DEFER_FREE)
                _check_ze(ze.zeMemFreeExt(self._context, ctypes.byref(desc), ptr),
                           "zeMemFreeExt")
            else:
                self.synchronize()
                ze.zeMemFree(self._context, ptr)

    def allocate_array(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> L0Buffer:
        return L0Buffer(self, dtype.numpy_dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing Level Zero device pointer without allocating or copying."""
        buf = L0Buffer.__new__(L0Buffer)
        buf._backend = self
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(math.prod(shape)) * buf._numpy_dtype.itemsize
        buf._device_ptr = ctypes.c_void_p(int(ptr))
        buf._owned = False
        buf._shared = True      # someone else's memory: launches on it complete
        return buf

    def texture_in_hardware(self, shape_3d) -> bool:
        max_dim = self._max_image_3d
        return self._has_hw_sampler and all(s <= max_dim for s in shape_3d)

    def create_texture_image(self, shape_3d) -> L0TextureImage:
        return L0TextureImage(shape_3d, self)

    _texture_image_type = L0TextureImage

    def _launch_guard(self, variant):
        # The pack, the kernel's argument state and the command list are
        # backend-wide, so the whole launch holds the backend's lock.
        return self._launch_lock

    def _launch(self, compiled, kernel_args, loop_end, extents):
        compiled(kernel_args, loop_end, self, extents)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledL0Kernel:
        """Compile Tack IR → OpenCL C → SPIR-V → ze_module → ze_kernel."""
        kernel_name = kernel_entry_name(ir_func.name)
        workgroup_size = min(WORKGROUP_SIZE, self._compute_props.maxGroupSizeX,
                             self._compute_props.maxTotalGroupSize)
        full_groups = requires_full_workgroups(ir_func)
        if full_groups:
            check_workgroup_launch(ir_func.name, 0, backend_label=self.label,
                                   workgroup_size=workgroup_size)

        # Generate OpenCL C source
        opencl_source = generate_opencl_source(ir_func)

        # Compile to SPIR-V via libocloc
        spirv_bytes = _compile_to_spirv(opencl_source, self._device_id)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_array = [p._is_array for p in ir_func.params]
        param_is_texture = [p._is_texture for p in ir_func.params]
        return _load_kernel(self, spirv_bytes, kernel_name, param_types, param_is_array,
                            param_is_texture, workgroup_size, full_groups,
                            opencl_source=opencl_source)

    # ── The disk cache: native binaries, with SPIR-V to fall back on ───
    # zeModuleCreate has IGC finalize SPIR-V for the device; the native
    # binary is what it made, and loads without IGC. So the target names the
    # device and its IP version (which includes the stepping), the driver
    # that finalizes, the ocloc and options that made the SPIR-V, and
    # whether the device has f64 (it changes supported_dtypes). Software
    # texture sampling is decided before the variant key, and the device
    # names whether it happens.

    _target = None

    def cache_target(self):
        if self._target is None:
            ze = _get_ze()
            props = ze_driver_properties_t(
                stype=ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES, pNext=None)
            _check_ze(ze.zeDriverGetProperties(self._driver, ctypes.byref(props)),
                       "zeDriverGetProperties")
            ip = ze_device_ip_version_ext_t(
                stype=ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT, pNext=None)
            dev_props = ze_device_properties_t(
                stype=ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES,
                pNext=ctypes.cast(ctypes.pointer(ip), ctypes.c_void_p))
            _check_ze(ze.zeDeviceGetProperties(self._device, ctypes.byref(dev_props)),
                       "zeDeviceGetProperties (IP version)")
            options = " ".join(o.decode() for o in _OCLOC_OPTIONS)
            fp64 = "f64" if f64 in self.supported_dtypes else "no f64"
            self._target = (f"{self._device_name} 0x{self._device_id:04x} "
                            f"ip {ip.ipVersion:#x}; "
                            f"driver {props.driverVersion:#x}; "
                            f"ocloc {_ocloc_version()} {options}; "
                            f"module {_MODULE_BUILD_FLAGS.decode()}; {fp64}")
        return self._target

    def variant_artifact(self, payload):
        compiled, pack_info, _ = payload
        if compiled.spirv is None:
            raise ValueError("no SPIR-V was kept for this kernel")
        return {"native": _native_binary(compiled._module),
                "spirv": compiled.spirv, "name": compiled._func_name,
                "param_types": compiled._param_types,
                "param_is_array": compiled._param_is_array,
                "param_is_texture": compiled._param_is_texture,
                "workgroup_size": compiled._workgroup_size,
                "full_workgroups": compiled._requires_full_workgroups,
                "pack_info": pack_info}

    def load_variant(self, ir_func, artifact, effective_args):
        from tack.runtime.kernel_utils import _create_pack_arrays
        # Module creation outside _launch_lock, as _compile_kernel does it.
        compiled = _load_kernel(self, artifact["spirv"], artifact["name"],
                                artifact["param_types"], artifact["param_is_array"],
                                artifact["param_is_texture"], artifact["workgroup_size"],
                                artifact["full_workgroups"], native=artifact["native"])
        pack_info = artifact["pack_info"]
        # The scalar pack buffers are device memory of this process, made
        # from this call's scalars as _build_variant makes them.
        pack_arrays = (_create_pack_arrays(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_arrays

    def reduce_array(self, array, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        if array.size == 0:
            return empty_reduction(op)
        if (array.dtype is not f32
                or self._compute_props.maxGroupSizeX < 256
                or self._compute_props.maxTotalGroupSize < 256):
            return reduce_numpy(array.numpy(), op)
        check_launch_size(f"Array {op}()", array.size, self._max_launch, self.label)

        if not hasattr(self, '_reduce_cache'):
            self._reduce_cache = {}
        if op not in self._reduce_cache:
            self._reduce_cache[op] = self._compile_reduce(op)

        compiled_kernel = self._reduce_cache[op]
        n = int(math.prod(array.shape))

        # Create output buffer with init value
        init_vals = REDUCTION_IDENTITIES
        out_np = np.array([init_vals[op]], dtype=np.float32)
        out_buf = L0Buffer(self, np.float32, (1,))
        out_buf.from_numpy(out_np)

        # Dispatch
        ze = _get_ze()
        kernel = compiled_kernel._kernel
        block_dim = compiled_kernel._workgroup_size

        # The reduction kernel's arguments and the command list are shared.
        with self._launch_lock:
            # Set arguments: input, output, __n__
            in_ptr = array._buffer.device_ptr
            out_ptr = out_buf.device_ptr
            _check_ze(ze.zeKernelSetArgumentValue(
                kernel, 0, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(in_ptr)),
                "zeKernelSetArgumentValue (reduce input)")
            _check_ze(ze.zeKernelSetArgumentValue(
                kernel, 1, ctypes.sizeof(ctypes.c_void_p), ctypes.byref(out_ptr)),
                "zeKernelSetArgumentValue (reduce output)")
            n_val = ctypes.c_longlong(n)
            _check_ze(ze.zeKernelSetArgumentValue(
                kernel, 2, ctypes.sizeof(n_val), ctypes.byref(n_val)),
                "zeKernelSetArgumentValue (reduce __n__)")

            _check_ze(ze.zeKernelSetGroupSize(kernel, block_dim, 1, 1),
                       "zeKernelSetGroupSize")

            group_count_x = (n + block_dim - 1) // block_dim
            group_count = ze_group_count_t(
                groupCountX=group_count_x, groupCountY=1, groupCountZ=1)

            self._ordered(ze.zeCommandListAppendLaunchKernel(
                self._queue, kernel, ctypes.byref(group_count), None, 0, None),
                "zeCommandListAppendLaunchKernel")

        return float(out_buf.to_numpy()[0])     # queued after the launch, and waited for

    def _compile_reduce(self, op: str) -> CompiledL0Kernel:
        """Compile a reduction kernel for the given op."""
        ze = _get_ze()
        _REDUCE_SRC = {"sum": _REDUCE_OCL_SUM, "min": _REDUCE_OCL_MIN, "max": _REDUCE_OCL_MAX}
        func_names = {"sum": "reduce_sum_f32", "min": "reduce_min_f32", "max": "reduce_max_f32"}

        src = _REDUCE_SRC[op]
        spirv = _compile_to_spirv(src, self._device_id)

        spirv_buf = (ctypes.c_uint8 * len(spirv)).from_buffer_copy(spirv)
        module_desc = ze_module_desc_t(
            stype=ZE_STRUCTURE_TYPE_MODULE_DESC, pNext=None,
            format=ZE_MODULE_FORMAT_IL_SPIRV,
            inputSize=len(spirv),
            pInputModule=ctypes.cast(spirv_buf, ctypes.c_void_p),
            pBuildFlags=_MODULE_BUILD_FLAGS,
            pConstants=None)

        module = ze_module_handle_t()
        build_log = ze_module_build_log_handle_t()
        result = ze.zeModuleCreate(
            self._context, self._device, ctypes.byref(module_desc),
            ctypes.byref(module), ctypes.byref(build_log))
        if result != ZE_RESULT_SUCCESS:
            raise RuntimeError(f"Reduce kernel compile failed (0x{result:08x})")
        if build_log:
            ze.zeModuleBuildLogDestroy(build_log)

        kernel_desc = ze_kernel_desc_t(
            stype=ZE_STRUCTURE_TYPE_KERNEL_DESC, pNext=None,
            flags=0, pKernelName=func_names[op].encode())
        kernel = ze_kernel_handle_t()
        _check_ze(ze.zeKernelCreate(module, ctypes.byref(kernel_desc),
                                     ctypes.byref(kernel)), "zeKernelCreate")

        workgroup_size = 256
        return CompiledL0Kernel(module, kernel, func_names[op],
                                [f32, f32, i64], [True, True, False],
                                workgroup_size)

    def __del__(self):
        # Level Zero cleanup at interpreter shutdown can segfault because
        # the driver may already be unloaded.  Safest to skip cleanup.
        pass

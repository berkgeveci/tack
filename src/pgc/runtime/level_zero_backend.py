"""PGC Level Zero backend — compiles kernels via ocloc and dispatches on Intel GPUs.

Pipeline:
    PGC IR → OpenCL C source → libocloc (SPIR-V) → zeModuleCreate → zeKernelCreate
           → zeCommandListAppendLaunchKernel

Fields are device-resident: ``pgc.field()`` allocates a device buffer via
``zeMemAllocDevice``.  Transfers are explicit:

    field.from_numpy(arr)   # host → device (zeCommandListAppendMemoryCopy)
    arr = field.to_numpy()  # device → host (zeCommandListAppendMemoryCopy)

No per-dispatch copies — data stays on the GPU between kernel calls.

Requires: libze_loader.so (Level Zero runtime), libocloc.so (Intel offline compiler).
No Python packages needed — uses ctypes directly.
"""

import ctypes
import ctypes.util

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.opencl_gen import generate_opencl_source

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
ZE_STRUCTURE_TYPE_CONTEXT_DESC = 0x0d
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC = 0x0e
ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC = 0x0f
ZE_STRUCTURE_TYPE_MODULE_DESC = 0x1b
ZE_STRUCTURE_TYPE_KERNEL_DESC = 0x1d
ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC = 0x15
ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC = 0x16
ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES = 0x03
ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES = 0x04
ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES = 0x06

# Module format
ZE_MODULE_FORMAT_IL_SPIRV = 0
ZE_MODULE_FORMAT_NATIVE = 1

# Command queue mode
ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS = 1

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

ZE_MAX_DEVICE_NAME = 256
ZE_MAX_DEVICE_UUID_SIZE = 16


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

    ze.zeModuleCreate.argtypes = [P, P, P, P, P]
    ze.zeModuleCreate.restype = ctypes.c_int32

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


def _compile_to_spirv(opencl_source: str, device_id: int) -> bytes:
    """Compile OpenCL C source to SPIR-V binary using libocloc in-process."""
    ocloc = _get_ocloc()

    # Null-terminate source and include the terminator in the length —
    # ocloc treats in-memory sources as C strings internally.
    src = opencl_source.encode("utf-8") + b"\x00"

    device_str = f"0x{device_id:04x}"
    args_list = [
        b"compile",
        b"-spv_only",
        b"-device", device_str.encode(),
        b"-options", b"-cl-std=CL2.0",
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


# ---------------------------------------------------------------------------
# L0Buffer
# ---------------------------------------------------------------------------
class L0Buffer(DeviceBuffer):
    """Device-resident buffer backed by a Level Zero device allocation.

    Data lives on the GPU.  ``from_numpy`` copies host→device,
    ``to_numpy`` copies device→host.
    """

    def __init__(self, backend, numpy_dtype, shape):
        self._backend = backend
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize

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

        # Zero-initialize via host staging buffer
        zeros = np.zeros(shape, dtype=self._numpy_dtype)
        self._copy_to_device(zeros)

    @property
    def device_ptr(self):
        return self._device_ptr

    def _copy_to_device(self, arr: np.ndarray):
        """Copy numpy array → device using an immediate command list."""
        ze = _get_ze()
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        _check_ze(ze.zeCommandListAppendMemoryCopy(
            self._backend._imm_cmd_list,
            self._device_ptr, src.ctypes.data, self._nbytes,
            None, 0, None),
            "zeCommandListAppendMemoryCopy (H2D)")

    def _copy_from_device(self, out: np.ndarray):
        """Copy device → numpy array using an immediate command list."""
        ze = _get_ze()
        _check_ze(ze.zeCommandListAppendMemoryCopy(
            self._backend._imm_cmd_list,
            out.ctypes.data, self._device_ptr, self._nbytes,
            None, 0, None),
            "zeCommandListAppendMemoryCopy (D2H)")

    def from_numpy(self, arr: np.ndarray):
        self._copy_to_device(arr)

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        self._copy_from_device(out)
        return out

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __del__(self):
        if hasattr(self, '_device_ptr') and self._device_ptr:
            try:
                ze = _get_ze()
                ze.zeMemFree(self._backend._context, self._device_ptr)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CompiledL0Kernel
# ---------------------------------------------------------------------------
_L0_CTYPES_MAP = {
    f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
    u32: ctypes.c_uint, u64: ctypes.c_ulonglong,
}


class CompiledL0Kernel:
    """A compiled Level Zero kernel ready for dispatch."""

    def __init__(self, module, kernel, func_name, param_types, param_is_field,
                 workgroup_size, param_is_texture=None, texture_shapes=None):
        self._module = module
        self._kernel = kernel
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._workgroup_size = workgroup_size
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self._texture_shapes = texture_shapes or {}  # param_index → (W, H, D)
        self._image_cache: dict[tuple, ctypes.c_void_p] = {}

    def _create_image(self, field, W, H, D, backend):
        """Create a Level Zero 3D image from a field's device buffer."""
        ze = _get_ze()
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

        # Device → host staging buffer
        _check_ze(ze.zeCommandListAppendMemoryCopy(
            backend._imm_cmd_list, host_ptr,
            field._buffer.device_ptr, nbytes,
            None, 0, None),
            "zeCommandListAppendMemoryCopy (D2H staging)")

        # Host staging buffer → image
        _check_ze(ze.zeCommandListAppendImageCopyFromMemory(
            backend._imm_cmd_list, image,
            host_ptr, None,
            None, 0, None),
            "zeCommandListAppendImageCopyFromMemory")

        # Free staging buffer
        ze.zeMemFree(backend._context, host_ptr)

        return image

    def __call__(self, kernel_args: list, loop_end: int, backend):
        """Dispatch the Level Zero kernel."""
        ze = _get_ze()
        kernel = self._kernel

        # Set kernel arguments
        arg_idx = 0
        for i, (arg, ptype, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field,
                    self._param_is_texture)):
            if is_tex:
                # Bind as image3d_t
                W, H, D = self._texture_shapes[i]
                cache_key = (arg._buffer.device_ptr.value, W, H, D)
                if cache_key not in self._image_cache:
                    self._image_cache[cache_key] = self._create_image(
                        arg, W, H, D, backend)
                img_handle = self._image_cache[cache_key]
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, arg_idx, ctypes.sizeof(ctypes.c_void_p),
                    ctypes.byref(img_handle)),
                    f"zeKernelSetArgumentValue (image arg {arg_idx})")
            elif is_field:
                ptr = arg._buffer.device_ptr
                _check_ze(ze.zeKernelSetArgumentValue(
                    kernel, arg_idx, ctypes.sizeof(ctypes.c_void_p),
                    ctypes.byref(ptr)),
                    f"zeKernelSetArgumentValue (field arg {arg_idx})")
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

        # Set workgroup size
        wg = self._workgroup_size
        _check_ze(ze.zeKernelSetGroupSize(kernel, wg, 1, 1),
                   "zeKernelSetGroupSize")

        # Calculate group count
        group_count_x = (loop_end + wg - 1) // wg
        group_count = ze_group_count_t(
            groupCountX=group_count_x,
            groupCountY=1,
            groupCountZ=1)

        # Append launch to command list, close, execute, sync, reset
        cmd_list = backend._cmd_list
        _check_ze(ze.zeCommandListReset(cmd_list), "zeCommandListReset")
        _check_ze(ze.zeCommandListAppendLaunchKernel(
            cmd_list, kernel, ctypes.byref(group_count),
            None, 0, None),
            "zeCommandListAppendLaunchKernel")
        _check_ze(ze.zeCommandListClose(cmd_list), "zeCommandListClose")

        cmd_lists = (ctypes.c_void_p * 1)(cmd_list)
        _check_ze(ze.zeCommandQueueExecuteCommandLists(
            backend._cmd_queue, 1, cmd_lists, None),
            "zeCommandQueueExecuteCommandLists")
        _check_ze(ze.zeCommandQueueSynchronize(
            backend._cmd_queue, 0xFFFFFFFFFFFFFFFF),
            "zeCommandQueueSynchronize")


# ---------------------------------------------------------------------------
# LevelZeroBackend
# ---------------------------------------------------------------------------
def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


class LevelZeroBackend:
    """Level Zero GPU backend — device-resident fields, ocloc compilation."""

    def __init__(self):
        ze = _get_ze()

        # Initialize Level Zero
        _check_ze(ze.zeInit(0), "zeInit")

        # Get first driver
        count = ctypes.c_uint32(0)
        _check_ze(ze.zeDriverGet(ctypes.byref(count), None), "zeDriverGet (count)")
        if count.value == 0:
            raise RuntimeError("No Level Zero drivers found")
        drivers = (ze_driver_handle_t * count.value)()
        _check_ze(ze.zeDriverGet(ctypes.byref(count), drivers), "zeDriverGet")
        self._driver = drivers[0]

        # Get first device
        count = ctypes.c_uint32(0)
        _check_ze(ze.zeDeviceGet(self._driver, ctypes.byref(count), None),
                   "zeDeviceGet (count)")
        if count.value == 0:
            raise RuntimeError("No Level Zero devices found")
        devices = (ze_device_handle_t * count.value)()
        _check_ze(ze.zeDeviceGet(self._driver, ctypes.byref(count), devices),
                   "zeDeviceGet")
        self._device = devices[0]

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

        # Create context
        ctx_desc = ze_context_desc_t(
            stype=ZE_STRUCTURE_TYPE_CONTEXT_DESC, pNext=None, flags=0)
        self._context = ze_context_handle_t()
        _check_ze(ze.zeContextCreate(self._driver, ctypes.byref(ctx_desc),
                                      ctypes.byref(self._context)),
                   "zeContextCreate")

        # Create command queue (synchronous mode for simplicity)
        queue_desc = ze_command_queue_desc_t(
            stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, pNext=None,
            ordinal=self._compute_ordinal, index=0, flags=0,
            mode=ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, priority=0)
        self._cmd_queue = ze_command_queue_handle_t()
        _check_ze(ze.zeCommandQueueCreate(
            self._context, self._device, ctypes.byref(queue_desc),
            ctypes.byref(self._cmd_queue)),
            "zeCommandQueueCreate")

        # Create a reusable command list for kernel dispatch
        list_desc = ze_command_list_desc_t(
            stype=ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, pNext=None,
            commandQueueGroupOrdinal=self._compute_ordinal, flags=0)
        self._cmd_list = ze_command_list_handle_t()
        _check_ze(ze.zeCommandListCreate(
            self._context, self._device, ctypes.byref(list_desc),
            ctypes.byref(self._cmd_list)),
            "zeCommandListCreate")

        # Create an immediate command list for memory copies
        imm_desc = ze_command_queue_desc_t(
            stype=ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, pNext=None,
            ordinal=self._compute_ordinal, index=0, flags=0,
            mode=ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, priority=0)
        self._imm_cmd_list = ze_command_list_handle_t()
        _check_ze(ze.zeCommandListCreateImmediate(
            self._context, self._device, ctypes.byref(imm_desc),
            ctypes.byref(self._imm_cmd_list)),
            "zeCommandListCreateImmediate")

        self._cache: dict[str, CompiledL0Kernel] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> L0Buffer:
        return L0Buffer(self, dtype.numpy_dtype, shape)

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Level Zero GPU."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        # Detect template arguments and expand them
        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args, _detect_texture_fields,
        )
        from pgc.lang.field import Texture3D
        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        # Detect vector and texture fields
        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
        texture_fields = _detect_texture_fields(kernel, args, template_args)

        # Get IR
        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
            texture_fields=texture_fields,
        )
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes and texture shapes
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference
        infer_param_types(ir_func, effective_args)

        # Store texture shapes on params for codegen/dispatch.
        # Fall back to software trilinear if the device lacks hardware samplers
        # (Xe-HPC) or any dimension exceeds the device's image size limit.
        max_dim = self._max_image_3d
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                W, H, D = arg.shape_3d
                if (self._has_hw_sampler
                        and W <= max_dim and H <= max_dim and D <= max_dim):
                    param._texture_shape = arg.shape_3d
                else:
                    param._is_texture = False  # software fallback

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Cache key (include texture shapes for uniqueness)
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tex_sig = tuple(
            getattr(p, '_texture_shape', None) for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tex_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            self._cache[cache_key] = self._compile_kernel(ir_func)

        compiled = self._cache[cache_key]

        # Build kernel args list — unwrap Texture3D to underlying Field
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]

        # Determine loop range
        loop_end = _get_loop_range(ir_func, kernel_args)

        # Dispatch
        compiled(kernel_args, loop_end, self)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledL0Kernel:
        """Compile PGC IR → OpenCL C → SPIR-V → ze_module → ze_kernel."""
        ze = _get_ze()
        workgroup_size = min(256, self._compute_props.maxGroupSizeX)

        # Generate OpenCL C source
        opencl_source = generate_opencl_source(ir_func)

        # Compile to SPIR-V via libocloc
        spirv_bytes = _compile_to_spirv(opencl_source, self._device_id)

        # Create module from SPIR-V
        spirv_buf = (ctypes.c_uint8 * len(spirv_bytes)).from_buffer_copy(spirv_bytes)
        module_desc = ze_module_desc_t(
            stype=ZE_STRUCTURE_TYPE_MODULE_DESC, pNext=None,
            format=ZE_MODULE_FORMAT_IL_SPIRV,
            inputSize=len(spirv_bytes),
            pInputModule=ctypes.cast(spirv_buf, ctypes.c_void_p),
            pBuildFlags=b"-cl-std=CL2.0",
            pConstants=None)

        module = ze_module_handle_t()
        build_log = ze_module_build_log_handle_t()
        result = ze.zeModuleCreate(
            self._context, self._device, ctypes.byref(module_desc),
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
            raise RuntimeError(
                f"zeModuleCreate failed (0x{result:08x}):\n{log_text}\n"
                f"Source:\n{opencl_source}")

        if build_log:
            ze.zeModuleBuildLogDestroy(build_log)

        # Create kernel
        kernel_desc = ze_kernel_desc_t(
            stype=ZE_STRUCTURE_TYPE_KERNEL_DESC, pNext=None,
            flags=0, pKernelName=ir_func.name.encode())
        kernel = ze_kernel_handle_t()
        _check_ze(ze.zeKernelCreate(module, ctypes.byref(kernel_desc),
                                     ctypes.byref(kernel)),
                   "zeKernelCreate")

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape
        return CompiledL0Kernel(module, kernel, ir_func.name,
                                param_types, param_is_field, workgroup_size,
                                param_is_texture, texture_shapes)

    def __del__(self):
        # Level Zero cleanup at interpreter shutdown can segfault because
        # the driver may already be unloaded.  Safest to skip cleanup.
        pass

"""Tack CUDA backend — compiles kernels via NVRTC and dispatches on NVIDIA GPUs.

Pipeline:
    Tack IR → CUDA C source → NVRTC (PTX) → cuModuleLoad → cuLaunchKernel

Fields are device-resident: ``tack.field()`` allocates a device buffer via
``cuMemAlloc``.  Transfers are explicit:

    field.from_numpy(arr)   # host → device (cuMemcpyHtoD)
    arr = field.to_numpy()  # device → host (cuMemcpyDtoH)

No per-dispatch copies — data stays on the GPU between kernel calls.
"""

import ctypes
import os
import sys

import numpy as np

from tack.codegen.reductions import field_reduction_source
from tack.lang import ir
from tack.lang.field import DeviceBuffer
from tack.lang.types import ScalarType, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import WORKGROUP_SIZE
from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    _get_loop_range,
    as_address,
    new_kernel_cache,
    resolve_variant,
)
from tack.runtime.reductions import REDUCTION_IDENTITIES, empty_reduction, reduce_numpy

_CUDA_SUPPORTED_DTYPES = {i8, u8, i16, u16, i32, u32, i64, u64, f32, f64}
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


_REDUCE_CUDA_SUM = field_reduction_source('cuda', 'sum')
_REDUCE_CUDA_MIN = field_reduction_source('cuda', 'min')
_REDUCE_CUDA_MAX = field_reduction_source('cuda', 'max')


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
    ``tack.init()`` builds the new backend before dropping the old one, so two
    of them routinely overlap, and the context must survive until the last one
    goes. ``owned`` is False for a context an embedding application created:
    Tack adopts those but never destroys them, so their token never dies here.
    """

    __slots__ = ("alive", "handle", "owned", "users")

    def __init__(self, handle, owned):
        self.handle = handle
        self.users = 1
        self.alive = True
        self.owned = owned


# Every CUDA context Tack is currently aware of, keyed by handle. Contexts an
# embedding application created are in here too, marked ``owned=False``, so
# that buffers allocated in them still get a token to hold.
_CONTEXTS: dict[int, _ContextToken] = {}


def _current_context_token():
    """Token for the context that is current right now, or None if untracked."""
    err, ctx = driver.cuCtxGetCurrent()
    if err != driver.CUresult.CUDA_SUCCESS or int(ctx) == 0:
        return None
    return _CONTEXTS.get(int(ctx))


_DEAD_CONTEXT_MSG = (
    "the CUDA context this field was allocated in has been destroyed. "
    "Switching backends -- tack.init(arch=tack.cpu) after tack.init("
    "arch=tack.cuda) -- tears down the CUDA context, and every device "
    "pointer allocated in it dies with the context. Fields do not survive "
    "that; allocate them again after switching back."
)


class CUDABuffer(DeviceBuffer):
    """Device-resident buffer backed by a CUDA device pointer.

    Data lives on the GPU.  ``from_numpy`` copies host→device,
    ``to_numpy`` copies device→host.
    """

    backend_name = "cuda"

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        self._token = _current_context_token()
        err, self._device_ptr = driver.cuMemAlloc(self._nbytes)
        _check(err)
        # Zero-initialise
        _check(driver.cuMemsetD8(self._device_ptr, 0, self._nbytes))

    def _live(self, verb):
        """Refuse to touch device memory whose context is gone."""
        token = getattr(self, "_token", None)
        if token is not None and not token.alive:
            raise RuntimeError(f"Cannot {verb} this CUDA field: {_DEAD_CONTEXT_MSG}")

    @property
    def address(self) -> int:
        return int(self._device_ptr)

    @property
    def device_ptr(self):
        # Guarded because this is what a kernel launch reads: without the
        # check a dispatch against a stale field faults in the driver.
        self._live("run a kernel against")
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        self._live("write to")
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        _check(driver.cuMemcpyHtoD(self._device_ptr, src, self._nbytes))

    def to_numpy(self) -> np.ndarray:
        self._live("read")
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        _check(driver.cuMemcpyDtoH(out, self._device_ptr, self._nbytes))
        return out

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def export_memory(self):
        """Export as ExportedMemory. Lazily copies into exportable memory."""
        self._live("export")
        if not hasattr(self, '_export_buf'):
            self._export_buf = ExportableCUDABuffer(self._numpy_dtype, self._shape)
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
                driver.cuMemFree(self._device_ptr)
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

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize

        # Set before anything can fail, so __del__ can tell what was reached.
        self._mem_handle = None
        self._device_ptr = None
        self._exported_handle = None
        self._handle_type_name = None
        self._token = _current_context_token()

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
            + "\nExportable memory is only needed by Field.export_memory() for "
              "sharing with another API; ordinary tack.field() allocations are "
              "unaffected."
        )

    @property
    def address(self) -> int:
        return int(self._device_ptr)

    @property
    def device_ptr(self):
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        _check(driver.cuMemcpyHtoD(self._device_ptr, src, self._nbytes))

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        _check(driver.cuMemcpyDtoH(out, self._device_ptr, self._nbytes))
        return out

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def export_memory(self):
        """Export as ExportedMemory (handle + size + UUID). The handle is cached."""
        from tack.lang.field import ExportedMemory
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
            if self._device_ptr is not None:
                driver.cuMemUnmap(self._device_ptr, self._alloc_size)
                driver.cuMemAddressFree(self._device_ptr, self._alloc_size)
            if self._mem_handle is not None:
                driver.cuMemRelease(self._mem_handle)
        except Exception:
            pass


def _compile_ptx(cuda_source: str, func_name: str) -> bytes:
    """Compile CUDA C source to PTX via NVRTC."""
    src = cuda_source.encode("utf-8")
    err, prog = nvrtc.nvrtcCreateProgram(src, f"{func_name}.cu".encode(), 0, None, None)
    _check(err)

    # Preserve NaNs, signed zeros, and expression grouping in every kernel.
    # Adjacent multiply/add contraction is permitted by the language contract.
    opts = [b"--ftz=false", b"--prec-div=true", b"--prec-sqrt=true",
            b"--fmad=true", b"--extra-device-vectorization"]
    # cuda-python marshals a list of bytes; a ctypes array can be misread.
    compile_result = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
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


_CUDA_CTYPES_MAP = {f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
                    u32: ctypes.c_uint, u64: ctypes.c_ulonglong}


class CompiledCUDAKernel:
    """A compiled CUDA kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types, param_is_field,
                 param_is_texture=None, texture_shapes=None):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self._texture_shapes = texture_shapes or {}  # param_index → (W, H, D)
        self._tex_cache: dict[tuple, int] = {}  # cache_key → CUtexObject

    def _create_texture_object(self, field, W, H, D):
        """Create a CUDA texture object from a field's device buffer.

        Allocates a CUDA 3D array, copies the field data into it, then creates
        a texture object with linear filtering and normalized coordinates.
        """
        # Create a CUDA array descriptor for a 3D float texture
        array_desc = driver.CUDA_ARRAY3D_DESCRIPTOR()
        array_desc.Width = W
        array_desc.Height = H
        array_desc.Depth = D
        array_desc.Format = driver.CUarray_format.CU_AD_FORMAT_FLOAT
        array_desc.NumChannels = 1
        array_desc.Flags = 0

        err, cuda_array = driver.cuArray3DCreate(array_desc)
        _check(err)

        # Copy field data (device linear buffer) → CUDA 3D array
        copy_params = driver.CUDA_MEMCPY3D()
        # Source: device pointer, pitched linear memory
        copy_params.srcMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_DEVICE
        copy_params.srcDevice = field._buffer.device_ptr
        copy_params.srcPitch = W * 4   # bytes per row
        copy_params.srcHeight = H
        # Destination: CUDA array
        copy_params.dstMemoryType = driver.CUmemorytype.CU_MEMORYTYPE_ARRAY
        copy_params.dstArray = cuda_array
        # Extent
        copy_params.WidthInBytes = W * 4
        copy_params.Height = H
        copy_params.Depth = D

        _check(driver.cuMemcpy3D(copy_params))

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
        res_desc.res.array.hArray = cuda_array

        # Create resource view descriptor (default — full mip level 0)
        view_desc = driver.CUDA_RESOURCE_VIEW_DESC()
        view_desc.format = driver.CUresourceViewFormat.CU_RES_VIEW_FORMAT_FLOAT_1X32
        view_desc.width = W
        view_desc.height = H
        view_desc.depth = D

        err, tex_obj = driver.cuTexObjectCreate(res_desc, tex_desc, view_desc)
        _check(err)

        return tex_obj, cuda_array

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the CUDA kernel."""
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for i, (arg, ptype, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field,
                    self._param_is_texture)):
            if is_tex:
                W, H, D = self._texture_shapes[i]
                cache_key = (int(arg._buffer.device_ptr), W, H, D)
                if cache_key not in self._tex_cache:
                    tex_obj, cuda_array = self._create_texture_object(arg, W, H, D)
                    self._tex_cache[cache_key] = (tex_obj, cuda_array)
                tex_obj, _ = self._tex_cache[cache_key]
                # cudaTextureObject_t is unsigned long long (64-bit handle)
                arg_values.append(ctypes.c_ulonglong(int(tex_obj)))
            elif is_field:
                arg_values.append(ctypes.c_void_p(int(arg._buffer.device_ptr)))
            else:
                ct = _CUDA_CTYPES_MAP[ptype]
                arg_values.append(ct(arg))
        arg_values.append(n_val)

        arg_ptrs = (ctypes.c_void_p * len(arg_values))()
        for i, val in enumerate(arg_values):
            arg_ptrs[i] = ctypes.addressof(val)

        block_dim = WORKGROUP_SIZE
        grid_dim = (loop_end + block_dim - 1) // block_dim

        _check(driver.cuLaunchKernel(
            self._func,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0, 0,
            arg_ptrs, 0,
        ))
        _check(driver.cuCtxSynchronize())


class CUDABackend(Backend):
    """CUDA GPU backend — device-resident fields, NVRTC compilation."""

    name = "cuda"
    display_name = "CUDA"
    supported_dtypes = _CUDA_SUPPORTED_DTYPES
    supports_device_reductions = True
    supports_workgroups = True
    device_memory_spaces = frozenset({"cuda", "cuda_pinned", "cuda_managed"})


    def __init__(self):
        _check(driver.cuInit(0))
        err, self._device = driver.cuDeviceGet(0)
        _check(err)

        # Reuse an existing CUDA context if one is already active (e.g. from
        # a simulation framework like AMReX).  Only create a new context when
        # no current context exists.
        #
        # Whether the context may be destroyed is a property of the context,
        # not of the backend that happens to hold it: `tack.init(arch=...)`
        # builds the new backend before dropping the old one, so two
        # CUDABackend objects routinely share one context for a moment. When
        # that context is ours, both must agree that the *last* one out
        # destroys it -- an adopter that recorded `_owns_context = False`
        # would be left holding a destroyed context as soon as its creator
        # was collected, and every later call would fail with
        # CUDA_ERROR_INVALID_CONTEXT. Refcounting the context gets that
        # right while leaving a foreign context untouched, which is the
        # whole point of adopting one.
        err, ctx = driver.cuCtxGetCurrent()
        if err == driver.CUresult.CUDA_SUCCESS and int(ctx) != 0:
            self._context = ctx
            token = _CONTEXTS.get(int(ctx))
            if token is None:
                # Nobody here created this one, so it belongs to the embedding
                # application: adopt it, but never destroy it.
                token = _ContextToken(ctx, owned=False)
                _CONTEXTS[int(ctx)] = token
            else:
                token.users += 1
            self._token = token
            self._owns_context = token.owned
        else:
            err, self._context = driver.cuCtxCreate(None, 0, self._device)
            _check(err)
            self._token = _ContextToken(self._context, owned=True)
            _CONTEXTS[int(self._context)] = self._token
            self._owns_context = True

        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledCUDAKernel}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> CUDABuffer:
        if exportable:
            return ExportableCUDABuffer(dtype.numpy_dtype, shape)
        return CUDABuffer(dtype.numpy_dtype, shape)

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
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        buf._device_ptr = ptr  # integer or CUdeviceptr
        buf._owned = False
        buf._token = _current_context_token()
        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the CUDA GPU.

        The IR passes and device compilation run only when this argument
        shape/type combination is new; see `resolve_variant`.
        """
        from tack.lang.field import Texture3D

        variant, effective_args = resolve_variant(
            self, kernel, args, kwargs,
            build=self._build_variant,
        )
        compiled, pack_info, pack_fields = variant.payload

        # Loop range comes from the pre-packing IR: packing rewrites the
        # parameter list, and the range expression names the original args.
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(variant.ir, kernel_args)
        if loop_end <= 0:
            # range(0) runs nothing, and cuLaunchKernel rejects an empty grid.
            return

        # Replace scalar args with the packed field buffers. The buffers
        # belong to the variant, so writing them and the launch that reads
        # them happen under its lock (see KernelVariant.dispatch_lock).
        with variant.dispatch_lock:
            if pack_info:
                from tack.lang.ir_pack_scalars import split_args
                from tack.runtime.kernel_utils import _update_pack_fields
                _update_pack_fields(pack_fields, pack_info, effective_args)
                kept_args = split_args(effective_args, pack_info)
                kernel_args = [a.field if isinstance(a, Texture3D) else a
                               for a in kept_args]
                kernel_args = list(kernel_args) + pack_fields

            compiled(kernel_args, loop_end)

    def _build_variant(self, ir_func, effective_args):
        """Pack scalars, annotate, compile. Runs once per variant.

        Packing rewrites the parameter list, so it works on its own copy —
        the caller keeps `ir_func` for loop-range resolution.
        """
        from tack.lang.ir_pack_scalars import pack_scalars
        from tack.lang.ir_traversal import clone_ir
        from tack.lang.ir_type_annotate import annotate_types
        from tack.lang.ir_verify import verify_ir
        from tack.runtime.kernel_utils import _create_pack_fields

        packed = clone_ir(ir_func)
        _, pack_info = pack_scalars(packed, effective_args)
        verify_ir(packed, 'packed')
        annotate_types(packed)
        verify_ir(packed, 'typed')
        compiled = self._compile_kernel(packed)
        pack_fields = (_create_pack_fields(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_fields

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledCUDAKernel:
        """Compile Tack IR → CUDA C → PTX → CUfunction."""
        kernel_name = kernel_entry_name(ir_func.name)
        cuda_source = generate_cuda_source(ir_func)
        ptx = _compile_ptx(cuda_source, kernel_name)

        err, module = driver.cuModuleLoadData(ptx)
        _check(err)

        err, func = driver.cuModuleGetFunction(module, kernel_name.encode())
        _check(err)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape
        return CompiledCUDAKernel(module, func, kernel_name, param_types,
                                  param_is_field, param_is_texture, texture_shapes)

    def reduce_field(self, field, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        from tack.lang.types import f32
        if field.size == 0:
            return empty_reduction(op)
        if field.dtype is not f32:
            return reduce_numpy(field.to_numpy(), op)

        if not hasattr(self, '_reduce_cache'):
            self._reduce_cache = {}
        if op not in self._reduce_cache:
            self._reduce_cache[op] = self._compile_reduce(op)

        func, module = self._reduce_cache[op]
        n = int(np.prod(field.shape))

        # Output: [result, n_as_uint_bits]
        import struct as _struct
        init_vals = REDUCTION_IDENTITIES
        out_np = np.array([init_vals[op],
                           np.frombuffer(_struct.pack('I', n), dtype=np.float32)[0]],
                          dtype=np.float32)
        out_buf = CUDABuffer(np.float32, (2,))
        out_buf.from_numpy(out_np)

        block_dim = 256
        grid_dim = (n + block_dim - 1) // block_dim

        # Dispatch
        in_ptr = ctypes.c_void_p(int(field._buffer.device_ptr))
        out_ptr = ctypes.c_void_p(int(out_buf.device_ptr))
        args = (ctypes.c_void_p * 2)()
        args[0] = ctypes.addressof(in_ptr)
        args[1] = ctypes.addressof(out_ptr)

        _check(driver.cuLaunchKernel(
            func, grid_dim, 1, 1, block_dim, 1, 1,
            block_dim * 4, 0, args, 0))
        _check(driver.cuCtxSynchronize())

        result = out_buf.to_numpy()
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
            if token.owned:
                # Mark dead before destroying, so any buffer still holding
                # this token reports the problem instead of faulting.
                token.alive = False
                driver.cuCtxDestroy(token.handle)
        except Exception:
            pass

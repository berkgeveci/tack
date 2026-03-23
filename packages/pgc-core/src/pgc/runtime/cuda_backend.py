"""PGC CUDA backend — compiles kernels via NVRTC and dispatches on NVIDIA GPUs.

Pipeline:
    PGC IR → CUDA C source → NVRTC (PTX) → cuModuleLoad → cuLaunchKernel

Fields are device-resident: ``pgc.field()`` allocates a device buffer via
``cuMemAlloc``.  Transfers are explicit:

    field.from_numpy(arr)   # host → device (cuMemcpyHtoD)
    arr = field.to_numpy()  # device → host (cuMemcpyDtoH)

No per-dispatch copies — data stays on the GPU between kernel calls.
"""

import ctypes

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64
from pgc.lang.type_inference import infer_param_types, check_dispatch_types

_CUDA_SUPPORTED_DTYPES = {i8, u8, i16, u16, i32, u32, i64, u64, f32, f64}
from pgc.codegen.cuda_gen import generate_cuda_source

from cuda.bindings import driver, nvrtc


_NUMPY_DTYPE = {
    f32: np.float32,
    f64: np.float64,
    i32: np.int32,
    i64: np.int64,
    u32: np.uint32,
    u64: np.uint64,
}


_REDUCE_CUDA_SUM = """
extern "C" __global__ void reduce_sum_f32(float* input, float* output) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = __float_as_uint(output[1]);
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(&output[0], sdata[0]);
}
"""

_REDUCE_CUDA_MIN = """
extern "C" __global__ void reduce_min_f32(float* input, float* output) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = __float_as_uint(output[1]);
    sdata[tid] = (i < n) ? input[i] : 1e38f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        int* addr = (int*)&output[0];
        int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_int(fminf(sdata[0], __int_as_float(assumed))));
        } while (assumed != old);
    }
}
"""

_REDUCE_CUDA_MAX = """
extern "C" __global__ void reduce_max_f32(float* input, float* output) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = __float_as_uint(output[1]);
    sdata[tid] = (i < n) ? input[i] : -1e38f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        int* addr = (int*)&output[0];
        int old = *addr, assumed;
        do {
            assumed = old;
            old = atomicCAS(addr, assumed,
                __float_as_int(fmaxf(sdata[0], __int_as_float(assumed))));
        } while (assumed != old);
    }
}
"""


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


class CUDABuffer(DeviceBuffer):
    """Device-resident buffer backed by a CUDA device pointer.

    Data lives on the GPU.  ``from_numpy`` copies host→device,
    ``to_numpy`` copies device→host.
    """

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        err, self._device_ptr = driver.cuMemAlloc(self._nbytes)
        _check(err)
        # Zero-initialise
        _check(driver.cuMemsetD8(self._device_ptr, 0, self._nbytes))

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
        """Export as ExportedMemory. Lazily copies into exportable memory."""
        if not hasattr(self, '_export_buf'):
            self._export_buf = ExportableCUDABuffer(self._numpy_dtype, self._shape)
            _check(driver.cuMemcpyDtoD(
                self._export_buf._device_ptr, self._device_ptr, self._nbytes))
        return self._export_buf.export_memory()

    def __del__(self):
        if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
            try:
                driver.cuMemFree(self._device_ptr)
            except Exception:
                pass


class ExportableCUDABuffer(DeviceBuffer):
    """Device buffer allocated via CUDA VMM with POSIX FD export capability.

    Uses cuMemCreate/cuMemMap instead of cuMemAlloc so the underlying
    memory can be exported as a file descriptor for cross-API sharing
    (e.g. Vulkan import via VK_KHR_external_memory_fd).
    """

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize

        err, self._cuda_device = driver.cuCtxGetDevice()
        _check(err)

        prop = driver.CUmemAllocationProp()
        prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._cuda_device
        prop.requestedHandleTypes = (
            driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)

        err, granularity = driver.cuMemGetAllocationGranularity(
            prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM)
        _check(err)

        self._alloc_size = ((max(self._nbytes, 1) + granularity - 1)
                            // granularity) * granularity

        err, self._mem_handle = driver.cuMemCreate(self._alloc_size, prop, 0)
        _check(err)

        err, self._device_ptr = driver.cuMemAddressReserve(
            self._alloc_size, granularity, 0, 0)
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

        self._exported_fd = None

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
        """Export as ExportedMemory (fd + size + UUID). FD is cached."""
        from pgc.lang.field import ExportedMemory
        if self._exported_fd is None:
            err, fd = driver.cuMemExportToShareableHandle(
                self._mem_handle,
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0)
            _check(err)
            self._exported_fd = fd
        err, uuid = driver.cuDeviceGetUuid(self._cuda_device)
        _check(err)
        return ExportedMemory(
            backend="cuda",
            size=self._nbytes,
            allocation_size=self._alloc_size,
            handle=self._exported_fd,
            device_uuid=bytes(uuid.bytes),
        )

    def __del__(self):
        try:
            if self._exported_fd is not None:
                import os
                os.close(self._exported_fd)
                self._exported_fd = None
            driver.cuMemUnmap(self._device_ptr, self._alloc_size)
            driver.cuMemAddressFree(self._device_ptr, self._alloc_size)
            driver.cuMemRelease(self._mem_handle)
        except Exception:
            pass


def _compile_ptx(cuda_source: str, func_name: str) -> bytes:
    """Compile CUDA C source to PTX via NVRTC."""
    src = cuda_source.encode("utf-8")
    err, prog = nvrtc.nvrtcCreateProgram(src, f"{func_name}.cu".encode(), 0, None, None)
    _check(err)

    opts = [b"--use_fast_math", b"--extra-device-vectorization"]
    c_opts = (ctypes.c_char_p * len(opts))(*opts)
    compile_result = nvrtc.nvrtcCompileProgram(prog, len(opts), c_opts)
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


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments."""
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


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

        block_dim = 256
        grid_dim = (loop_end + block_dim - 1) // block_dim

        _check(driver.cuLaunchKernel(
            self._func,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0, 0,
            arg_ptrs, 0,
        ))
        _check(driver.cuCtxSynchronize())


class CUDABackend:
    """CUDA GPU backend — device-resident fields, NVRTC compilation."""

    def __init__(self):
        _check(driver.cuInit(0))
        err, self._device = driver.cuDeviceGet(0)
        _check(err)

        # Reuse an existing CUDA context if one is already active (e.g. from
        # a simulation framework like AMReX).  Only create a new context when
        # no current context exists.
        err, ctx = driver.cuCtxGetCurrent()
        if err == driver.CUresult.CUDA_SUCCESS and ctx:
            self._context = ctx
            self._owns_context = False
        else:
            err, self._context = driver.cuCtxCreate(None, 0, self._device)
            _check(err)
            self._owns_context = True

        self._cache: dict[str, CompiledCUDAKernel] = {}

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
        try:
            err, mem_type = driver.cuPointerGetAttribute(
                driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                int(ptr))
            if err != driver.CUresult.CUDA_SUCCESS:
                return "cpu"
            # CU_MEMORYTYPE_HOST=1, CU_MEMORYTYPE_DEVICE=2,
            # CU_MEMORYTYPE_ARRAY=3, CU_MEMORYTYPE_UNIFIED=4
            return {1: "cuda_pinned", 2: "cuda", 4: "cuda_managed"}.get(
                int(mem_type), "cpu")
        except Exception:
            return "cpu"

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing CUDA device pointer without allocating or copying."""
        buf = CUDABuffer.__new__(CUDABuffer)
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        buf._device_ptr = ptr  # integer or CUdeviceptr
        buf._owned = False
        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the CUDA GPU."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        # Detect template arguments and expand them
        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args, _detect_texture_fields,
        )
        from pgc.lang.field import Field, Texture3D
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

        # Type inference and dispatch-time type checking
        infer_param_types(ir_func, effective_args)
        check_dispatch_types(ir_func, effective_args,
                             supported_dtypes=_CUDA_SUPPORTED_DTYPES,
                             backend_name="CUDA")

        # Store texture shapes on params for codegen/dispatch
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                param._texture_shape = arg.shape_3d

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Determine loop range BEFORE packing
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(ir_func, kernel_args)

        # Cache key (include texture shapes for uniqueness)
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tex_sig = tuple(
            getattr(p, '_texture_shape', None) for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tex_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            import copy
            from pgc.lang.ir_pack_scalars import pack_scalars
            from pgc.lang.ir_type_annotate import annotate_types
            from pgc.runtime.cpu import _create_pack_fields
            ir_func_copy = copy.deepcopy(ir_func)
            from pgc.codegen.cuda_gen import _safe_kernel_name
            ir_func_copy.name = _safe_kernel_name(ir_func_copy.name)
            _, pack_info = pack_scalars(ir_func_copy, effective_args)
            annotate_types(ir_func_copy)
            compiled = self._compile_kernel(ir_func_copy)
            pack_fields = _create_pack_fields(pack_info, effective_args, self) if pack_info else None
            self._cache[cache_key] = (compiled, pack_info, pack_fields)

        compiled, pack_info, pack_fields = self._cache[cache_key]

        # Build dispatch args
        if pack_info:
            from pgc.runtime.cpu import _update_pack_fields
            from pgc.lang.ir_pack_scalars import split_args
            _update_pack_fields(pack_fields, pack_info, effective_args)
            kept_args = split_args(effective_args, pack_info)
            kernel_args = [a.field if isinstance(a, Texture3D) else a
                           for a in kept_args]
            kernel_args = list(kernel_args) + pack_fields
        else:
            kernel_args = [a.field if isinstance(a, Texture3D) else a
                           for a in effective_args]

        # Dispatch
        compiled(kernel_args, loop_end)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledCUDAKernel:
        """Compile PGC IR → CUDA C → PTX → CUfunction."""
        cuda_source = generate_cuda_source(ir_func)
        ptx = _compile_ptx(cuda_source, ir_func.name)

        err, module = driver.cuModuleLoadData(ptx)
        _check(err)

        err, func = driver.cuModuleGetFunction(module, ir_func.name.encode())
        _check(err)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape
        return CompiledCUDAKernel(module, func, ir_func.name, param_types,
                                  param_is_field, param_is_texture, texture_shapes)

    def reduce_field(self, field, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        from pgc.lang.types import f32
        if field.dtype is not f32:
            return float(getattr(field.to_numpy(), op)())

        if not hasattr(self, '_reduce_cache'):
            self._reduce_cache = {}
        if op not in self._reduce_cache:
            self._reduce_cache[op] = self._compile_reduce(op)

        func, module = self._reduce_cache[op]
        n = int(np.prod(field.shape))

        # Output: [result, n_as_uint_bits]
        import struct as _struct
        init_vals = {"sum": 0.0, "min": 1e38, "max": -1e38}
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
        if hasattr(self, '_context'):
            try:
                driver.cuCtxDestroy(self._context)
            except Exception:
                pass

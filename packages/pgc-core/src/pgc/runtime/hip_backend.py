"""PGC HIP backend — compiles kernels via hipRTC and dispatches on AMD GPUs.

Pipeline:
    PGC IR → HIP C source → hipRTC (code object) → hipModuleLoad → hipLaunchKernel

Fields are device-resident: ``pgc.field()`` allocates a device buffer via
``hipMalloc``.  Transfers are explicit:

    field.from_numpy(arr)   # host → device (hipMemcpyHtoD)
    arr = field.to_numpy()  # device → host (hipMemcpyDtoH)

No per-dispatch copies — data stays on the GPU between kernel calls.

Requires: hip-python (``pip install hip-python``)
"""

import ctypes

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, i8, u8, i16, u16, i32, u32, i64, u64, f32, f64
from pgc.lang.type_inference import infer_param_types, check_dispatch_types

_HIP_SUPPORTED_DTYPES = {i8, u8, i16, u16, i32, u32, i64, u64, f32, f64}
from pgc.codegen.hip_gen import generate_hip_source

from hip import hip, hiprtc


_REDUCE_HIP_SUM = """
#include <hip/hip_runtime.h>
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

_REDUCE_HIP_MIN = """
#include <hip/hip_runtime.h>
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

_REDUCE_HIP_MAX = """
#include <hip/hip_runtime.h>
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

_NUMPY_DTYPE = {
    f32: np.float32,
    f64: np.float64,
    i32: np.int32,
    i64: np.int64,
    u32: np.uint32,
    u64: np.uint64,
}


def _check_hip(result):
    """Check a HIP runtime result, raise on error (handles tuple returns)."""
    err = result[0] if isinstance(result, tuple) else result
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(f"HIP error: {err}")


def _check_hiprtc(result):
    """Check a hipRTC result, raise on error (handles tuple returns)."""
    err = result[0] if isinstance(result, tuple) else result
    if isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        raise RuntimeError(f"hipRTC error: {err}")


class HIPBuffer(DeviceBuffer):
    """Device-resident buffer backed by a HIP device pointer.

    Data lives on the GPU.  ``from_numpy`` copies host→device,
    ``to_numpy`` copies device→host.
    """

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        err, self._device_ptr = hip.hipMalloc(self._nbytes)
        _check_hip(err)
        # Zero-initialise
        _check_hip(hip.hipMemset(self._device_ptr, 0, self._nbytes))

    @property
    def device_ptr(self):
        return self._device_ptr

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        _check_hip(hip.hipMemcpy(
            self._device_ptr, src, self._nbytes,
            hip.hipMemcpyKind.hipMemcpyHostToDevice,
        ))

    def to_numpy(self) -> np.ndarray:
        out = np.empty(self._shape, dtype=self._numpy_dtype)
        _check_hip(hip.hipMemcpy(
            out, self._device_ptr, self._nbytes,
            hip.hipMemcpyKind.hipMemcpyDeviceToHost,
        ))
        return out

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __del__(self):
        if hasattr(self, '_device_ptr') and getattr(self, '_owned', True):
            try:
                hip.hipFree(self._device_ptr)
            except Exception:
                pass


def _compile_code_object(hip_source: str, func_name: str) -> bytes:
    """Compile HIP C source to a code object via hipRTC."""
    src = hip_source.encode("utf-8")
    err, prog = hiprtc.hiprtcCreateProgram(
        src, f"{func_name}.hip".encode(), 0, [], [],
    )
    _check_hiprtc(err)

    # Compile for the current device architecture
    compile_result = hiprtc.hiprtcCompileProgram(prog, 0, [])
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
    return code


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments."""
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


_HIP_CTYPES_MAP = {f32: ctypes.c_float, i32: ctypes.c_int, i64: ctypes.c_longlong,
                   u32: ctypes.c_uint, u64: ctypes.c_ulonglong}


class CompiledHIPKernel:
    """A compiled HIP kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types, param_is_field,
                 param_is_texture=None, texture_shapes=None):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self._texture_shapes = texture_shapes or {}  # param_index → (W, H, D)
        self._tex_cache: dict[tuple, int] = {}

    def _create_texture_object(self, field, W, H, D):
        """Create a HIP texture object from a field's device buffer.

        Allocates a HIP 3D array, copies the field data into it, then creates
        a texture object with linear filtering and normalized coordinates.
        """
        # Create channel format descriptor: 1 channel, 32-bit float
        channel_desc = hip.hipCreateChannelDesc(
            32, 0, 0, 0, hip.hipChannelFormatKind.hipChannelFormatKindFloat)

        # Create extent for the 3D array
        extent = hip.make_hipExtent(W, H, D)

        # Allocate 3D array
        err, hip_array = hip.hipMalloc3DArray(channel_desc, extent, 0)
        _check_hip(err)

        # Copy device linear buffer → HIP 3D array
        copy_params = hip.hipMemcpy3DParms()
        # Source: device pointer as pitched pointer
        copy_params.srcPtr = hip.make_hipPitchedPtr(
            field._buffer.device_ptr, W * 4, W, H)
        copy_params.srcPos = hip.make_hipPos(0, 0, 0)
        # Destination: 3D array
        copy_params.dstArray = hip_array
        copy_params.dstPos = hip.make_hipPos(0, 0, 0)
        copy_params.extent = extent
        copy_params.kind = hip.hipMemcpyKind.hipMemcpyDeviceToDevice
        _check_hip(hip.hipMemcpy3D(copy_params))

        # Create resource descriptor
        res_desc = hip.hipResourceDesc()
        res_desc.resType = hip.hipResourceType.hipResourceTypeArray
        res_desc.res.array.array = hip_array

        # Create texture descriptor
        tex_desc = hip.hipTextureDesc()
        tex_desc.addressMode = (
            hip.hipTextureAddressMode.hipAddressModeClamp,
            hip.hipTextureAddressMode.hipAddressModeClamp,
            hip.hipTextureAddressMode.hipAddressModeClamp,
        )
        tex_desc.filterMode = hip.hipTextureFilterMode.hipFilterModeLinear
        tex_desc.normalizedCoords = 1
        tex_desc.readMode = hip.hipTextureReadMode.hipReadModeElementType

        # Create texture object
        err, tex_obj = hip.hipCreateTextureObject(res_desc, tex_desc, None)
        _check_hip(err)

        return tex_obj, hip_array

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the HIP kernel."""
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for i, (arg, ptype, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field,
                    self._param_is_texture)):
            if is_tex:
                W, H, D = self._texture_shapes[i]
                cache_key = (int(arg._buffer.device_ptr), W, H, D)
                if cache_key not in self._tex_cache:
                    tex_obj, hip_array = self._create_texture_object(arg, W, H, D)
                    self._tex_cache[cache_key] = (tex_obj, hip_array)
                tex_obj, _ = self._tex_cache[cache_key]
                # hipTextureObject_t is unsigned long long (64-bit handle)
                arg_values.append(ctypes.c_ulonglong(tex_obj))
            elif is_field:
                arg_values.append(ctypes.c_void_p(int(arg._buffer.device_ptr)))
            else:
                ct = _HIP_CTYPES_MAP[ptype]
                arg_values.append(ct(arg))
        arg_values.append(n_val)

        arg_ptrs = (ctypes.c_void_p * len(arg_values))()
        for i, val in enumerate(arg_values):
            arg_ptrs[i] = ctypes.addressof(val)

        block_dim = 256
        grid_dim = (loop_end + block_dim - 1) // block_dim

        _check_hip(hip.hipModuleLaunchKernel(
            self._func,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0, None,
            arg_ptrs, None,
        ))
        _check_hip(hip.hipDeviceSynchronize())


class HIPBackend:
    """HIP GPU backend — device-resident fields, hipRTC compilation."""

    def __init__(self):
        _check_hip(hip.hipInit(0))
        err, device = hip.hipGetDevice()
        _check_hip(err)
        self._device = device

        self._cache: dict[str, CompiledHIPKernel] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> HIPBuffer:
        return HIPBuffer(dtype.numpy_dtype, shape)

    def memory_space(self, ptr) -> str:
        """Query where a pointer resides: 'cpu', 'hip', or 'hip_managed'.

        Uses hipPointerGetAttributes to classify the pointer.

        Returns:
            'hip'         — device memory (hipMalloc)
            'hip_pinned'  — pinned host memory (hipHostMalloc)
            'hip_managed' — unified memory (hipMallocManaged)
            'cpu'         — unregistered host memory
        """
        try:
            err, attrs = hip.hipPointerGetAttributes(hip.hipDeviceptr_t(int(ptr)))
            if err != hip.hipSuccess:
                return "cpu"
            mem_type = attrs.type if hasattr(attrs, 'type') else attrs.memoryType
            # hipMemoryTypeHost=1, hipMemoryTypeDevice=2, hipMemoryTypeUnified=3
            return {1: "hip_pinned", 2: "hip", 3: "hip_managed"}.get(
                int(mem_type), "cpu")
        except Exception:
            return "cpu"

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing HIP device pointer without allocating or copying."""
        buf = HIPBuffer.__new__(HIPBuffer)
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        buf._device_ptr = ptr
        buf._owned = False
        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the HIP GPU."""
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
                             supported_dtypes=_HIP_SUPPORTED_DTYPES,
                             backend_name="HIP")

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

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledHIPKernel:
        """Compile PGC IR → HIP C → code object → hipFunction."""
        hip_source = generate_hip_source(ir_func)
        code = _compile_code_object(hip_source, ir_func.name)

        err, module = hip.hipModuleLoadData(code)
        _check_hip(err)

        err, func = hip.hipModuleGetFunction(module, ir_func.name.encode())
        _check_hip(err)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape
        return CompiledHIPKernel(module, func, ir_func.name, param_types,
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
        out_buf = HIPBuffer(np.float32, (2,))
        out_buf.from_numpy(out_np)

        block_dim = 256
        grid_dim = (n + block_dim - 1) // block_dim

        # Dispatch
        in_ptr = ctypes.c_void_p(int(field._buffer.device_ptr))
        out_ptr = ctypes.c_void_p(int(out_buf.device_ptr))
        args = (ctypes.c_void_p * 2)()
        args[0] = ctypes.addressof(in_ptr)
        args[1] = ctypes.addressof(out_ptr)

        _check_hip(hip.hipModuleLaunchKernel(
            func, grid_dim, 1, 1, block_dim, 1, 1,
            block_dim * 4, None, args, 0))
        _check_hip(hip.hipDeviceSynchronize())

        result = out_buf.to_numpy()
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

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
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
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

    def __del__(self):
        if hasattr(self, '_device_ptr'):
            try:
                driver.cuMemFree(self._device_ptr)
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

    def __init__(self, module, func, func_name, param_types, param_is_field):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the CUDA kernel."""
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for arg, ptype, is_field in zip(kernel_args, self._param_types, self._param_is_field):
            if is_field:
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
        err, self._context = driver.cuCtxCreate(None, 0, self._device)
        _check(err)

        self._cache: dict[str, CompiledCUDAKernel] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> CUDABuffer:
        return CUDABuffer(dtype.numpy_dtype, shape)

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

        # Type inference
        infer_param_types(ir_func, effective_args)

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Determine loop range BEFORE packing
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(ir_func, kernel_args)

        # Cache key
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            import copy
            from pgc.lang.ir_pack_scalars import pack_scalars
            ir_func_copy = copy.deepcopy(ir_func)
            _, pack_info = pack_scalars(ir_func_copy, effective_args)
            compiled = self._compile_kernel(ir_func_copy)
            self._cache[cache_key] = (compiled, pack_info)

        compiled, pack_info = self._cache[cache_key]

        # Build dispatch args
        if pack_info:
            from pgc.runtime.cpu import _create_pack_fields
            from pgc.lang.ir_pack_scalars import split_args
            kept_args = split_args(effective_args, pack_info)
            pack_fields = _create_pack_fields(pack_info, effective_args, self)
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
        return CompiledCUDAKernel(module, func, ir_func.name, param_types, param_is_field)

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

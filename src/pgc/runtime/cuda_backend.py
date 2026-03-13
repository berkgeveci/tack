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

    compile_result = nvrtc.nvrtcCompileProgram(prog, 0, None)
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


class CompiledCUDAKernel:
    """A compiled CUDA kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types

    def __call__(self, device_ptrs: list, loop_end: int):
        """Dispatch the CUDA kernel."""
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for dptr in device_ptrs:
            arg_values.append(ctypes.c_void_p(int(dptr)))
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

        # Detect vector fields and get appropriate IR
        from pgc.runtime.cpu import _detect_vector_fields
        from pgc.lang.field import Field
        vector_fields = _detect_vector_fields(kernel, args)
        ir_module = kernel.get_ir(vector_fields)
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes
        name_to_field = {}
        for param, arg in zip(ir_func.params, args):
            if isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference
        infer_param_types(ir_func, args)

        # Cache key
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}"

        if cache_key not in self._cache:
            self._cache[cache_key] = self._compile_kernel(ir_func)

        compiled = self._cache[cache_key]

        # Get device pointers directly from fields (already on device)
        device_ptrs = []
        for arg in args:
            if isinstance(arg, Field):
                device_ptrs.append(arg._buffer.device_ptr)
            else:
                raise NotImplementedError(
                    "Scalar kernel arguments not yet supported in CUDA mode."
                )

        # Determine loop range
        loop_end = _get_loop_range(ir_func, args)

        # Dispatch — no copies, data is already on device
        compiled(device_ptrs, loop_end)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledCUDAKernel:
        """Compile PGC IR → CUDA C → PTX → CUfunction."""
        cuda_source = generate_cuda_source(ir_func)
        ptx = _compile_ptx(cuda_source, ir_func.name)

        err, module = driver.cuModuleLoadData(ptx)
        _check(err)

        err, func = driver.cuModuleGetFunction(module, ir_func.name.encode())
        _check(err)

        param_types = [p.type_annotation for p in ir_func.params]
        return CompiledCUDAKernel(module, func, ir_func.name, param_types)

    def __del__(self):
        if hasattr(self, '_context'):
            try:
                driver.cuCtxDestroy(self._context)
            except Exception:
                pass

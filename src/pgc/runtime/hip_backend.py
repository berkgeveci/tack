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
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.hip_gen import generate_hip_source

from hip import hip, hiprtc


_NUMPY_DTYPE = {
    f32: np.float32,
    f64: np.float64,
    i32: np.int32,
    i64: np.int64,
    u32: np.uint32,
    u64: np.uint64,
}


def _check_hip(err):
    """Check a HIP runtime result, raise on error."""
    if isinstance(err, hip.hipError_t):
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"HIP error: {err}")


def _check_hiprtc(err):
    """Check a hipRTC result, raise on error."""
    if isinstance(err, hiprtc.hiprtcResult):
        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            raise RuntimeError(f"hipRTC error: {err}")


def _check(result):
    """Check a HIP or hipRTC result (handles tuple returns)."""
    if isinstance(result, tuple):
        err = result[0]
    else:
        err = result
    if isinstance(err, hip.hipError_t):
        _check_hip(err)
    elif isinstance(err, hiprtc.hiprtcResult):
        _check_hiprtc(err)


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
        if hasattr(self, '_device_ptr'):
            try:
                hip.hipFree(self._device_ptr)
            except Exception:
                pass


def _compile_code_object(hip_source: str, func_name: str) -> bytes:
    """Compile HIP C source to a code object via hipRTC."""
    src = hip_source.encode("utf-8")
    err, prog = hiprtc.hiprtcCreateProgram(
        src, f"{func_name}.hip".encode(), 0, None, None,
    )
    _check_hiprtc(err)

    # Compile for the current device architecture
    compile_result = hiprtc.hiprtcCompileProgram(prog, 0, None)
    compile_err = compile_result[0] if isinstance(compile_result, tuple) else compile_result

    if compile_err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
        err, log_size = hiprtc.hiprtcGetProgramLogSize(prog)
        log = b" " * log_size
        hiprtc.hiprtcGetProgramLog(prog, log)
        hiprtc.hiprtcDestroyProgram(prog)
        raise RuntimeError(
            f"hipRTC compilation failed:\n{log.decode(errors='replace')}\n"
            f"Source:\n{hip_source}"
        )

    err, code_size = hiprtc.hiprtcGetCodeSize(prog)
    _check_hiprtc(err)
    code = b" " * code_size
    _check_hiprtc(hiprtc.hiprtcGetCode(prog, code))
    hiprtc.hiprtcDestroyProgram(prog)
    return code


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments."""
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


class CompiledHIPKernel:
    """A compiled HIP kernel ready for dispatch."""

    def __init__(self, module, func, func_name, param_types):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types

    def __call__(self, device_ptrs: list, loop_end: int):
        """Dispatch the HIP kernel."""
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

        _check_hip(hip.hipModuleLaunchKernel(
            self._func,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0, None,
            arg_ptrs,
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

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> HIPBuffer:
        return HIPBuffer(dtype.numpy_dtype, shape)

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the HIP GPU."""
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

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

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
                    "Scalar kernel arguments not yet supported in HIP mode."
                )

        # Determine loop range
        loop_end = _get_loop_range(ir_func, args)

        # Dispatch — no copies, data is already on device
        compiled(device_ptrs, loop_end)

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledHIPKernel:
        """Compile PGC IR → HIP C → code object → hipFunction."""
        hip_source = generate_hip_source(ir_func)
        code = _compile_code_object(hip_source, ir_func.name)

        err, module = hip.hipModuleLoadData(code)
        _check_hip(err)

        err, func = hip.hipModuleGetFunction(module, ir_func.name)
        _check_hip(err)

        param_types = [p.type_annotation for p in ir_func.params]
        return CompiledHIPKernel(module, func, ir_func.name, param_types)

    def __del__(self):
        pass  # HIP context is managed by the runtime

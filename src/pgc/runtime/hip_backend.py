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
        if hasattr(self, '_device_ptr'):
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

    def __init__(self, module, func, func_name, param_types, param_is_field):
        self._module = module
        self._func = func
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the HIP kernel."""
        n_val = ctypes.c_longlong(loop_end)

        arg_values = []
        for arg, ptype, is_field in zip(kernel_args, self._param_types, self._param_is_field):
            if is_field:
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

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> HIPBuffer:
        return HIPBuffer(dtype.numpy_dtype, shape)

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
        return CompiledHIPKernel(module, func, ir_func.name, param_types, param_is_field)

    def __del__(self):
        pass  # HIP context is managed by the runtime

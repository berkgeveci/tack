"""Tack HIP backend — compiles kernels via hipRTC and dispatches on AMD GPUs.

Pipeline:
    Tack IR → HIP C source → hipRTC (code object) → hipModuleLoad → hipLaunchKernel

Fields are device-resident: ``tack.field()`` allocates a device buffer via
``hipMalloc``.  Transfers are explicit:

    field.from_numpy(arr)   # host → device (hipMemcpyHtoD)
    arr = field.to_numpy()  # device → host (hipMemcpyDtoH)

No per-dispatch copies — data stays on the GPU between kernel calls.

Requires: hip-python (``pip install 'tack-core[hip]'``)
"""

import ctypes

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

_HIP_SUPPORTED_DTYPES = frozenset({i8, u8, i16, u16, i32, u32, i64, u64, f32, f64})
from hip import hip, hiprtc

from tack.codegen.hip_gen import generate_hip_source
from tack.codegen.identifiers import kernel_entry_name

_REDUCE_HIP_SUM = field_reduction_source('hip', 'sum')
_REDUCE_HIP_MIN = field_reduction_source('hip', 'min')
_REDUCE_HIP_MAX = field_reduction_source('hip', 'max')

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

    backend_name = "hip"

    def __init__(self, numpy_dtype, shape):
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = int(np.prod(shape)) * self._numpy_dtype.itemsize
        err, self._device_ptr = hip.hipMalloc(self._nbytes)
        _check_hip(err)
        # Zero-initialise
        _check_hip(hip.hipMemset(self._device_ptr, 0, self._nbytes))

    @property
    def address(self) -> int:
        return int(self._device_ptr)

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

        block_dim = WORKGROUP_SIZE
        grid_dim = (loop_end + block_dim - 1) // block_dim

        _check_hip(hip.hipModuleLaunchKernel(
            self._func,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0, None,
            arg_ptrs, None,
        ))
        _check_hip(hip.hipDeviceSynchronize())


class HIPBackend(Backend):
    """HIP GPU backend — device-resident fields, hipRTC compilation."""

    name = "hip"
    display_name = "HIP"
    supported_dtypes = _HIP_SUPPORTED_DTYPES
    supports_device_reductions = True
    supports_workgroups = True
    device_memory_spaces = frozenset({"hip", "hip_pinned", "hip_managed"})


    def __init__(self):
        _check_hip(hip.hipInit(0))
        err, device = hip.hipGetDevice()
        _check_hip(err)
        self._device = device

        # Whether this device has texture/image hardware. CDNA parts
        # (gfx940/941/942 — MI300 and friends) have none: hipRTC marks
        # tex3D "unavailable: The image/texture API not supported on the
        # device" and refuses to compile. The runtime answers honestly via
        # hipDeviceAttributeImageSupport, so ask rather than assume, and
        # sample in software where the answer is no.
        self._has_image_support = self._query_image_support()
        self._max_image_3d = (
            self._query_max_image_3d() if self._has_image_support else 0)

        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledHIPKernel}

    def _query_image_support(self) -> bool:
        """True when the device exposes the texture/image API."""
        err, val = hip.hipDeviceGetAttribute(
            hip.hipDeviceAttribute_t.hipDeviceAttributeImageSupport,
            self._device)
        _check_hip(err)
        return bool(val)

    def _query_max_image_3d(self) -> int:
        """Smallest of the three max 3D texture extents, 0 if unreported.

        Only consulted on devices that claim image support; a device that
        reports a non-positive extent is treated as unbounded rather than
        as forbidding every texture, since the extent is advisory and the
        support flag above is the real gate.
        """
        dims = []
        for name in ("hipDeviceAttributeMaxTexture3DWidth",
                     "hipDeviceAttributeMaxTexture3DHeight",
                     "hipDeviceAttributeMaxTexture3DDepth"):
            err, val = hip.hipDeviceGetAttribute(
                getattr(hip.hipDeviceAttribute_t, name), self._device)
            _check_hip(err)
            dims.append(val)
        return min(dims) if all(d > 0 for d in dims) else 0

    def _store_texture_shapes(self, ir_func, effective_args):
        """Record Texture3D extents, falling back to software sampling.

        Mirrors the Level Zero backend: devices with no texture hardware,
        or textures past the device's 3D image limit, are sampled in
        software instead. That changes the generated code, so it is decided
        here — before the variant key is built — rather than at codegen time.
        """
        from tack.lang.field import Texture3D
        max_dim = self._max_image_3d
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                W, H, D = arg.shape_3d
                if self._has_image_support and (
                        max_dim == 0
                        or (W <= max_dim and H <= max_dim and D <= max_dim)):
                    param._texture_shape = arg.shape_3d
                else:
                    param._is_texture = False  # software fallback

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

        This answered 'cpu' for every pointer until 2026-08-11, on the first
        machine that could run it. Two mistakes, both swallowed by a broad
        `except` that returned 'cpu': `hipPointerGetAttributes` takes the
        attribute struct as an out-parameter rather than returning it, and
        `hip.hipSuccess` does not exist (it is `hip.hipError_t.hipSuccess`).
        Since `field_from_ptr` validates against this, *every* attempt to
        wrap a HIP device pointer was rejected as host memory — which is the
        DLPack import path and the VTK device interop, both of them.
        """
        addr = as_address(ptr)
        if addr is None:
            # Not an address at all — Metal hands MTLBuffer objects around,
            # and callers pass whatever they have. The range half of that
            # question matters here too: `int()` accepts 1 << 200 happily
            # and the binding then refuses it from inside its own
            # marshalling, which used to be caught and reported as "cpu".
            return "cpu"

        # `attributes` is an out-parameter: hip-python allocates nothing for
        # you, and the result tuple carries only the error code. Getting this
        # wrong is how this function came to answer "cpu" for every pointer
        # it was ever given -- see the note below.
        attrs = hip.hipPointerAttribute_t()
        res = hip.hipPointerGetAttributes(attrs, hip.hipDeviceptr_t(addr))
        err = res[0] if isinstance(res, tuple) else res
        if int(err) != int(hip.hipError_t.hipSuccess):
            # Documented outcome for a pointer HIP does not recognise, which
            # is what an ordinary host allocation is.
            return "cpu"

        # hipMemoryTypeUnregistered=0 (plain host memory), Host=1, Device=2,
        # Managed=3, Array=10, Unified=11. Anything not device-addressable
        # falls through to "cpu".
        return {1: "hip_pinned", 2: "hip", 3: "hip_managed",
                11: "hip_managed"}.get(int(attrs.type), "cpu")

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
        """Execute a kernel on the HIP GPU.

        The IR passes and device compilation run only when this argument
        shape/type combination is new; see `resolve_variant`.
        """
        from tack.lang.field import Texture3D

        variant, effective_args = resolve_variant(
            self, kernel, args, kwargs,
            build=self._build_variant,
            store_texture_shapes=self._store_texture_shapes,
        )
        compiled, pack_info, pack_fields = variant.payload

        # Loop range comes from the pre-packing IR: packing rewrites the
        # parameter list, and the range expression names the original args.
        kernel_args = [a.field if isinstance(a, Texture3D) else a
                       for a in effective_args]
        loop_end = _get_loop_range(variant.ir, kernel_args)
        if loop_end <= 0:
            # range(0) runs nothing; do not ask the driver for an empty grid.
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

    def _compile_kernel(self, ir_func: ir.IRFunction) -> CompiledHIPKernel:
        """Compile Tack IR → HIP C → code object → hipFunction."""
        kernel_name = kernel_entry_name(ir_func.name)
        hip_source = generate_hip_source(ir_func)
        code = _compile_code_object(hip_source, kernel_name)

        err, module = hip.hipModuleLoadData(code)
        _check_hip(err)

        err, func = hip.hipModuleGetFunction(module, kernel_name.encode())
        _check_hip(err)

        param_types = [p.type_annotation for p in ir_func.params]
        param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
        param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
        texture_shapes = {}
        for i, p in enumerate(ir_func.params):
            if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
                texture_shapes[i] = p._texture_shape
        return CompiledHIPKernel(module, func, kernel_name, param_types,
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

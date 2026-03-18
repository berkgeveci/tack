"""PGC Dawn/WebGPU backend — compute via pydawn (Google's Dawn engine).

Cross-platform GPU compute using Dawn's WebGPU implementation
(Metal on macOS, Vulkan on Linux, D3D12 on Windows).
Enables interop with VTK's Dawn-based WebGPU renderer.

    PGC IR → WGSL source → Dawn shader module → compute pipeline → dispatch
"""

import ctypes
import numpy as np

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.lang.field import Field, DeviceBuffer

try:
    from pydawn import utils as dawn, webgpu
except ImportError:
    dawn = None
    webgpu = None

_USAGE_STORAGE = None
_USAGE_COPY_SRC = None
_USAGE_COPY_DST = None


def _init_usage():
    global _USAGE_STORAGE, _USAGE_COPY_SRC, _USAGE_COPY_DST
    _USAGE_STORAGE = webgpu.WGPUBufferUsage_Storage
    _USAGE_COPY_SRC = webgpu.WGPUBufferUsage_CopySrc
    _USAGE_COPY_DST = webgpu.WGPUBufferUsage_CopyDst


class DawnBuffer(DeviceBuffer):
    """Device-resident buffer backed by a Dawn WebGPU buffer."""

    def __init__(self, device, numpy_dtype, shape):
        self._device = device
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = max(int(np.prod(shape)) * self._numpy_dtype.itemsize, 4)
        self._buffer = dawn.create_buffer(
            device, self._nbytes,
            _USAGE_STORAGE | _USAGE_COPY_SRC | _USAGE_COPY_DST)

    @property
    def gpu_buffer(self):
        return self._buffer

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        dawn.write_buffer(self._device, self._buffer, 0, src.tobytes())

    def to_numpy(self) -> np.ndarray:
        data = dawn.read_buffer(self._device, self._buffer)
        return np.frombuffer(data, dtype=self._numpy_dtype).reshape(self._shape).copy()

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes


class CompiledDawnKernel:
    """A compiled Dawn compute pipeline ready for dispatch."""

    def __init__(self, device, pipeline, bind_group_layout,
                 param_is_field, param_is_texture, texture_shapes,
                 num_bindings):
        self._device = device
        self._pipeline = pipeline
        self._bind_group_layout = bind_group_layout
        self._param_is_field = param_is_field
        self._param_is_texture = param_is_texture
        self._texture_shapes = texture_shapes
        self._num_bindings = num_bindings

    def __call__(self, kernel_args, loop_end):
        device = self._device

        # Build bind group entries
        entries = []
        binding_idx = 0
        for i, (arg, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_is_field, self._param_is_texture)):
            if is_tex:
                # TODO: hardware texture support for Dawn
                # For now, textures are passed as regular storage buffers
                # (software trilinear via WGSL helper)
                entries.append({
                    "binding": binding_idx,
                    "resource": {"buffer": arg._buffer.gpu_buffer, "offset": 0,
                                 "size": arg._buffer.nbytes},
                })
            elif is_field:
                entries.append({
                    "binding": binding_idx,
                    "resource": {"buffer": arg._buffer.gpu_buffer, "offset": 0,
                                 "size": arg._buffer.nbytes},
                })
            else:
                val_np = np.array([arg], dtype=np.float32)
                buf = dawn.create_buffer(device, 4,
                    _USAGE_STORAGE | _USAGE_COPY_DST)
                dawn.write_buffer(device, buf, 0, val_np.tobytes())
                entries.append({
                    "binding": binding_idx,
                    "resource": {"buffer": buf, "offset": 0, "size": 4},
                })
            binding_idx += 1

        # Loop-end parameter buffer (pgc_params)
        params_np = np.array([loop_end], dtype=np.uint32)
        params_buf = dawn.create_buffer(device, 4,
            _USAGE_STORAGE | _USAGE_COPY_DST)
        dawn.write_buffer(device, params_buf, 0, params_np.tobytes())
        entries.append({
            "binding": binding_idx,
            "resource": {"buffer": params_buf, "offset": 0, "size": 4},
        })

        bind_group = dawn.create_bind_group(
            device, self._bind_group_layout, entries)

        # Dispatch with 2D grid if needed (WebGPU max 65535 per dimension)
        num_groups = (loop_end + 255) // 256
        max_dim = 65535

        encoder = dawn.create_command_encoder(device)
        pass_enc = dawn.begin_compute_pass(encoder)
        dawn.set_pipeline(pass_enc, self._pipeline)
        dawn.set_bind_group(pass_enc, bind_group)
        if num_groups <= max_dim:
            dawn.dispatch_workgroups(pass_enc, num_groups, 1, 1)
        else:
            gx = max_dim
            gy = (num_groups + max_dim - 1) // max_dim
            dawn.dispatch_workgroups(pass_enc, gx, gy, 1)
        dawn.end_compute_pass(pass_enc)
        cmd = dawn.command_encoder_finish(encoder)
        dawn.submit(device, [cmd])

        # Wait for GPU to finish
        dawn.sync(device)


def _compile_kernel(device, ir_func: ir.IRFunction) -> CompiledDawnKernel:
    """Compile PGC IR → WGSL → Dawn compute pipeline."""
    from pgc.codegen.wgsl_gen import generate_wgsl_source

    wgsl_source = generate_wgsl_source(ir_func)

    import os
    if os.environ.get("PGC_DUMP_WGSL"):
        path = f"/tmp/pgc_{ir_func.name}.wgsl"
        with open(path, "w") as f:
            f.write(wgsl_source)
        print(f"[PGC] Dumped WGSL to {path}")

    shader = dawn.create_shader_module(device, wgsl_source)

    # Build bind group layout
    param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
    param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
    texture_shapes = {}
    for i, p in enumerate(ir_func.params):
        if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
            texture_shapes[i] = p._texture_shape

    layout_entries = []
    binding_idx = 0
    for i in range(len(ir_func.params)):
        layout_entries.append({
            "binding": binding_idx,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},
        })
        binding_idx += 1

    # pgc_params binding (loop end)
    num_bindings = binding_idx
    layout_entries.append({
        "binding": binding_idx,
        "visibility": webgpu.WGPUShaderStage_Compute,
        "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
    })

    bind_group_layout = dawn.create_bind_group_layout(device, layout_entries)
    pipeline_layout = dawn.create_pipeline_layout(device, [bind_group_layout])
    pipeline = dawn.create_compute_pipeline(
        device, pipeline_layout,
        {"module": shader, "entry_point": ir_func.name})

    return CompiledDawnKernel(device, pipeline, bind_group_layout,
                              param_is_field, param_is_texture, texture_shapes,
                              num_bindings)


class DawnBackend:
    """Dawn/WebGPU backend — cross-platform GPU compute via pydawn."""

    def __init__(self):
        if dawn is None:
            raise ImportError(
                "Dawn backend requires dawn-python. "
                "Install with: pip install dawn-python")

        _init_usage()
        adapter = dawn.request_adapter_sync(
            power_preference=webgpu.WGPUPowerPreference_HighPerformance)
        if adapter is None:
            raise RuntimeError("No Dawn WebGPU adapter found")
        self._device = dawn.request_device_sync(adapter)
        self._cache: dict[str, tuple] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> DawnBuffer:
        return DawnBuffer(self._device, dtype.numpy_dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing Dawn GPUBuffer."""
        buf = DawnBuffer.__new__(DawnBuffer)
        buf._device = self._device
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        buf._buffer = ptr
        return buf

    def execute(self, kernel, args, kwargs):
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported")

        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args, _detect_texture_fields,
            _get_loop_range,
        )
        from pgc.lang.field import Texture3D

        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)
        texture_fields = _detect_texture_fields(kernel, args, template_args)

        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
            texture_fields=texture_fields,
        )
        ir_func = ir_module.functions[0]

        # Resolve and type inference
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        infer_param_types(ir_func, effective_args)

        # Textures use software trilinear on Dawn (no hardware texture layout
        # support in pydawn utils yet). Clear _is_texture so codegen uses
        # the software fallback path.
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                param._is_texture = False

        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Loop range before packing
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
            from pgc.lang.ir_type_annotate import annotate_types
            from pgc.runtime.cpu import _create_pack_fields
            ir_func_copy = copy.deepcopy(ir_func)
            _, pack_info = pack_scalars(ir_func_copy, effective_args)
            annotate_types(ir_func_copy)
            compiled = _compile_kernel(self._device, ir_func_copy)
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

        compiled(kernel_args, loop_end)

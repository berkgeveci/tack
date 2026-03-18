"""PGC WebGPU backend — compute via wgpu-py.

Cross-platform GPU compute using WebGPU (Metal on macOS, Vulkan on Linux,
D3D12 on Windows). Requires the wgpu-py package.

    PGC IR → WGSL source → wgpu shader module → compute pipeline → dispatch
"""

import ctypes
import numpy as np

from pgc.lang import ir
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.lang.field import Field, DeviceBuffer

try:
    import wgpu
except ImportError:
    wgpu = None


class WGPUBuffer(DeviceBuffer):
    """Device-resident buffer backed by a WebGPU storage buffer."""

    def __init__(self, device, numpy_dtype, shape):
        self._device = device
        self._numpy_dtype = np.dtype(numpy_dtype)
        self._shape = shape
        self._nbytes = max(int(np.prod(shape)) * self._numpy_dtype.itemsize, 4)
        self._buffer = device.create_buffer(
            size=self._nbytes,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC |
                   wgpu.BufferUsage.COPY_DST))

    @property
    def gpu_buffer(self):
        return self._buffer

    def from_numpy(self, arr: np.ndarray):
        src = np.ascontiguousarray(arr, dtype=self._numpy_dtype)
        self._device.queue.write_buffer(self._buffer, 0, src.tobytes())

    def to_numpy(self) -> np.ndarray:
        data = self._device.queue.read_buffer(self._buffer)
        return np.frombuffer(data.cast("B"), dtype=self._numpy_dtype).reshape(self._shape).copy()

    def fill(self, value):
        arr = np.full(self._shape, value, dtype=self._numpy_dtype)
        self.from_numpy(arr)

    @property
    def nbytes(self) -> int:
        return self._nbytes


class CompiledWGPUKernel:
    """A compiled WebGPU compute pipeline ready for dispatch."""

    def __init__(self, device, pipeline, bind_group_layout,
                 param_is_field, num_bindings):
        self._device = device
        self._pipeline = pipeline
        self._bind_group_layout = bind_group_layout
        self._param_is_field = param_is_field
        self._num_bindings = num_bindings

    def __call__(self, kernel_args, loop_end):
        device = self._device

        # Build bind group entries
        entries = []
        for i, (arg, is_field) in enumerate(zip(kernel_args, self._param_is_field)):
            if is_field:
                entries.append({
                    "binding": i,
                    "resource": {"buffer": arg._buffer.gpu_buffer},
                })
            else:
                # Scalar — shouldn't happen after packing, but handle it
                val_np = np.array([arg], dtype=np.float32)
                buf = device.create_buffer_with_data(
                    data=val_np.tobytes(),
                    usage=wgpu.BufferUsage.STORAGE)
                entries.append({
                    "binding": i,
                    "resource": {"buffer": buf},
                })

        # Loop-end parameter buffer (__params__)
        params_np = np.array([loop_end], dtype=np.uint32)
        params_buf = device.create_buffer_with_data(
            data=params_np.tobytes(),
            usage=wgpu.BufferUsage.STORAGE)
        entries.append({
            "binding": self._num_bindings,
            "resource": {"buffer": params_buf},
        })

        bind_group = device.create_bind_group(
            layout=self._bind_group_layout, entries=entries)

        # Dispatch with 2D grid if needed (WebGPU max 65535 per dimension)
        num_groups = (loop_end + 255) // 256
        max_dim = 65535

        encoder = device.create_command_encoder()
        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(self._pipeline)
        pass_enc.set_bind_group(0, bind_group)
        if num_groups <= max_dim:
            pass_enc.dispatch_workgroups(num_groups)
        else:
            gx = max_dim
            gy = (num_groups + max_dim - 1) // max_dim
            pass_enc.dispatch_workgroups(gx, gy)
        pass_enc.end()
        device.queue.submit([encoder.finish()])

        # Wait for GPU to finish
        device.queue.on_submitted_work_done_sync()


def _compile_kernel(device, ir_func: ir.IRFunction) -> CompiledWGPUKernel:
    """Compile PGC IR → WGSL → WebGPU compute pipeline."""
    from pgc.codegen.wgsl_gen import generate_wgsl_source

    wgsl_source = generate_wgsl_source(ir_func)

    import os
    if os.environ.get("PGC_DUMP_WGSL"):
        path = f"/tmp/pgc_{ir_func.name}.wgsl"
        with open(path, "w") as f:
            f.write(wgsl_source)
        print(f"[PGC] Dumped WGSL to {path}")

    shader_module = device.create_shader_module(code=wgsl_source)

    # Build bind group layout: one storage buffer per param + __params__
    param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
    num_bindings = len(ir_func.params)

    layout_entries = []
    for i in range(num_bindings):
        layout_entries.append({
            "binding": i,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {"type": wgpu.BufferBindingType.storage},
        })
    # __params__ binding (loop end)
    layout_entries.append({
        "binding": num_bindings,
        "visibility": wgpu.ShaderStage.COMPUTE,
        "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
    })

    bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout])

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": ir_func.name})

    return CompiledWGPUKernel(device, pipeline, bind_group_layout,
                              param_is_field, num_bindings)


class WebGPUBackend:
    """WebGPU backend — cross-platform GPU compute via wgpu-py."""

    def __init__(self):
        if wgpu is None:
            raise ImportError(
                "WebGPU backend requires wgpu. Install with: pip install wgpu")

        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        if adapter is None:
            raise RuntimeError("No WebGPU adapter found")

        # Request the adapter's maximum limits for compute workloads
        self._device = adapter.request_device_sync(
            required_limits={
                "max-buffer-size": adapter.limits["max-buffer-size"],
                "max-storage-buffer-binding-size": adapter.limits["max-storage-buffer-binding-size"],
                "max-compute-workgroups-per-dimension": adapter.limits["max-compute-workgroups-per-dimension"],
                "max-storage-buffers-per-shader-stage": adapter.limits["max-storage-buffers-per-shader-stage"],
            }
        )
        self._adapter_info = adapter.info
        self._cache: dict[str, tuple] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> WGPUBuffer:
        return WGPUBuffer(self._device, dtype.numpy_dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing wgpu.GPUBuffer."""
        buf = WGPUBuffer.__new__(WGPUBuffer)
        buf._device = self._device
        buf._numpy_dtype = np.dtype(dtype.numpy_dtype)
        buf._shape = shape
        buf._nbytes = int(np.prod(shape)) * buf._numpy_dtype.itemsize
        buf._buffer = ptr  # expects a wgpu.GPUBuffer
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
        from pgc.lang.field import Texture3D
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Texture3D):
                name_to_field[param.name] = arg
            elif isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        infer_param_types(ir_func, effective_args)

        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Loop range before packing — unwrap Texture3D
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

        # Build dispatch args — unwrap Texture3D to underlying Field
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

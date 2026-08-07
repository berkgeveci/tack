"""Tack Metal compute backend — compiles kernels to MSL and dispatches on GPU.

Pipeline:
    Tack IR → MSL (via msl_gen.py) → Metal compute pipeline

Each field parameter becomes a Metal buffer at index matching its binding number.
The parallel loop range is dispatched as a 1D grid of threads.

On Apple Silicon, Metal shared buffers live in unified memory accessible by both
CPU and GPU.  Fields are backed directly by Metal buffer memory — no per-dispatch
copies are needed.
"""

import numpy as np

from tack.lang import ir
from tack.lang.field import DeviceBuffer, ExportedMemory
from tack.lang.types import ScalarType, f32, i8, i16, i32, i64, u8, u16, u32, u64
from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    _get_loop_range,
    new_kernel_cache,
    resolve_variant,
)

_METAL_SUPPORTED_DTYPES = {i8, u8, i16, u16, i32, u32, i64, u64, f32}
from tack.codegen.msl_gen import generate_msl_source

try:
    import Metal  # pyobjc-framework-Metal
except ImportError:
    Metal = None


class MetalBuffer(DeviceBuffer):
    """Metal shared buffer — zero-copy on Apple Silicon unified memory.

    The numpy view points directly into Metal shared buffer memory, so
    from_numpy/to_numpy are just memory copies within CPU-accessible space
    (no DMA transfers).
    """

    def __init__(self, device, numpy_dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(numpy_dtype).itemsize
        # MTLResourceStorageModeShared = 0 (CPU+GPU unified memory)
        self._metal_buffer = device.newBufferWithLength_options_(nbytes, 0)
        raw = self._metal_buffer.contents().as_buffer(nbytes)
        self._view = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)
        self._view[:] = 0

    @property
    def metal_buffer(self):
        return self._metal_buffer

    def from_numpy(self, arr: np.ndarray):
        np.copyto(self._view, arr)

    def to_numpy(self) -> np.ndarray:
        return self._view.copy()

    def fill(self, value):
        self._view.fill(value)

    @property
    def nbytes(self) -> int:
        return self._view.nbytes

    def export_memory(self):
        """Export as ExportedMemory with the MTLBuffer pointer."""
        import objc
        return ExportedMemory(
            backend="metal",
            size=self._view.nbytes,
            allocation_size=self._metal_buffer.length(),
            handle=objc.pyobjc_id(self._metal_buffer),
        )


class CompiledMetalKernel:
    """A compiled Metal compute pipeline ready for dispatch."""

    _NUMPY_MAP = {f32: np.float32, i32: np.int32, i64: np.int64, u32: np.uint32, u64: np.uint64}

    def __init__(self, device, command_queue, pipeline, func_name,
                 param_types, param_is_field, param_is_texture=None,
                 texture_shapes=None):
        self._device = device
        self._command_queue = command_queue
        self._pipeline = pipeline
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self._texture_shapes = texture_shapes or {}  # param_index → (W, H, D)
        self._thread_execution_width = pipeline.threadExecutionWidth()
        self._max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup()

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the compute kernel on the GPU."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoderWithDescriptor_(
            Metal.MTLComputePassDescriptor.computePassDescriptor()
        )
        encoder.setComputePipelineState_(self._pipeline)

        # Bind buffers and textures.
        # Textures use a separate binding namespace (texture indices).
        temp_buffers = []
        buf_idx = 0
        tex_idx = 0
        for i, (arg, ptype, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field,
                    self._param_is_texture)):
            if is_tex:
                # Create MTLTexture and copy buffer data into it
                W, H, D = self._texture_shapes[i]
                cache_key = (id(arg._buffer.metal_buffer), W, H, D)
                if not hasattr(self, '_tex_cache'):
                    self._tex_cache = {}
                if cache_key not in self._tex_cache:
                    desc = Metal.MTLTextureDescriptor.alloc().init()
                    desc.setTextureType_(7)  # MTLTextureType3D
                    desc.setPixelFormat_(55)  # MTLPixelFormatR32Float
                    desc.setWidth_(W)
                    desc.setHeight_(H)
                    desc.setDepth_(D)
                    desc.setUsage_(1)  # MTLTextureUsageShaderRead
                    desc.setStorageMode_(0)  # MTLStorageModeShared
                    tex = self._device.newTextureWithDescriptor_(desc)
                    # Copy from buffer to texture via blit
                    blit_buf = self._command_queue.commandBuffer()
                    blit_enc = blit_buf.blitCommandEncoder()
                    bytes_per_row = W * 4
                    bytes_per_image = W * H * 4
                    blit_enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
                        arg._buffer.metal_buffer, 0, bytes_per_row, bytes_per_image,
                        Metal.MTLSizeMake(W, H, D), tex, 0, 0,
                        Metal.MTLOriginMake(0, 0, 0))
                    blit_enc.endEncoding()
                    blit_buf.commit()
                    blit_buf.waitUntilCompleted()
                    self._tex_cache[cache_key] = tex
                encoder.setTexture_atIndex_(self._tex_cache[cache_key], tex_idx)
                tex_idx += 1
            elif is_field:
                encoder.setBuffer_offset_atIndex_(arg._buffer.metal_buffer, 0, buf_idx)
                buf_idx += 1
            else:
                # Scalar: create a tiny shared buffer with the value
                ndt = self._NUMPY_MAP[ptype]
                arr = np.array([arg], dtype=ndt)
                nbytes = arr.nbytes
                buf = self._device.newBufferWithBytes_length_options_(
                    arr.tobytes(), nbytes, 0)
                encoder.setBuffer_offset_atIndex_(buf, 0, buf_idx)
                temp_buffers.append(buf)  # prevent GC
                buf_idx += 1

        # Dispatch threads
        threads_per_group = min(self._max_threads_per_group, 256)
        grid_size = Metal.MTLSizeMake(loop_end, 1, 1)
        group_size = Metal.MTLSizeMake(threads_per_group, 1, 1)

        encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, group_size)
        encoder.endEncoding()

        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        error = command_buffer.error()
        if error is not None:
            raise RuntimeError(f"Metal compute error: {error}")


def _compile_kernel(device, command_queue, ir_func: ir.IRFunction) -> CompiledMetalKernel:
    """Compile a Tack IR function to a Metal compute pipeline."""
    msl_source = generate_msl_source(ir_func)

    # Debug: dump MSL source for analysis
    import os
    if os.environ.get("TACK_DUMP_MSL"):
        path = f"/tmp/tack_{ir_func.name}.msl"
        with open(path, "w") as f:
            f.write(msl_source)
        print(f"[Tack] Dumped MSL to {path}")

    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(
        msl_source, options, None
    )
    if library is None:
        raise RuntimeError(f"Metal shader compilation failed:\n{error}\n\nMSL source:\n{msl_source}")

    func = library.newFunctionWithName_(ir_func.name)
    if func is None:
        raise RuntimeError(f"Could not find '{ir_func.name}' function in Metal library")

    pipeline, error = device.newComputePipelineStateWithFunction_error_(func, None)
    if pipeline is None:
        raise RuntimeError(f"Metal pipeline creation failed: {error}")

    param_types = [p.type_annotation for p in ir_func.params]
    param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
    param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
    # Collect texture shapes from IRTextureSample nodes in the IR
    texture_shapes = {}
    for i, p in enumerate(ir_func.params):
        if getattr(p, '_is_texture', False) and hasattr(p, '_texture_shape'):
            texture_shapes[i] = p._texture_shape
    return CompiledMetalKernel(device, command_queue, pipeline, ir_func.name,
                               param_types, param_is_field, param_is_texture,
                               texture_shapes)


_REDUCE_MSL_SUM = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum_f32(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];
    uint n = as_type<uint>(output[1]);
    sdata[local_tid] = (tid < n) ? input[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_tid < s) sdata[local_tid] += sdata[local_tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_tid == 0) {
        atomic_fetch_add_explicit(
            (volatile device atomic_float*)&output[0],
            sdata[0], memory_order_relaxed);
    }
}
"""

_REDUCE_MSL_MIN = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_min_f32(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];
    uint n = as_type<uint>(output[1]);
    sdata[local_tid] = (tid < n) ? input[tid] : 1e38f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_tid < s) sdata[local_tid] = min(sdata[local_tid], sdata[local_tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_tid == 0) {
        // CAS loop for atomic min
        volatile device atomic_uint* p = (volatile device atomic_uint*)&output[0];
        uint old_bits = atomic_load_explicit(p, memory_order_relaxed);
        while (true) {
            float old_f = as_type<float>(old_bits);
            if (old_f <= sdata[0]) break;
            uint new_bits = as_type<uint>(sdata[0]);
            if (atomic_compare_exchange_weak_explicit(p, &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) break;
        }
    }
}
"""

_REDUCE_MSL_MAX = """
#include <metal_stdlib>
using namespace metal;

kernel void reduce_max_f32(
    device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint local_tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    threadgroup float sdata[256];
    uint n = as_type<uint>(output[1]);
    sdata[local_tid] = (tid < n) ? input[tid] : -1e38f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_tid < s) sdata[local_tid] = max(sdata[local_tid], sdata[local_tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_tid == 0) {
        volatile device atomic_uint* p = (volatile device atomic_uint*)&output[0];
        uint old_bits = atomic_load_explicit(p, memory_order_relaxed);
        while (true) {
            float old_f = as_type<float>(old_bits);
            if (old_f >= sdata[0]) break;
            uint new_bits = as_type<uint>(sdata[0]);
            if (atomic_compare_exchange_weak_explicit(p, &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) break;
        }
    }
}
"""


class MetalBackend(Backend):
    """Metal GPU backend — zero-copy dispatch on Apple Silicon unified memory.

    Fields are backed by Metal shared buffers allocated at field creation time.
    from_numpy/to_numpy operate on the numpy view into shared memory — no
    host↔device transfers needed.
    """

    name = "metal"
    display_name = "Metal"
    supported_dtypes = _METAL_SUPPORTED_DTYPES   # no f64: Apple GPUs lack it
    supports_device_reductions = True
    # Metal shared buffers live in unified memory, so a pointer into one is
    # CPU-addressable; the inherited memory_space() answer is right.


    def __init__(self):
        if Metal is None:
            raise ImportError(
                "Metal backend requires pyobjc-framework-Metal. "
                "Install with: pip install 'tack[metal]'"
            )

        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal-capable GPU found")

        self._command_queue = self._device.newCommandQueue()
        self._cache = new_kernel_cache()  # Kernel -> {variant_key: CompiledMetalKernel}
        self._reduce_pipelines: dict[str, object] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...],
                        exportable: bool = False) -> MetalBuffer:
        return MetalBuffer(self._device, dtype.numpy_dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing MTLBuffer as a MetalBuffer without copying."""
        buf = MetalBuffer.__new__(MetalBuffer)
        buf._metal_buffer = ptr  # expects an MTLBuffer object
        nbytes = int(np.prod(shape)) * np.dtype(dtype.numpy_dtype).itemsize
        raw = ptr.contents().as_buffer(nbytes)
        buf._view = np.frombuffer(raw, dtype=dtype.numpy_dtype).reshape(shape)
        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Metal GPU with zero-copy dispatch.

        The IR passes and MSL compilation run only when this argument
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

        # Replace scalar args with the packed field buffers
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
        import copy

        from tack.codegen.msl_gen import _safe_kernel_name
        from tack.lang.ir_pack_scalars import pack_scalars
        from tack.lang.ir_type_annotate import annotate_types
        from tack.runtime.kernel_utils import _create_pack_fields

        packed = copy.deepcopy(ir_func)
        packed.name = _safe_kernel_name(packed.name)
        _, pack_info = pack_scalars(packed, effective_args)
        annotate_types(packed)
        compiled = _compile_kernel(self._device, self._command_queue, packed)
        pack_fields = (_create_pack_fields(pack_info, effective_args, self)
                       if pack_info else None)
        return compiled, pack_info, pack_fields

    def _get_reduce_pipeline(self, op: str):
        """Get or compile a Metal reduction pipeline."""
        if op in self._reduce_pipelines:
            return self._reduce_pipelines[op]

        sources = {"sum": _REDUCE_MSL_SUM, "min": _REDUCE_MSL_MIN, "max": _REDUCE_MSL_MAX}
        func_names = {"sum": "reduce_sum_f32", "min": "reduce_min_f32", "max": "reduce_max_f32"}
        msl_source = sources[op]
        func_name = func_names[op]

        options = Metal.MTLCompileOptions.alloc().init()
        library, error = self._device.newLibraryWithSource_options_error_(
            msl_source, options, None)
        if library is None:
            raise RuntimeError(f"Metal reduce kernel compilation failed: {error}")

        func = library.newFunctionWithName_(func_name)
        pipeline, error = self._device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline is None:
            raise RuntimeError(f"Metal reduce pipeline failed: {error}")

        self._reduce_pipelines[op] = pipeline
        return pipeline

    def reduce_field(self, field, op: str) -> float:
        """GPU-side reduction: sum, min, or max."""
        from tack.lang.types import f32
        if field.dtype is not f32:
            # Fall back to numpy for non-f32
            arr = field.to_numpy()
            return float(getattr(arr, op)())

        pipeline = self._get_reduce_pipeline(op)
        n = int(np.prod(field.shape))

        # Create output buffer: [result, n_as_float_bits]
        import struct
        init_vals = {"sum": 0.0, "min": 1e38, "max": -1e38}
        out_data = np.array([init_vals[op], 0.0], dtype=np.float32)
        # Pack n as uint32 bits into float slot
        out_data[1] = np.frombuffer(struct.pack('I', n), dtype=np.float32)[0]
        out_buf = self._device.newBufferWithBytes_length_options_(
            out_data.tobytes(), out_data.nbytes, 0)

        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoderWithDescriptor_(
            Metal.MTLComputePassDescriptor.computePassDescriptor())
        encoder.setComputePipelineState_(pipeline)
        encoder.setBuffer_offset_atIndex_(field._buffer.metal_buffer, 0, 0)
        encoder.setBuffer_offset_atIndex_(out_buf, 0, 1)

        block_dim = 256
        num_groups = (n + block_dim - 1) // block_dim
        grid_size = Metal.MTLSizeMake(num_groups, 1, 1)
        group_size = Metal.MTLSizeMake(block_dim, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(grid_size, group_size)
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

        # Read result
        raw = out_buf.contents().as_buffer(8)
        result = np.frombuffer(raw, dtype=np.float32)[0]
        return float(result)

"""Tack Metal compute backend — compiles kernels to MSL and dispatches on GPU.

Pipeline:
    Tack IR → MSL (via msl_gen.py) → Metal compute pipeline

Field pointers are encoded in one argument buffer to permit overlapping storage.
The parallel loop range is dispatched as a 1D grid of threads.

On Apple Silicon, Metal shared buffers live in unified memory accessible by both
CPU and GPU.  Fields are backed directly by Metal buffer memory — no per-dispatch
copies are needed.
"""

import numpy as np

from tack.codegen.reductions import field_reduction_source
from tack.lang import ir
from tack.lang.field import DeviceBuffer, ExportedMemory
from tack.lang.types import ScalarType, f32, i8, i16, i32, i64, u8, u16, u32, u64
from tack.lang.workgroup_participation import (
    WORKGROUP_SIZE,
    check_workgroup_launch,
    requires_full_workgroups,
)
from tack.runtime.backend import Backend
from tack.runtime.kernel_utils import (
    _get_loop_range,
    bind_textures,
    check_launch_size,
    new_kernel_cache,
    resolve_variant,
)
from tack.runtime.reductions import REDUCTION_IDENTITIES, empty_reduction, reduce_numpy

_METAL_SUPPORTED_DTYPES = frozenset({i8, u8, i16, u16, i32, u32, i64, u64, f32})
from tack.codegen.identifiers import kernel_entry_name
from tack.codegen.msl_gen import generate_msl_source

try:
    import Metal  # pyobjc-framework-Metal
except ImportError:
    Metal = None

# Kernels read their index from a `uint` [[thread_position_in_grid]], so one
# dispatch can index 2^32 threads; past that the position would wrap.
_MAX_LAUNCH = 2**32


class MetalBuffer(DeviceBuffer):
    """Metal shared buffer — zero-copy on Apple Silicon unified memory.

    The numpy view points directly into Metal shared buffer memory, so
    from_numpy/to_numpy are just memory copies within CPU-accessible space
    (no DMA transfers).
    """

    backend_name = "metal"

    def __init__(self, device, numpy_dtype, shape):
        nbytes = int(np.prod(shape)) * np.dtype(numpy_dtype).itemsize
        # MTLResourceStorageModeShared = 0 (CPU+GPU unified memory). A
        # zero-length request returns no buffer, so an empty field (the
        # result of a filter that selected nothing) gets one byte it never
        # reads; the view below still has zero elements.
        self._metal_buffer = device.newBufferWithLength_options_(max(nbytes, 1), 0)
        raw = self._metal_buffer.contents().as_buffer(nbytes)
        self._view = np.frombuffer(raw, dtype=numpy_dtype).reshape(shape)
        self._view[:] = 0

    @property
    def address(self) -> int:
        return self._view.ctypes.data

    @property
    def metal_buffer(self):
        return self._metal_buffer

    def from_numpy(self, arr: np.ndarray):
        # A reshaped field shares this buffer under another shape; the
        # element count already matches, so copy in the buffer's own shape.
        np.copyto(self._view, arr.reshape(self._view.shape))

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
            handle_type="mtl_buffer",
        )


class MetalTextureImage:
    """A shader-read R32Float 3D MTLTexture.

    Owned by one `Texture3D`, which uploads its field's data at creation and
    on ``update()``. Nothing else keeps a reference, so the texture is
    released with it: no later texture can be served one that was made from
    someone else's buffer.
    """

    def __init__(self, device, command_queue, shape_3d):
        W, H, D = shape_3d
        self._command_queue = command_queue
        self._shape = shape_3d
        desc = Metal.MTLTextureDescriptor.alloc().init()
        desc.setTextureType_(7)  # MTLTextureType3D
        desc.setPixelFormat_(55)  # MTLPixelFormatR32Float
        desc.setWidth_(W)
        desc.setHeight_(H)
        desc.setDepth_(D)
        desc.setUsage_(1)  # MTLTextureUsageShaderRead
        desc.setStorageMode_(0)  # MTLStorageModeShared
        self.texture = device.newTextureWithDescriptor_(desc)

    def upload(self, field):
        """Copy a field's buffer into the texture via blit."""
        W, H, D = self._shape
        blit_buf = self._command_queue.commandBuffer()
        blit_enc = blit_buf.blitCommandEncoder()
        bytes_per_row = W * 4
        bytes_per_image = W * H * 4
        blit_enc.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            field._buffer.metal_buffer, 0, bytes_per_row, bytes_per_image,
            Metal.MTLSizeMake(W, H, D), self.texture, 0, 0,
            Metal.MTLOriginMake(0, 0, 0))
        blit_enc.endEncoding()
        blit_buf.commit()
        blit_buf.waitUntilCompleted()


class CompiledMetalKernel:
    """A compiled Metal compute pipeline ready for dispatch."""

    _NUMPY_MAP = {f32: np.float32, i32: np.int32, i64: np.int64, u32: np.uint32, u64: np.uint64}

    def __init__(self, device, command_queue, pipeline, func_name,
                 param_types, param_is_field, param_is_texture=None,
                 argument_encoder=None, *,
                 requires_full_workgroups=False):
        self._max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup()
        self._workgroup_size = min(self._max_threads_per_group, WORKGROUP_SIZE)
        self._requires_full_workgroups = requires_full_workgroups
        if requires_full_workgroups:
            check_workgroup_launch(func_name, 0, backend_label='Metal',
                                   workgroup_size=self._workgroup_size)
        self._device = device
        self._command_queue = command_queue
        self._pipeline = pipeline
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._param_is_texture = param_is_texture or [False] * len(param_types)
        self._argument_encoder = argument_encoder
        self._argument_buffer = None
        if argument_encoder is not None:
            self._argument_buffer = device.newBufferWithLength_options_(
                argument_encoder.encodedLength(), Metal.MTLResourceStorageModeShared)
            argument_encoder.setArgumentBuffer_offset_(self._argument_buffer, 0)
        self._thread_execution_width = pipeline.threadExecutionWidth()

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the compute kernel on the GPU."""
        if self._requires_full_workgroups:
            check_workgroup_launch(self._func_name, loop_end, backend_label='Metal',
                                   workgroup_size=self._workgroup_size)
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoderWithDescriptor_(
            Metal.MTLComputePassDescriptor.computePassDescriptor()
        )
        encoder.setComputePipelineState_(self._pipeline)

        # Bind buffers and textures.
        # Textures use a separate binding namespace (texture indices).
        temp_buffers = []
        buf_idx = 0
        if self._argument_encoder is not None:
            encoder.setBuffer_offset_atIndex_(self._argument_buffer, 0, 0)
            buf_idx = 1
        tex_idx = 0
        for i, (arg, ptype, is_field, is_tex) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field,
                    self._param_is_texture)):
            if is_tex:
                # A MetalTextureImage, in the texture binding namespace
                encoder.setTexture_atIndex_(arg.texture, tex_idx)
                tex_idx += 1
            elif is_field:
                # Update every dispatch: a cached variant can receive new
                # buffers or a different alias relationship. The indirect
                # resources also require explicit residency declarations.
                buffer = arg._buffer.metal_buffer
                self._argument_encoder.setBuffer_offset_atIndex_(buffer, 0, i)
                encoder.useResource_usage_(
                    buffer, Metal.MTLResourceUsageRead | Metal.MTLResourceUsageWrite)
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
        threads_per_group = self._workgroup_size
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
    kernel_name = kernel_entry_name(ir_func.name)
    msl_source = generate_msl_source(ir_func)

    # Debug: dump MSL source for analysis
    from tack.runtime.dispatch import env_flag
    if env_flag("TACK_DUMP_MSL"):
        path = f"/tmp/tack_{kernel_name}.msl"
        with open(path, "w") as f:
            f.write(msl_source)
        print(f"[Tack] Dumped MSL to {path}")

    options = Metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    library, error = device.newLibraryWithSource_options_error_(
        msl_source, options, None
    )
    if library is None:
        raise RuntimeError(f"Metal shader compilation failed:\n{error}\n\nMSL source:\n{msl_source}")

    func = library.newFunctionWithName_(kernel_name)
    if func is None:
        raise RuntimeError(f"Could not find '{kernel_name}' function in Metal library")

    pipeline, error = device.newComputePipelineStateWithFunction_error_(func, None)
    if pipeline is None:
        raise RuntimeError(f"Metal pipeline creation failed: {error}")

    param_types = [p.type_annotation for p in ir_func.params]
    param_is_field = [getattr(p, '_is_field', True) for p in ir_func.params]
    param_is_texture = [getattr(p, '_is_texture', False) for p in ir_func.params]
    argument_encoder = None
    if any(is_field and not is_texture for is_field, is_texture
           in zip(param_is_field, param_is_texture)):
        argument_encoder = func.newArgumentEncoderWithBufferIndex_(0)
        if argument_encoder is None:
            raise RuntimeError(f"Could not create argument encoder for '{kernel_name}'")
    return CompiledMetalKernel(device, command_queue, pipeline, kernel_name,
                               param_types, param_is_field, param_is_texture,
                               argument_encoder,
                               requires_full_workgroups=requires_full_workgroups(ir_func))


_REDUCE_MSL_SUM = field_reduction_source('metal', 'sum')
_REDUCE_MSL_MIN = field_reduction_source('metal', 'min')
_REDUCE_MSL_MAX = field_reduction_source('metal', 'max')


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
    supports_workgroups = True
    # Metal shared buffers live in unified memory, so a pointer into one is
    # CPU-addressable; the inherited memory_space() answer is right.

    # ...which is exactly why refusing a host tensor needs explaining. The
    # generic "cannot address it" is false here and sends a reader after the
    # wrong problem: the GPU can read that memory perfectly well. What it
    # cannot do is wrap it without a copy.
    dlpack_refusal_note = (
        "Metal's own allocations are host-addressable, but not the reverse: "
        "wrapping a host pointer as an MTLBuffer without copying needs a "
        "page-aligned address, which host allocations only get by accident "
        "of size. Use tack.field() + from_numpy(), or "
        "tack.from_dlpack(source, copy=True)."
    )


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

    def texture_in_hardware(self, shape_3d) -> bool:
        return True

    def create_texture_image(self, shape_3d) -> MetalTextureImage:
        return MetalTextureImage(self._device, self._command_queue, shape_3d)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap an existing MTLBuffer as a MetalBuffer without copying.

        An MTLBuffer *object*, not an address: Metal owns the mapping
        between the two and there is no way back from an integer. Saying
        so costs a line and saves reading `AttributeError: 'int' object
        has no attribute 'contents'` from four frames down.
        """
        if not hasattr(ptr, "contents"):
            raise TypeError(
                f"the Metal backend wraps an MTLBuffer object, not "
                f"{type(ptr).__name__}. An address cannot be turned back "
                f"into an MTLBuffer; allocate with tack.field() and copy "
                f"into it with from_numpy().")
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
        if loop_end <= 0:
            # range(0) runs nothing; do not dispatch an empty grid.
            return
        check_launch_size(f"Kernel '{kernel.name}'", loop_end, _MAX_LAUNCH, self.label)

        # Textures bind their own snapshot, not the field they came from.
        kernel_args = bind_textures(effective_args)

        # Replace scalar args with the packed field buffers. The buffers and
        # the argument buffer the launch encodes field bindings into belong
        # to the variant, so both are written, and read by the launch, under
        # its lock (see KernelVariant.dispatch_lock).
        with variant.dispatch_lock:
            if pack_info:
                from tack.lang.ir_pack_scalars import split_args
                from tack.runtime.kernel_utils import _update_pack_fields
                _update_pack_fields(pack_fields, pack_info, effective_args)
                kept_args = split_args(effective_args, pack_info)
                kernel_args = bind_textures(kept_args) + pack_fields

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
        options.setFastMathEnabled_(False)
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
        if field.size == 0:
            return empty_reduction(op)
        if field.dtype is not f32:
            # Fall back to numpy for non-f32
            return reduce_numpy(field.to_numpy(), op)
        n = int(np.prod(field.shape))
        if n >= _MAX_LAUNCH:
            # The kernel holds the count and its thread position in 32 bits.
            # Reduce the shared buffer in place rather than copy 16 GB.
            return reduce_numpy(field._buffer._view, op)

        pipeline = self._get_reduce_pipeline(op)

        # Create output buffer: [result, n_as_float_bits]
        import struct
        init_vals = REDUCTION_IDENTITIES
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

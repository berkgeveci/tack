"""PGC Metal compute backend — compiles kernels to MSL and dispatches on GPU.

Pipeline:
    PGC IR → MSL (via msl_gen.py) → Metal compute pipeline

Each field parameter becomes a Metal buffer at index matching its binding number.
The parallel loop range is dispatched as a 1D grid of threads.

On Apple Silicon, Metal shared buffers live in unified memory accessible by both
CPU and GPU.  Fields are backed directly by Metal buffer memory — no per-dispatch
copies are needed.
"""

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.msl_gen import generate_msl_source

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


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments."""
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


class CompiledMetalKernel:
    """A compiled Metal compute pipeline ready for dispatch."""

    _NUMPY_MAP = {f32: np.float32, i32: np.int32, i64: np.int64, u32: np.uint32, u64: np.uint64}

    def __init__(self, device, command_queue, pipeline, func_name, param_types, param_is_field):
        self._device = device
        self._command_queue = command_queue
        self._pipeline = pipeline
        self._func_name = func_name
        self._param_types = param_types
        self._param_is_field = param_is_field
        self._thread_execution_width = pipeline.threadExecutionWidth()
        self._max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup()

    def __call__(self, kernel_args: list, loop_end: int):
        """Dispatch the compute kernel on the GPU."""
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoderWithDescriptor_(
            Metal.MTLComputePassDescriptor.computePassDescriptor()
        )
        encoder.setComputePipelineState_(self._pipeline)

        # Bind buffers — fields use their Metal buffers, scalars use temp buffers
        temp_buffers = []
        for i, (arg, ptype, is_field) in enumerate(
                zip(kernel_args, self._param_types, self._param_is_field)):
            if is_field:
                encoder.setBuffer_offset_atIndex_(arg._buffer.metal_buffer, 0, i)
            else:
                # Scalar: create a tiny shared buffer with the value
                ndt = self._NUMPY_MAP[ptype]
                arr = np.array([arg], dtype=ndt)
                nbytes = arr.nbytes
                buf = self._device.newBufferWithBytes_length_options_(
                    arr.tobytes(), nbytes, 0)
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
                temp_buffers.append(buf)  # prevent GC

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
    """Compile a PGC IR function to a Metal compute pipeline."""
    msl_source = generate_msl_source(ir_func)

    # Debug: dump MSL source for analysis
    import os
    if os.environ.get("PGC_DUMP_MSL"):
        path = f"/tmp/pgc_{ir_func.name}.msl"
        with open(path, "w") as f:
            f.write(msl_source)
        print(f"[PGC] Dumped MSL to {path}")

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
    return CompiledMetalKernel(device, command_queue, pipeline, ir_func.name,
                               param_types, param_is_field)


class MetalBackend:
    """Metal GPU backend — zero-copy dispatch on Apple Silicon unified memory.

    Fields are backed by Metal shared buffers allocated at field creation time.
    from_numpy/to_numpy operate on the numpy view into shared memory — no
    host↔device transfers needed.
    """

    def __init__(self):
        if Metal is None:
            raise ImportError(
                "Metal backend requires pyobjc-framework-Metal. "
                "Install with: pip install 'pgc[metal]'"
            )

        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError("No Metal-capable GPU found")

        self._command_queue = self._device.newCommandQueue()
        self._cache: dict[str, CompiledMetalKernel] = {}

    def allocate_field(self, dtype: ScalarType, shape: tuple[int, ...]) -> MetalBuffer:
        return MetalBuffer(self._device, dtype.numpy_dtype, shape)

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Metal GPU with zero-copy dispatch."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        # Detect template arguments and expand them
        from pgc.runtime.cpu import (
            _detect_template_args, _expand_template_args,
            _detect_vector_fields_from_args,
        )
        template_args = _detect_template_args(kernel, args)
        effective_args = _expand_template_args(args, template_args)

        # Detect vector fields
        vector_fields = _detect_vector_fields_from_args(kernel, args, template_args)

        # Get IR
        ir_module = kernel.get_ir(
            vector_fields,
            template_args=template_args if template_args else None,
        )
        ir_func = ir_module.functions[0]

        # Resolve dimension sizes
        name_to_field = {}
        for param, arg in zip(ir_func.params, effective_args):
            if isinstance(arg, Field):
                name_to_field[param.name] = arg
        from pgc.lang.ir_resolve import resolve_ir
        resolve_ir(ir_func, name_to_field)

        # Type inference
        infer_param_types(ir_func, effective_args)

        # Optimization passes (LICM, CSE)
        from pgc.lang.ir_optimize import optimize_ir
        optimize_ir(ir_func)

        # Cache key
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        tmpl_key = ""
        if template_args:
            tmpl_key = str(kernel._make_cache_key(vector_fields, template_args))
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}_{tmpl_key}"

        if cache_key not in self._cache:
            self._cache[cache_key] = _compile_kernel(
                self._device, self._command_queue, ir_func
            )

        compiled = self._cache[cache_key]

        # Build kernel args list (fields and scalars in order)
        kernel_args = list(effective_args)

        # Determine loop range
        loop_end = _get_loop_range(ir_func, effective_args)

        # Dispatch
        compiled(kernel_args, loop_end)

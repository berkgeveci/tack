"""PGC Metal compute backend — compiles kernels via SPIR-V → MSL and dispatches on GPU.

Pipeline:
    PGC IR → SPIR-V binary → MSL (via spirv-cross) → Metal compute pipeline

Each field parameter becomes a Metal buffer at index matching its binding number.
The parallel loop range is dispatched as a 1D grid of threads.

On Apple Silicon, Metal shared buffers live in unified memory accessible by both
CPU and GPU.  Fields are backed directly by Metal buffer memory — no per-dispatch
copies are needed.
"""

import subprocess
import tempfile

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field, DeviceBuffer
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.spirv_gen import generate_spirv

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


def _spirv_to_msl(spirv_bytes: bytes) -> str:
    """Convert SPIR-V binary to Metal Shading Language via spirv-cross."""
    with tempfile.NamedTemporaryFile(suffix=".spv", delete=False) as f:
        f.write(spirv_bytes)
        f.flush()
        result = subprocess.run(
            ["spirv-cross", "--msl", f.name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"spirv-cross failed: {result.stderr}")
        return result.stdout


def _get_loop_range(ir_func: ir.IRFunction, args: tuple) -> int:
    """Extract the parallel for-loop range from the IR and actual arguments."""
    from pgc.runtime.cpu import _get_loop_range as cpu_get_loop_range
    return cpu_get_loop_range(ir_func, args)


class CompiledMetalKernel:
    """A compiled Metal compute pipeline ready for dispatch."""

    def __init__(self, device, command_queue, pipeline, func_name, param_types):
        self._device = device
        self._command_queue = command_queue
        self._pipeline = pipeline
        self._func_name = func_name
        self._param_types = param_types
        self._thread_execution_width = pipeline.threadExecutionWidth()
        self._max_threads_per_group = pipeline.maxTotalThreadsPerThreadgroup()

    def __call__(self, metal_buffers: list, loop_end: int):
        """Dispatch the compute kernel on the GPU.

        metal_buffers: list of MTLBuffer objects (already contain the field data).
        """
        command_buffer = self._command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoderWithDescriptor_(
            Metal.MTLComputePassDescriptor.computePassDescriptor()
        )
        encoder.setComputePipelineState_(self._pipeline)

        # Bind pre-existing Metal buffers — no data copy
        for i, buf in enumerate(metal_buffers):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

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
    spirv_bytes = generate_spirv(ir_func)
    msl_source = _spirv_to_msl(spirv_bytes)

    options = Metal.MTLCompileOptions.alloc().init()
    library, error = device.newLibraryWithSource_options_error_(
        msl_source, options, None
    )
    if library is None:
        raise RuntimeError(f"Metal shader compilation failed: {error}")

    func = library.newFunctionWithName_("main0")
    if func is None:
        raise RuntimeError("Could not find 'main0' function in Metal library")

    pipeline, error = device.newComputePipelineStateWithFunction_error_(func, None)
    if pipeline is None:
        raise RuntimeError(f"Metal pipeline creation failed: {error}")

    param_types = [p.type_annotation for p in ir_func.params]
    return CompiledMetalKernel(device, command_queue, pipeline, ir_func.name, param_types)


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

        # Get Metal buffers directly from fields (already allocated)
        metal_buffers = []
        for arg in effective_args:
            if isinstance(arg, Field):
                metal_buffers.append(arg._buffer.metal_buffer)
            else:
                raise NotImplementedError(
                    "Scalar kernel arguments not yet supported in Metal mode."
                )

        # Determine loop range
        loop_end = _get_loop_range(ir_func, effective_args)

        # Dispatch — no data copies, buffers already in GPU-visible memory
        compiled(metal_buffers, loop_end)

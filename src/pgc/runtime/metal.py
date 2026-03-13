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
import weakref

import numpy as np

from pgc.lang import ir
from pgc.lang.field import Field
from pgc.lang.types import ScalarType, f32, f64, i32, i64, u32, u64
from pgc.lang.type_inference import infer_param_types
from pgc.codegen.spirv_gen import generate_spirv

try:
    import Metal  # pyobjc-framework-Metal
except ImportError:
    Metal = None


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

    Fields are backed by Metal shared buffers.  The numpy array returned by
    field.data is a view into the Metal buffer, so CPU reads/writes and GPU
    compute operate on the same physical memory with no copies.
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

        # field id → MTLBuffer  (we also store a weakref to the field
        # so we can detect when a field is garbage-collected)
        self._field_buffers: dict[int, tuple] = {}  # id(field) → (MTLBuffer, weakref)

    def _get_metal_buffer(self, field: Field):
        """Get or create a Metal shared buffer backing this field.

        On first call for a given field, allocates a Metal shared buffer,
        copies the current numpy data in, then replaces field._data with a
        numpy view of the Metal buffer memory.  Subsequent calls return the
        cached buffer immediately.
        """
        fid = id(field)

        if fid in self._field_buffers:
            buf, ref = self._field_buffers[fid]
            # Check weakref is still alive (field not GC'd and id reused)
            if ref() is field:
                return buf
            # Stale entry — field was GC'd and id was reused
            del self._field_buffers[fid]

        nbytes = field.data.nbytes

        # Allocate Metal shared buffer with current field data
        # MTLResourceStorageModeShared = 0 (CPU+GPU unified memory)
        buf = self._device.newBufferWithBytes_length_options_(
            field.data.tobytes(), nbytes, 0
        )

        # Create a numpy view pointing directly at the Metal buffer memory
        raw = buf.contents().as_buffer(nbytes)
        view = np.frombuffer(raw, dtype=field.data.dtype).reshape(field.data.shape)

        # Replace the field's backing array with the Metal-backed view.
        # Now field.data, field.to_numpy(), field.from_numpy() all operate
        # directly on Metal shared memory — zero copy.
        field._data = view

        # Cache with a weak reference so we can detect field GC
        self._field_buffers[fid] = (buf, weakref.ref(field))

        return buf

    def execute(self, kernel, args, kwargs):
        """Execute a kernel on the Metal GPU with zero-copy dispatch."""
        if kwargs:
            raise NotImplementedError("Keyword arguments not supported in kernels")

        ir_module = kernel._ir
        ir_func = ir_module.functions[0]

        # Type inference
        infer_param_types(ir_func, args)

        # Cache key
        type_sig = tuple(p.type_annotation for p in ir_func.params)
        cache_key = f"{kernel.name}_{id(kernel)}_{type_sig}"

        if cache_key not in self._cache:
            self._cache[cache_key] = _compile_kernel(
                self._device, self._command_queue, ir_func
            )

        compiled = self._cache[cache_key]

        # Get Metal buffers for each field (zero-copy after first call)
        metal_buffers = []
        for arg in args:
            if isinstance(arg, Field):
                metal_buffers.append(self._get_metal_buffer(arg))
            else:
                raise NotImplementedError(
                    "Scalar kernel arguments not yet supported in Metal mode. "
                    "Use constants in the kernel body instead."
                )

        # Determine loop range
        loop_end = _get_loop_range(ir_func, args)

        # Dispatch — no data copies, buffers already in GPU-visible memory
        compiled(metal_buffers, loop_end)

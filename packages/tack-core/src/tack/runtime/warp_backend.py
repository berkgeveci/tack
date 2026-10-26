"""A prototype backend on NVIDIA Warp: Tack kernels compiled and launched by Warp.

``tack.device("warp")`` drives Warp's default device (its first CUDA
device, else its CPU). Arrays are ``wp.array`` buffers, and each variant
is a ``@wp.kernel`` that `codegen/warp_gen.py` writes from the typed IR,
which Warp compiles (and caches) and launches on its own stream. Scalars
go to Warp as kernel arguments, so nothing is packed.

Not offered: ``tack.inspect`` (``supports_inspect``). Workgroups on CUDA
devices only. Not yet: textures, device reductions and the disk cache's
variants (Warp's own kernel cache keeps the compiled code).
"""

import ctypes
import itertools
import linecache
import math

import numpy as np

from tack.codegen.warp_gen import _WARP_TYPES, generate_warp_source
from tack.lang.array import DeviceBuffer
from tack.runtime.gpu_backend import GPUBackend, _no_guard
from tack.runtime.kernel_utils import as_address, new_kernel_cache

_modules = itertools.count()


def _warp():
    import warp as wp
    return wp


def _warp_dtype(dtype):
    return getattr(_warp(), _WARP_TYPES[dtype][3:])


class WarpBuffer(DeviceBuffer):
    """A flat ``wp.array`` holding an array's elements."""

    backend_name = "warp"
    #: Memory something outside Tack reads or owns (a wrapped pointer, an
    #: export): launches that use it complete before returning.
    _shared = False

    def __init__(self, backend, dtype, shape, array=None):
        wp = _warp()
        self._backend = backend
        self._shape = tuple(shape)
        self._dtype = dtype
        if array is None:
            array = wp.empty(math.prod(self._shape), dtype=_warp_dtype(dtype),
                             device=backend.wp_device)
        self.array = array

    def share(self):
        """Complete the queued launches, and from now on every one that uses this
        array, as the other GPU backends do for memory handed outside Tack."""
        self._shared = True
        self.synchronize()

    def from_numpy(self, arr: np.ndarray):
        """On CUDA, one copy from ``arr``'s memory on the backend's stream, after
        the launches queued so far. From pageable memory the copy returns once
        ``arr`` is staged, so ``arr`` may change at once. Through a CPU
        ``wp.array`` and ``wp.copy`` it took 140 us for 1000 elements where
        Tack's CUDA backend takes 66."""
        wp = _warp()
        source = np.ascontiguousarray(arr, dtype=self._dtype.numpy_dtype).reshape(-1)
        device = self._backend.wp_device
        if not device.is_cuda:
            wp.copy(self.array, wp.array(source, dtype=self.array.dtype, device="cpu"))
            return
        if source.size:
            from warp._src.context import runtime
            if not runtime.core.wp_memcpy_h2d(device.context, ctypes.c_void_p(self.array.ptr),
                                              ctypes.c_void_p(source.ctypes.data),
                                              source.nbytes, device.stream.cuda_stream):
                raise RuntimeError("Warp backend: a copy to the device failed")

    def to_numpy(self) -> np.ndarray:
        return self.read_range(0, math.prod(self._shape)).reshape(self._shape)

    def read_range(self, start: int, count: int) -> np.ndarray:
        """``count`` elements from ``start``, after the launches queued so far.

        On CUDA, one copy on the backend's stream into a new NumPy array, then
        a wait for that stream: ``wp.array.numpy()`` made a host array, copied
        and waited for the whole device, 110 us for one element where Tack's
        CUDA backend takes 38 (RTX 4060 Ti)."""
        device = self._backend.wp_device
        if not device.is_cuda:
            return self.array[start:start + count].numpy().copy()
        out = np.empty(count, self._dtype.numpy_dtype)
        if count:
            from warp._src.context import runtime
            size = out.itemsize
            if not runtime.core.wp_memcpy_d2h(device.context, ctypes.c_void_p(out.ctypes.data),
                                              ctypes.c_void_p(self.array.ptr + start * size),
                                              count * size, device.stream.cuda_stream):
                raise RuntimeError("Warp backend: a copy from the device failed")
            _warp().synchronize_stream(device.stream)
        return out

    def fill(self, value):
        self.array.fill_(value)

    def synchronize(self):
        self._backend.synchronize()

    @property
    def address(self) -> int:
        return self.array.ptr or 0

    @property
    def nbytes(self) -> int:
        return math.prod(self._shape) * self._dtype.numpy_dtype.itemsize


class WarpBackend(GPUBackend):
    """Tack on Warp: Warp compiles and launches every kernel."""

    name = "warp"
    display_name = "Warp"
    supported_dtypes = frozenset(_WARP_TYPES)
    supports_device_reductions = False
    supports_workgroups = False
    supports_inspect = False
    #: wp.tid() is a 32-bit index.
    _max_launch = 2**31 - 1

    def __init__(self):
        wp = _warp()
        wp.config.quiet = True
        # Tack differentiates nothing; Warp's adjoint code is compile time spent.
        wp.config.enable_backward = False
        wp.init()
        self.wp_device = wp.get_device()
        # Shared arrays and barriers are CUDA's (native helpers in warp_gen).
        self.supports_workgroups = self.wp_device.is_cuda
        self._cache = new_kernel_cache()  # Kernel -> {variant_key: variant}
        self._kernels = {}                # generated source -> its wp.Kernel

    @property
    def memory_domain(self) -> tuple:
        return (self.name, str(self.wp_device))

    def allocate_array(self, dtype, shape, exportable=False):
        return WarpBuffer(self, dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        """Wrap memory someone else owns: an address on Warp's device, or on
        Warp's CPU a NumPy array, as the CPU backend takes. Warp does not free
        it (no deleter), and launches using it complete before returning
        (``_shared``), as on the other GPU backends."""
        count = math.prod(shape)
        nbytes = count * np.dtype(dtype.numpy_dtype).itemsize
        owner = None
        if isinstance(ptr, np.ndarray):
            if self.wp_device.is_cuda:
                raise TypeError("the Warp backend wraps a NumPy array only on Warp's CPU")
            owner = ptr                       # keeps the memory alive
            ptr = ptr.ctypes.data
        array = _warp().array(ptr=as_address(ptr), dtype=_warp_dtype(dtype), shape=(count,),
                              capacity=nbytes, device=self.wp_device)
        buf = WarpBuffer(self, dtype, shape, array)
        buf._owner = owner
        buf._shared = True
        return buf

    def synchronize(self):
        # Tack queues everything on the device's stream, so that is all to wait for.
        wp = _warp()
        if self.wp_device.is_cuda:
            wp.synchronize_stream(self.wp_device.stream)
        else:
            wp.synchronize_device(self.wp_device)

    def _build_variant(self, ir_func, effective_args):
        """Annotate and compile; Warp takes scalars as arguments, so no packing."""
        from tack.lang.ir_traversal import clone_ir
        from tack.lang.ir_type_annotate import annotate_types
        from tack.lang.ir_verify import verify_ir

        typed = clone_ir(ir_func)
        annotate_types(typed)
        verify_ir(typed, 'typed')
        return self._compile_kernel(typed), None, None

    def _compile_kernel(self, typed_ir):
        """The ``wp.Kernel`` for ``typed_ir``, defined from generated source.

        Warp reads a kernel's source through `inspect`, so the source is
        registered with `linecache` under a name of its own; the module
        name is unique too, so each variant is a Warp module of its own.
        """
        wp = _warp()
        source, entry, tables = generate_warp_source(typed_ir)
        kernel = self._kernels.get(source)
        if kernel is not None:
            # Two variants that differ only in what the code does not depend
            # on (a CPU-only disjointness, a dimension it never reads) compiled
            # the same module twice.
            return kernel
        number = next(_modules)
        filename = f"<tack-warp-{number}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"__name__": f"tack_warp_{number}"}
        exec(compile(source, filename, "exec"), namespace)
        kernel = namespace[entry]
        kernel.tack_source = source
        self._kernels[source] = kernel
        # Constant tables, made once on the device and passed at every launch.
        kernel.tack_tables = [
            wp.array(np.asarray(values, dtype=dtype.numpy_dtype),
                     dtype=getattr(wp, _WARP_TYPES[dtype][3:]), device=self.wp_device)
            for dtype, values in tables]
        return kernel

    def _launch(self, compiled, kernel_args, loop_end, extents):
        from tack.lang.array import Array
        # Arrays go by address: the generated kernel takes wp.uint64.
        inputs = [a._buffer.address if isinstance(a, Array) else a for a in kernel_args]
        inputs.append(loop_end)
        inputs.extend(extents)
        inputs.extend(t.ptr for t in compiled.tack_tables)
        # Blocks of 256 lanes: Tack's workgroup size, and Warp's default.
        _warp().launch(compiled, dim=loop_end, inputs=inputs, device=self.wp_device,
                       block_dim=256)

    def _launch_guard(self, variant):
        # Every launch passes all its arguments; nothing is shared between calls.
        return _no_guard

"""A prototype backend on NVIDIA Warp: Tack kernels compiled and launched by Warp.

``tack.device("warp")`` drives Warp's default device (its first CUDA
device, else its CPU). Arrays are ``wp.array`` buffers, and each variant
is a ``@wp.kernel`` that `codegen/warp_gen.py` writes from the typed IR,
which Warp compiles (and caches) and launches on its own stream. Scalars
go to Warp as kernel arguments, so nothing is packed.

Not yet: textures, workgroups, device reductions, ``wrap_ptr`` and the
disk cache's variants (Warp's own kernel cache keeps the compiled code).
"""

import itertools
import linecache
import math

import numpy as np

from tack.codegen.warp_gen import _WARP_TYPES, generate_warp_source
from tack.lang.array import DeviceBuffer
from tack.runtime.gpu_backend import GPUBackend, _no_guard
from tack.runtime.kernel_utils import new_kernel_cache

_modules = itertools.count()


def _warp():
    import warp as wp
    return wp


class WarpBuffer(DeviceBuffer):
    """A flat ``wp.array`` holding an array's elements."""

    backend_name = "warp"

    def __init__(self, backend, dtype, shape):
        wp = _warp()
        self._backend = backend
        self._shape = tuple(shape)
        self._dtype = dtype
        self.array = wp.empty(math.prod(self._shape), dtype=getattr(wp, _WARP_TYPES[dtype][3:]),
                              device=backend.wp_device)

    def from_numpy(self, arr: np.ndarray):
        wp = _warp()
        source = np.ascontiguousarray(arr, dtype=self._dtype.numpy_dtype).reshape(-1)
        wp.copy(self.array, wp.array(source, dtype=self.array.dtype, device="cpu"))

    def to_numpy(self) -> np.ndarray:
        return self.array.numpy().reshape(self._shape).copy()

    def read_range(self, start: int, count: int) -> np.ndarray:
        return self.array[start:start + count].numpy().copy()

    def fill(self, value):
        self.array.fill_(value)

    def synchronize(self):
        _warp().synchronize_device(self._backend.wp_device)

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
    #: wp.tid() is a 32-bit index.
    _max_launch = 2**31 - 1

    def __init__(self):
        wp = _warp()
        wp.config.quiet = True
        # Tack differentiates nothing; Warp's adjoint code is compile time spent.
        wp.config.enable_backward = False
        wp.init()
        self.wp_device = wp.get_device()
        self._cache = new_kernel_cache()  # Kernel -> {variant_key: variant}

    @property
    def memory_domain(self) -> tuple:
        return (self.name, str(self.wp_device))

    def allocate_array(self, dtype, shape, exportable=False):
        return WarpBuffer(self, dtype, shape)

    def wrap_ptr(self, ptr, dtype, shape):
        raise NotImplementedError("Warp backend: wrap_ptr is not supported yet")

    def synchronize(self):
        _warp().synchronize_device(self.wp_device)

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
        source, entry = generate_warp_source(typed_ir)
        number = next(_modules)
        filename = f"<tack-warp-{number}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"__name__": f"tack_warp_{number}"}
        exec(compile(source, filename, "exec"), namespace)
        kernel = namespace[entry]
        kernel.tack_source = source
        return kernel

    def _launch(self, compiled, kernel_args, loop_end, extents):
        from tack.lang.array import Array
        inputs = [a._buffer.array if isinstance(a, Array) else a for a in kernel_args]
        inputs.append(loop_end)
        inputs.extend(extents)
        _warp().launch(compiled, dim=loop_end, inputs=inputs, device=self.wp_device)

    def _launch_guard(self, variant):
        # Every launch passes all its arguments; nothing is shared between calls.
        return _no_guard

    def _must_complete(self, kernel_args) -> bool:
        return False

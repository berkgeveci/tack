"""Tack: a Python-first compute framework.

Kernels are Python functions decorated with ``@tack.kernel``, compiled when
first called for the device they run on: the CPU (through LLVM), Metal, CUDA,
HIP or Level Zero. Arrays (``tack.empty``, ``tack.array``, ...) hold typed
values on a device, and a kernel runs where its arrays are. ``tack.dataset``,
in the tack-vis package, adds meshes, fields on them, and the algorithms of
scientific visualization.
"""

# Enable namespace package merging across tack-core and tack-vis
from importlib.metadata import PackageNotFoundError as _NotInstalled
from importlib.metadata import version as _installed_version
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
del extend_path

try:
    __version__ = _installed_version("tack-core")
except _NotInstalled:          # a source tree that was never installed
    __version__ = "0+unknown"
del _NotInstalled, _installed_version

from tack.lang.array import (
    Array,
    ExportedMemory,
    SoaArray,
    Texture3D,
    arange,
    array,
    array_from_ptr,
    asarray,
    concat,
    copyto,
    empty,
    empty_like,
    from_dlpack,
    full,
    full_like,
    memory_space,
    ones,
    ones_like,
    soa,
    zeros,
    zeros_like,
)
from tack.lang.constant import constant
from tack.lang.device_class import device_class
from tack.lang.func import func
from tack.lang.inspect_kernel import inspect
from tack.lang.kernel import kernel
from tack.lang.types import (
    ElementType,
    ScalarType,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    mat,
    mat2,
    mat3,
    mat4,
    u8,
    u16,
    u32,
    u64,
    vec,
    vec2,
    vec3,
    vec4,
)
from tack.runtime.dispatch import (
    Device,
    device,
    device_count,
    get_default_device,
    set_debug,
    set_default_device,
    sync,
    sync_async,
)


# Shared memory, barrier, thread_id — only usable inside @tack.kernel
def shared(dtype, size):
    """Allocate threadgroup shared memory. Only usable inside a @tack.kernel."""
    raise RuntimeError("shared() can only be used inside a @tack.kernel")

def shared_like(array, size):
    """Allocate shared memory with the same dtype as an array. Only usable inside a @tack.kernel."""
    raise RuntimeError("shared_like() can only be used inside a @tack.kernel")

def select(mask, a, b):
    """``a`` where ``mask`` is true, else ``b``, component by component.

    Only usable inside a kernel or device function. ``mask`` is a scalar
    or a comparison vector such as ``v > 0.0``; ``a`` and ``b`` are scalars
    or vectors of one shape, a scalar standing for every component.
    """
    raise RuntimeError("tack.select() can only be used inside a @tack.kernel")


def barrier():
    """Threadgroup synchronization barrier. Only usable inside a @tack.kernel."""
    raise RuntimeError("barrier() can only be used inside a @tack.kernel")

def thread_id():
    """Thread index within workgroup. Only usable inside a @tack.kernel."""
    raise RuntimeError("thread_id() can only be used inside a @tack.kernel")

# Atomic operations — only usable inside @tack.kernel
def atomic_add(array, index, value):
    """Atomic add: array[index] += value. Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_add() can only be used inside a @tack.kernel")

def atomic_min(array, index, value):
    """Atomic min: array[index] = min(array[index], value). Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_min() can only be used inside a @tack.kernel")

def atomic_max(array, index, value):
    """Atomic max: array[index] = max(array[index], value). Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_max() can only be used inside a @tack.kernel")

# ndindex for multi-dimensional parallel iteration
def ndindex(*args):
    """Multi-dimensional parallel iteration, as ``np.ndindex``.

    Used in kernels: ``for i, j in tack.ndindex(w, h)``; each argument is a
    size or a ``(start, end)`` pair. Cannot be called from Python directly.
    """
    raise RuntimeError("ndindex() can only be used inside a @tack.kernel")


def identity(n):
    """The n-by-n identity matrix, as ``np.identity``. Only usable inside a @tack.kernel."""
    raise RuntimeError("identity() can only be used inside a @tack.kernel")

def block_sum(value):
    """Sum a value across all threads in the workgroup. Only usable inside a @tack.kernel."""
    raise RuntimeError("block_sum() can only be used inside a @tack.kernel")

def block_max(value):
    """Max of a value across all threads in the workgroup. Only usable inside a @tack.kernel."""
    raise RuntimeError("block_max() can only be used inside a @tack.kernel")

def block_min(value):
    """Min of a value across all threads in the workgroup. Only usable inside a @tack.kernel."""
    raise RuntimeError("block_min() can only be used inside a @tack.kernel")

def local_array(dtype, size):
    """Allocate a per-thread local array. Only usable inside a @tack.kernel.

    Usage: arr = tack.local_array(tack.f32, 8)
    Then: arr[i] = value, value = arr[i]
    """
    raise RuntimeError("local_array() can only be used inside a @tack.kernel")

def local_array_like(array, size):
    """Allocate a per-thread local array with the same dtype as an array. Only usable inside a @tack.kernel."""
    raise RuntimeError("local_array_like() can only be used inside a @tack.kernel")

def texture3d(source_array, shape=None, interp='linear'):
    """Create a 3D texture from an array for hardware-accelerated sampling.

    The texture copies the array's data now. Later writes to the array --
    kernel stores, ``from_numpy``, external or DLPack writes -- are not
    visible to it until ``tex.update()`` copies the array again. That holds
    on every backend, including CPU.

    Args:
        source_array: A Tack array with f32 dtype holding W*H*D elements,
               x varying fastest.
        shape: (W, H, D) tuple for the 3D dimensions.  If the array's shape
               is already 3D, this can be omitted.
        interp: Interpolation mode. Only 'linear' (the default) is
               implemented.
    """
    if shape is None:
        shape = source_array.shape
    if len(shape) != 3:
        raise ValueError("texture3d requires a 3D shape")
    return Texture3D(source_array, shape_3d=shape, interp=interp)


#: Public submodules, loaded on first use, so ``import tack`` alone reaches
#: ``tack.arrays.CountingArray``. ``dataset`` and ``interop`` are tack-vis's.
_SUBMODULES = frozenset({"algorithms", "arrays", "cache", "math", "random", "dataset", "interop"})


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"{__name__}.{name}":
                raise
            raise AttributeError(f"module 'tack' has no attribute {name!r}: tack.{name} is "
                                 f"in the tack-vis package, which is not installed") from error
    raise AttributeError(f"module 'tack' has no attribute {name!r}")


__all__ = [
    "Array",
    "Device",
    "ElementType",
    "ExportedMemory",
    "ScalarType",
    "SoaArray",
    "Texture3D",
    "arange",
    "array",
    "array_from_ptr",
    "asarray",
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "barrier",
    "block_max",
    "block_min",
    "block_sum",
    "concat",
    "constant",
    "copyto",
    "device",
    "device_class",
    "device_count",
    "empty",
    "empty_like",
    "f32",
    "f64",
    "from_dlpack",
    "full",
    "full_like",
    "func",
    "get_default_device",
    "i8",
    "i16",
    "i32",
    "i64",
    "identity",
    "inspect",
    "kernel",
    "local_array",
    "local_array_like",
    "mat",
    "mat2",
    "mat3",
    "mat4",
    "memory_space",
    "ndindex",
    "ones",
    "ones_like",
    "select",
    "set_debug",
    "set_default_device",
    "shared",
    "shared_like",
    "soa",
    "sync",
    "sync_async",
    "texture3d",
    "thread_id",
    "u8",
    "u16",
    "u32",
    "u64",
    "vec",
    "vec2",
    "vec3",
    "vec4",
    "zeros",
    "zeros_like",
]

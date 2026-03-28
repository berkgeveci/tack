# Enable namespace package merging across tack-core, tack-rendering, tack-vis
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

"""Tack — GPU compute framework framework."""

from tack.lang.types import i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, template
from tack.lang.kernel import kernel
from tack.lang.inspect_kernel import inspect
from tack.lang.func import func
from tack.lang.field import (
    field, field_like, field_from_ptr, from_dlpack, memory_space,
    zeros, ones, full, arange, concat,
    Vector, Texture3D, ExportedMemory,
)
from tack.lang.data_oriented import data_oriented
from tack.runtime.dispatch import init

# Shared memory, barrier, thread_id — only usable inside @tack.kernel
def shared(dtype, size):
    """Allocate threadgroup shared memory. Only usable inside a @tack.kernel."""
    raise RuntimeError("shared() can only be used inside a @tack.kernel")

def shared_like(field, size):
    """Allocate shared memory with the same dtype as a field. Only usable inside a @tack.kernel."""
    raise RuntimeError("shared_like() can only be used inside a @tack.kernel")

def barrier():
    """Threadgroup synchronization barrier. Only usable inside a @tack.kernel."""
    raise RuntimeError("barrier() can only be used inside a @tack.kernel")

def thread_id():
    """Thread index within workgroup. Only usable inside a @tack.kernel."""
    raise RuntimeError("thread_id() can only be used inside a @tack.kernel")

# Atomic operations — only usable inside @tack.kernel
def atomic_add(field, index, value):
    """Atomic add: field[index] += value. Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_add() can only be used inside a @tack.kernel")

def atomic_min(field, index, value):
    """Atomic min: field[index] = min(field[index], value). Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_min() can only be used inside a @tack.kernel")

def atomic_max(field, index, value):
    """Atomic max: field[index] = max(field[index], value). Only usable inside a @tack.kernel."""
    raise RuntimeError("atomic_max() can only be used inside a @tack.kernel")

# ndrange for multi-dimensional parallel iteration
def ndrange(*args):
    """Multi-dimensional parallel iteration range.

    Used in kernels: for i, j in tack.ndrange(w, h)
    Cannot be called from Python directly.
    """
    raise RuntimeError("ndrange() can only be used inside a @tack.kernel")

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

def local_array_like(field, size):
    """Allocate a per-thread local array with the same dtype as a field. Only usable inside a @tack.kernel."""
    raise RuntimeError("local_array_like() can only be used inside a @tack.kernel")

def texture3d(source_field, shape=None, interp='linear'):
    """Create a 3D texture from a field for hardware-accelerated sampling.

    Args:
        source_field: A tack.field with f32 dtype.
        shape: (W, H, D) tuple for the 3D dimensions.  If the field's shape
               is already 3D, this can be omitted.
        interp: Interpolation mode — 'linear' (default) or 'nearest'.
    """
    if shape is None:
        shape = source_field.shape
    if len(shape) != 3:
        raise ValueError("texture3d requires a 3D shape")
    return Texture3D(source_field, shape_3d=shape, interp=interp)


# Backend selectors
cpu = "cpu"
metal = "metal"
cuda = "cuda"
hip = "hip"
level_zero = "level_zero"

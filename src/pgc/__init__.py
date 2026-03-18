"""PGC — Portable GPU Compute framework."""

from pgc.lang.types import f32, f64, i32, i64, u32, u64, template
from pgc.lang.kernel import kernel
from pgc.lang.func import func
from pgc.lang.field import field, field_like, field_from_ptr, Vector, Texture3D
from pgc.lang.data_oriented import data_oriented
from pgc.runtime.dispatch import init

# Shared memory, barrier, thread_id — only usable inside @pgc.kernel
def shared(dtype, size):
    """Allocate threadgroup shared memory. Only usable inside a @pgc.kernel."""
    raise RuntimeError("shared() can only be used inside a @pgc.kernel")

def barrier():
    """Threadgroup synchronization barrier. Only usable inside a @pgc.kernel."""
    raise RuntimeError("barrier() can only be used inside a @pgc.kernel")

def thread_id():
    """Thread index within workgroup. Only usable inside a @pgc.kernel."""
    raise RuntimeError("thread_id() can only be used inside a @pgc.kernel")

# Atomic operations — only usable inside @pgc.kernel
def atomic_add(field, index, value):
    """Atomic add: field[index] += value. Only usable inside a @pgc.kernel."""
    raise RuntimeError("atomic_add() can only be used inside a @pgc.kernel")

def atomic_min(field, index, value):
    """Atomic min: field[index] = min(field[index], value). Only usable inside a @pgc.kernel."""
    raise RuntimeError("atomic_min() can only be used inside a @pgc.kernel")

def atomic_max(field, index, value):
    """Atomic max: field[index] = max(field[index], value). Only usable inside a @pgc.kernel."""
    raise RuntimeError("atomic_max() can only be used inside a @pgc.kernel")

# ndrange for multi-dimensional parallel iteration
def ndrange(*args):
    """Multi-dimensional parallel iteration range.

    Used in kernels: for i, j in pgc.ndrange(w, h)
    Cannot be called from Python directly.
    """
    raise RuntimeError("ndrange() can only be used inside a @pgc.kernel")

def block_sum(value):
    """Sum a value across all threads in the workgroup. Only usable inside a @pgc.kernel."""
    raise RuntimeError("block_sum() can only be used inside a @pgc.kernel")

def block_max(value):
    """Max of a value across all threads in the workgroup. Only usable inside a @pgc.kernel."""
    raise RuntimeError("block_max() can only be used inside a @pgc.kernel")

def block_min(value):
    """Min of a value across all threads in the workgroup. Only usable inside a @pgc.kernel."""
    raise RuntimeError("block_min() can only be used inside a @pgc.kernel")

def local_array(dtype, size):
    """Allocate a per-thread local array. Only usable inside a @pgc.kernel.

    Usage: arr = pgc.local_array(pgc.f32, 8)
    Then: arr[i] = value, value = arr[i]
    """
    raise RuntimeError("local_array() can only be used inside a @pgc.kernel")

def texture3d(source_field, shape=None, interp='linear'):
    """Create a 3D texture from a field for hardware-accelerated sampling.

    Args:
        source_field: A pgc.field with f32 dtype.
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
vulkan = "vulkan"
hip = "hip"
level_zero = "level_zero"

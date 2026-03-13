"""PGC — Portable GPU Compute framework."""

from pgc.lang.types import f32, f64, i32, i64, u32, u64, template
from pgc.lang.kernel import kernel
from pgc.lang.func import func
from pgc.lang.field import field, Vector
from pgc.lang.data_oriented import data_oriented
from pgc.runtime.dispatch import init

# ndrange for multi-dimensional parallel iteration
def ndrange(*args):
    """Multi-dimensional parallel iteration range.

    Used in kernels: for i, j in pgc.ndrange(w, h)
    Cannot be called from Python directly.
    """
    raise RuntimeError("ndrange() can only be used inside a @pgc.kernel")

# Backend selectors
cpu = "cpu"
metal = "metal"
cuda = "cuda"
vulkan = "vulkan"
hip = "hip"

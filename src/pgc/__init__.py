"""PGC — Portable GPU Compute framework."""

from pgc.lang.types import f32, f64, i32, i64, u32, u64, template
from pgc.lang.kernel import kernel
from pgc.lang.field import field
from pgc.runtime.dispatch import init

# Backend selectors
cpu = "cpu"
metal = "metal"
vulkan = "vulkan"
hip = "hip"

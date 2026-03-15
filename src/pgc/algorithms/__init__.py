"""pgc.algorithms — Reusable GPU algorithm primitives.

Provides portable, backend-agnostic building blocks for common parallel
patterns: scan, copy, fill.  All operate on pgc.field objects and run
entirely on the active backend (CPU, Metal, CUDA, HIP, Vulkan).
"""

from pgc.algorithms.scan import exclusive_scan, inclusive_scan
from pgc.algorithms.copy import copy, fill_value

__all__ = [
    "exclusive_scan",
    "inclusive_scan",
    "copy",
    "fill_value",
]

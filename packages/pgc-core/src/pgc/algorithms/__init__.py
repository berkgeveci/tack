"""pgc.algorithms — Reusable GPU algorithm primitives (core).

Provides portable, backend-agnostic building blocks for common parallel
patterns: scan, copy, fill.  All operate on pgc.field objects and run
entirely on the active backend (CPU, Metal, CUDA, HIP, Level Zero).
"""

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from pgc.algorithms.scan import exclusive_scan, inclusive_scan
from pgc.algorithms.copy import copy, fill_value

__all__ = [
    "exclusive_scan",
    "inclusive_scan",
    "copy",
    "fill_value",
]

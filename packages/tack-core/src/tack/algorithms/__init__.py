"""tack.algorithms — Reusable GPU algorithm primitives (core).

Provides portable, backend-agnostic building blocks for common parallel
patterns: scan, copy, fill.  All operate on tack.field objects and run
entirely on the active backend (CPU, Metal, CUDA, HIP, Level Zero).
"""

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from tack.algorithms.scan import exclusive_scan, inclusive_scan
from tack.algorithms.copy import copy, fill_value
from tack.algorithms.stats import (
    var, std, norm, absmax, count_nonzero, dot, histogram,
)

__all__ = [
    "exclusive_scan",
    "inclusive_scan",
    "copy",
    "fill_value",
    "var",
    "std",
    "norm",
    "absmax",
    "count_nonzero",
    "dot",
    "histogram",
]

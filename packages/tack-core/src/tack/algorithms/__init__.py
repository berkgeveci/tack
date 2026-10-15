"""tack.algorithms — GPU algorithm primitives.

Portable, backend-agnostic building blocks for common parallel patterns:
scan, sort, segmented reductions, copy, fill, stats. All operate on
tack.field objects and run as kernels on the active backend (CPU, Metal,
CUDA, HIP, Level Zero), reading back at most a scalar result.

Visualization algorithms are in ``tack.dataset`` (tack-vis).
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from tack.algorithms.copy import copy, fill_value
from tack.algorithms.scan import exclusive_scan, inclusive_scan
from tack.algorithms.sort import argsort, gather, reduce_by_key, sort_by_key, unique
from tack.algorithms.stats import (
    absmax,
    count_nonzero,
    dot,
    histogram,
    norm,
    std,
    var,
)

__all__ = [
    "absmax",
    "argsort",
    "copy",
    "count_nonzero",
    "dot",
    "exclusive_scan",
    "fill_value",
    "gather",
    "histogram",
    "inclusive_scan",
    "norm",
    "reduce_by_key",
    "sort_by_key",
    "std",
    "unique",
    "var",
]

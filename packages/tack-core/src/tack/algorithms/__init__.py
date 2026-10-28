"""tack.algorithms — GPU algorithm primitives.

Portable, backend-agnostic building blocks for common parallel patterns:
scan, sort, segmented reductions, copy, fill, stats. Their inputs are any
arrays a kernel indexes -- stored Tack arrays or the implicit arrays of
``tack.arrays`` -- and their outputs stored arrays; they run as kernels on
their inputs' device (CPU, Metal, CUDA, HIP, Level Zero), reading back at
most a scalar result.

Visualization algorithms are in ``tack.dataset`` (tack-vis).
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from tack.algorithms.copy import copy, fill_value
from tack.algorithms.scan import exclusive_scan, inclusive_scan
from tack.algorithms.segmented import (
    segment_broadcast,
    segment_ids,
    segment_reduce,
    segment_scan,
)
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
    "segment_broadcast",
    "segment_ids",
    "segment_reduce",
    "segment_scan",
    "sort_by_key",
    "std",
    "unique",
    "var",
]

"""tack.algorithms — GPU algorithm primitives, plus the vis worklets.

Core provides portable, backend-agnostic building blocks for common
parallel patterns: scan, copy, fill, stats.  All operate on tack.field
objects and run entirely on the active backend (CPU, Metal, CUDA, HIP,
Level Zero).

When tack-vis is installed, its visualization worklets join this same
namespace — flying edges, compute normals, cell-to-point.

Why the re-export below lives here rather than in tack-vis
---------------------------------------------------------
Both packages ship a ``tack/algorithms/`` directory, and ``extend_path``
merges their *contents* — but only one ``__init__.py`` ever executes, the
first one found on the path, which is this one.  tack-vis's copy was dead
code: ``from tack.algorithms import flying_edges`` raised ImportError in a
fresh process even though the docs promised it, and it only appeared to
work after something else had already imported the submodule (which binds
the *module* under that name, not the function).

So the names are re-exported here, guarded, and tack-core keeps working
on its own when tack-vis is not installed.
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

# tack-vis worklets, if that package is installed. The modules live in the
# tack-vis tree and reach this namespace through extend_path above.
try:
    from tack.algorithms.flying_edges import (
        flying_edges, flying_edges_multiblock, UniformGrid, MCTables,
    )
    from tack.algorithms.compute_normals import compute_normals
    from tack.algorithms.cell_to_point import cell_to_point
except ImportError:
    pass  # tack-vis not installed — core primitives above still work
else:
    __all__ += [
        "flying_edges",
        "flying_edges_multiblock",
        "UniformGrid",
        "MCTables",
        "compute_normals",
        "cell_to_point",
    ]

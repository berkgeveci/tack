"""tack.algorithms — Visualization algorithms (tack-vis).

Extends tack.algorithms with scientific visualization worklets:
flying edges, compute normals, cell-to-point, AMR blanking.
"""

from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from tack.algorithms.flying_edges import flying_edges, flying_edges_multiblock, UniformGrid, MCTables
from tack.algorithms.compute_normals import compute_normals
from tack.algorithms.cell_to_point import cell_to_point

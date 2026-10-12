"""The first vis worklets, superseded by ``tack.data.algorithms``; see ``tack.deprecated``."""

from tack.deprecated.algorithms.cell_to_point import cell_to_point
from tack.deprecated.algorithms.compute_normals import compute_normals
from tack.deprecated.algorithms.flying_edges import (
    MCTables,
    UniformGrid,
    flying_edges,
    flying_edges_multiblock,
)

__all__ = [
    "MCTables",
    "UniformGrid",
    "cell_to_point",
    "compute_normals",
    "flying_edges",
    "flying_edges_multiblock",
]

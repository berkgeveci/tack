"""Tack's dataset API, prototype: see ``docs/design/dataset-api.md``.

``shapes`` holds the linear cells as template classes; ``topology`` the
unstructured and structured topologies and the faces and edges derived
from them; ``dataset`` the spaces, fields, geometry, ``DataSet`` and
``for_each``; ``algorithms`` a few algorithms over them.
"""

from tack.data import algorithms, shapes
from tack.data.dataset import (
    H1,
    L2,
    Constant,
    DataSet,
    Field,
    RectilinearCoordinates,
    Values,
    for_each,
    rectilinear_grid,
)
from tack.data.topology import StructuredTopology, UnstructuredTopology

__all__ = ["H1", "L2", "Constant", "DataSet", "Field", "RectilinearCoordinates",
           "StructuredTopology", "UnstructuredTopology", "Values", "algorithms",
           "for_each", "rectilinear_grid", "shapes"]

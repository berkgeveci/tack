"""Tack's dataset API, prototype: see ``docs/design/dataset-api.md``.

``shapes`` holds the linear cells as template classes; ``arrays`` the
implicit arrays a field's values may be; ``topology`` the
unstructured and structured topologies and the faces and edges derived
from them; ``dataset`` the spaces, fields, geometry, ``DataSet`` and
``for_each``; ``algorithms`` a few algorithms over them.
"""

from tack.data import algorithms, arrays, shapes
from tack.data.arrays import CartesianProduct, ConstantArray, CountingArray
from tack.data.dataset import (
    H1,
    L2,
    Constant,
    DataSet,
    Field,
    Values,
    for_each,
    rectilinear_grid,
)
from tack.data.topology import StructuredTopology, UnstructuredTopology

__all__ = ["H1", "L2", "CartesianProduct", "Constant", "ConstantArray", "CountingArray",
           "DataSet", "Field",
           "StructuredTopology", "UnstructuredTopology", "Values", "algorithms",
           "for_each", "rectilinear_grid", "shapes"]

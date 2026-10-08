"""Tack's dataset API, prototype: see ``docs/design/dataset-api.md``.

``shapes`` holds the linear cells as template classes; ``arrays`` the
implicit arrays a field's values may be; ``topology`` the unstructured and
structured topologies and the faces and edges derived from them; ``spaces``
where a field's values live on a topology, and the layout each owns;
``dataset`` fields, geometry, ``DataSet`` and ``for_each``; ``algorithms``
a few algorithms over them.
"""

from tack.data import algorithms, arrays, shapes, spaces
from tack.data.arrays import CartesianProduct, ConstantArray, CountingArray
from tack.data.dataset import DataSet, Field, for_each, rectilinear_grid, traces
from tack.data.spaces import H1, L2, Constant, SideTraces, Space, Values
from tack.data.topology import StructuredTopology, UnstructuredTopology

__all__ = ["H1", "L2", "CartesianProduct", "Constant", "ConstantArray", "CountingArray",
           "DataSet", "Field", "SideTraces", "Space", "StructuredTopology",
           "UnstructuredTopology", "Values", "algorithms", "arrays", "for_each",
           "rectilinear_grid", "shapes", "spaces", "traces"]

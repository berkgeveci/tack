"""Tack's dataset API, prototype: see ``docs/design/dataset-api.md``.

``shapes`` holds the linear cells as template classes; ``arrays`` the
implicit arrays a field's values may be; ``topology`` the unstructured and
structured topologies and the faces and edges derived from them; ``spaces``
where a field's values live on a topology, and the layout each owns;
``dataset`` fields, geometry, ``DataSet`` and ``for_each``; ``algorithms``
a few algorithms over them; ``filters`` contour, slice, threshold and
external faces; ``polyhedra`` the polyhedral topology, every cell a list of
faces stored once.
"""

from tack.data import algorithms, arrays, filters, polyhedra, shapes, spaces
from tack.data.arrays import CartesianProduct, ConstantArray, CountingArray
from tack.data.dataset import DataSet, Field, for_each, rectilinear_grid, traces
from tack.data.filters import contour, external_faces, slice_plane, threshold
from tack.data.polyhedra import (
    PolygonalTopology,
    PolyhedralTopology,
    SizeBuckets,
    as_polygons,
    as_polyhedra,
    check_winding,
    orient,
)
from tack.data.spaces import H1, L2, Constant, SideTraces, Space, Values
from tack.data.topology import StructuredTopology, UnstructuredTopology

__all__ = [
    "H1",
    "L2",
    "CartesianProduct",
    "Constant",
    "ConstantArray",
    "CountingArray",
    "DataSet",
    "Field",
    "PolygonalTopology",
    "PolyhedralTopology",
    "SideTraces",
    "SizeBuckets",
    "Space",
    "StructuredTopology",
    "UnstructuredTopology",
    "Values",
    "algorithms",
    "arrays",
    "as_polygons",
    "as_polyhedra",
    "check_winding",
    "contour",
    "external_faces",
    "filters",
    "for_each",
    "orient",
    "polyhedra",
    "rectilinear_grid",
    "shapes",
    "slice_plane",
    "spaces",
    "threshold",
    "traces",
]

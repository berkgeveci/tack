"""Tack's dataset API, prototype: see ``docs/design/dataset-api.md``.

``shapes`` holds the linear cells as template classes; ``arrays`` the
implicit arrays a field's values may be; ``topology`` the unstructured and
structured topologies and the faces and edges derived from them; ``spaces``
where a field's values live on a topology, and the layout each owns;
``dataset`` fields, geometry, ``DataSet`` and ``for_each``; ``algorithms``
a few algorithms over them; ``filters`` contour, slice, threshold, external
faces and the extractions and masks; ``carry`` how filters carry fields;
``implicit`` the implicit functions (plane, sphere, cylinder, box, planes);
``polyhedra`` the polyhedral topology, every cell a list of faces stored
once.
"""

from tack.data import algorithms, arrays, filters, implicit, polyhedra, shapes, spaces
from tack.data.arrays import CartesianProduct, ConstantArray, CountingArray
from tack.data.dataset import DataSet, Field, for_each, rectilinear_grid, traces
from tack.data.filters import (
    contour,
    external_faces,
    extract_cells,
    extract_geometry,
    extract_points,
    mask,
    mask_points,
    slice,
    slice_plane,
    threshold,
    threshold_points,
)
from tack.data.implicit import Box, Cylinder, Plane, Planes, Sphere
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
    "Box",
    "CartesianProduct",
    "Constant",
    "ConstantArray",
    "CountingArray",
    "Cylinder",
    "DataSet",
    "Field",
    "Plane",
    "Planes",
    "PolygonalTopology",
    "PolyhedralTopology",
    "SideTraces",
    "SizeBuckets",
    "Space",
    "Sphere",
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
    "extract_cells",
    "extract_geometry",
    "extract_points",
    "filters",
    "for_each",
    "implicit",
    "mask",
    "mask_points",
    "orient",
    "polyhedra",
    "rectilinear_grid",
    "shapes",
    "slice",
    "slice_plane",
    "spaces",
    "threshold",
    "threshold_points",
    "traces",
]

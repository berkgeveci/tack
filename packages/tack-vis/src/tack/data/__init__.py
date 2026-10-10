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

from tack.data import (
    algorithms,
    arrays,
    filters,
    implicit,
    polyhedra,
    shapes,
    sources,
    spaces,
    transforms,
)
from tack.data.arrays import (
    CartesianProduct,
    Cast,
    Components,
    Concatenate,
    ConstantArray,
    CountingArray,
    ExtractComponent,
    Function,
    Permutation,
    RandomNormal,
    RandomUniform,
    Strided,
    Transform,
    UniformCoordinates,
    View,
)
from tack.data.clip import clip
from tack.data.dataset import (
    DataSet,
    Field,
    for_each,
    rectilinear_grid,
    traces,
    uniform_grid,
)
from tack.data.filters import (
    contour,
    external_faces,
    extract_cells,
    extract_geometry,
    extract_points,
    mask,
    mask_points,
    point_cloud,
    shrink,
    slice,
    slice_plane,
    tetrahedralize,
    threshold,
    threshold_points,
    triangulate,
)
from tack.data.flow import advect, streamlines
from tack.data.implicit import Box, Cylinder, Plane, Planes, Sphere
from tack.data.locator import CellLocator, probe
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
    "Cast",
    "CellLocator",
    "Components",
    "Concatenate",
    "Constant",
    "ConstantArray",
    "CountingArray",
    "Cylinder",
    "DataSet",
    "ExtractComponent",
    "Field",
    "Function",
    "Permutation",
    "Plane",
    "Planes",
    "PolygonalTopology",
    "PolyhedralTopology",
    "RandomNormal",
    "RandomUniform",
    "SideTraces",
    "SizeBuckets",
    "Space",
    "Sphere",
    "Strided",
    "StructuredTopology",
    "Transform",
    "UniformCoordinates",
    "UnstructuredTopology",
    "Values",
    "View",
    "advect",
    "algorithms",
    "arrays",
    "as_polygons",
    "as_polyhedra",
    "check_winding",
    "clip",
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
    "point_cloud",
    "polyhedra",
    "probe",
    "rectilinear_grid",
    "shapes",
    "shrink",
    "slice",
    "slice_plane",
    "sources",
    "spaces",
    "streamlines",
    "tetrahedralize",
    "threshold",
    "threshold_points",
    "traces",
    "transforms",
    "triangulate",
    "uniform_grid",
]

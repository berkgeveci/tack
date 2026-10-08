"""Tack's data model for visualization: cell shapes, cell sets, datasets and filters.

``tack.data.shapes`` holds VTK's linear cells as template classes;
``tack.data.cell_set`` the cell sets, ``tack.data.dataset`` the ``DataSet``
and ``for_each_shape``, which runs a kernel over a dataset's cells one
shape at a time, and ``tack.data.filters`` filters built on them.
"""

from tack.data import shapes
from tack.data.cell_set import ExplicitCellSet, SingleTypeCellSet, StructuredCellSet
from tack.data.coordinates import RectilinearCoordinates
from tack.data.dataset import DataSet, for_each_shape, rectilinear_grid
from tack.data.filters import (
    cell_centers,
    cell_data_to_point_data,
    contour,
    external_faces,
    point_data_to_cell_data,
    point_links,
    slice_plane,
    threshold,
)

__all__ = [
    "DataSet",
    "ExplicitCellSet",
    "RectilinearCoordinates",
    "SingleTypeCellSet",
    "StructuredCellSet",
    "cell_centers",
    "cell_data_to_point_data",
    "contour",
    "external_faces",
    "for_each_shape",
    "point_data_to_cell_data",
    "point_links",
    "rectilinear_grid",
    "shapes",
    "slice_plane",
    "threshold",
]

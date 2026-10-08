"""Tack's data model for visualization: cell shapes, cell sets, datasets and filters.

``tack.data.shapes`` holds VTK's linear cells as template classes;
``tack.data.cell_set`` the cell sets, ``tack.data.dataset`` the ``DataSet``
and ``for_each_shape``, which runs a kernel over a dataset's cells one
shape at a time, and ``tack.data.filters`` filters built on them.
"""

from tack.data import shapes
from tack.data.cell_set import ExplicitCellSet, SingleTypeCellSet, StructuredCellSet
from tack.data.dataset import DataSet, for_each_shape
from tack.data.filters import (
    cell_centers,
    cell_data_to_point_data,
    external_faces,
    point_data_to_cell_data,
    point_links,
)

__all__ = ["DataSet", "ExplicitCellSet", "SingleTypeCellSet", "StructuredCellSet",
           "cell_centers", "cell_data_to_point_data", "external_faces", "for_each_shape",
           "point_data_to_cell_data", "point_links", "shapes"]

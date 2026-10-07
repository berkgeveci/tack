"""Tack's data model for visualization: cell shapes, cell sets and datasets.

``tack.data.shapes`` holds VTK's linear cells as template classes;
``tack.data.cell_set`` the cell sets, and ``tack.data.dataset`` the
``DataSet`` and ``for_each_shape``, which runs a kernel over a dataset's
cells one shape at a time.
"""

from tack.data import shapes
from tack.data.cell_set import ExplicitCellSet, SingleTypeCellSet, StructuredCellSet
from tack.data.dataset import DataSet, for_each_shape

__all__ = ["DataSet", "ExplicitCellSet", "SingleTypeCellSet", "StructuredCellSet",
           "for_each_shape", "shapes"]

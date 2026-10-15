"""``to_cells``: ``field`` averaged over each cell's corners, a ``Constant`` field --
``project`` into ``Constant``."""

from tack.dataset.algorithms.project import project
from tack.dataset.spaces import Constant

__all__ = ['to_cells']


def to_cells(data, field):
    """``field`` averaged over each cell's corners: a ``Constant`` field,
    ``project(data, field, Constant(data))``.

    For point data -- ``H1`` of order 1, or values on points -- each cell gets
    the mean of its points' values, as VTK's vtkPointDataToCellData does. Any
    field with a basis goes through its corner values: a DG field averages each
    cell's own, an order-2 field its values at the corners, and a cell
    constant comes back as itself. Scalars or vectors; the values must be
    floating point. On polyhedra, polygons and polylines, each cell averages
    its distinct points' values.
    """
    return project(data, field, Constant(data))

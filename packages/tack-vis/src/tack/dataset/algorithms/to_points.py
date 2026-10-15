"""``to_points``: ``field`` averaged onto the points, an ``H1`` field -- ``project``
into ``H1``."""

from tack.dataset.algorithms.project import project
from tack.dataset.spaces import H1

__all__ = ['to_points']


def to_points(data, field):
    """``field`` averaged onto the points: an order-1 ``H1`` field,
    ``project(data, field, H1(data))``. Each point averages the values the cells
    around it give it -- a cell's value, or for a DG field the cell's own value
    at that corner -- as vtkCellDataToPointData does; an order-2 field gives its
    values at the points. The sums run in a fixed order. Scalars or vectors. On
    polyhedra, polygons and polylines, cell values average over the cells
    around each point."""
    return project(data, field, H1(data))

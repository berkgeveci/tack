"""``to_points``: a ``Constant`` or ``L2`` field averaged onto the points: an ``H1`` field. Each point averages the values the cells around it give it -- a cell's value, or for a DG field the cell's own value at that corner. The sums run in a fixed order. On polyhedra and polygons, cell values (``Constant``, values on cells) average over the cells around each point, as vtkCellDataToPointData does."""

import tack
from tack.data.algorithms._common import _field
from tack.data.algorithms._links import _average_links
from tack.data.arrays import materialize, width_of
from tack.data.dataset import Field
from tack.data.spaces import H1, L2, Constant, Values
from tack.data.topology import corner_layout

__all__ = ['to_points']


@tack.kernel
def _incidences(cells, u, start, contributions):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            contributions[first + j] = u.dof(c, j)


@tack.kernel
def _polyhedral_incidences(point_offsets, values, contributions, n_cells):
    for c in range(n_cells):
        for e in range(point_offsets[c], point_offsets[c + 1]):
            contributions[e] = values[c]


def _polyhedral_to_points(data, field):
    """Cell values averaged onto the points of a polyhedral or polygonal topology,
    through its point links: each point's cells in cell order."""
    offsets, point_ids = data.topology.cell_points()
    total = int(point_ids.shape[0])
    values = materialize(field.values)
    contributions = tack.field(values.dtype, shape=(total,))
    out = tack.zeros(values.dtype, (data.num_points,))
    if total:
        _polyhedral_incidences(offsets, values, contributions, data.num_cells)
        link_offsets, entries = data.topology.point_links()
        _average_links(link_offsets, entries, contributions, out)
    return Field(H1(data), out)


def to_points(data, field):
    """A ``Constant`` or ``L2`` field averaged onto the points: an ``H1`` field. Each
    point averages the values the cells around it give it -- a cell's value, or for
    a DG field the cell's own value at that corner. The sums run in a fixed order.
    On polyhedra and polygons, cell values (``Constant``, values on cells) average
    over the cells around each point, as vtkCellDataToPointData does."""
    field = _field(data, field)
    if isinstance(field.space, Values) and field.space.on == "cells":
        field = Field(Constant(data), field.values)
    if not getattr(data.topology, "reference_cells", True):
        if not isinstance(field.space, Constant):
            raise TypeError("to_points on polyhedra averages cell values")
        if width_of(field.values):
            raise TypeError("to_points averages scalars")
        return _polyhedral_to_points(data, field)
    if not isinstance(field.space, (Constant, L2)):
        raise TypeError("to_points projects a cell or L2 field")
    # One contribution per (cell, corner), laid out by the topology's groups; a
    # subgroup's cells sit at their positions in their group (index(c)). Each
    # point then averages its corners through the topology's point links, in
    # layout order.
    start_of, total = corner_layout(data.topology)
    contributions = tack.field(field.values.dtype, shape=(total,))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            view = data.domain_view("cells", group)
            _incidences(view, field.view(group), start_of[id(group.root)], contributions)
    out = tack.zeros(field.values.dtype, (data.num_points,))
    if total:
        offsets, entries = data.topology.point_links()
        _average_links(offsets, entries, contributions, out)
    return Field(H1(data), out)

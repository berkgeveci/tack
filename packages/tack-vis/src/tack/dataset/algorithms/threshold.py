"""``threshold``: the cells whose ``field`` lies in ``[lower, upper]``, with only the points they use."""

import tack
from tack import arrays
from tack.dataset.algorithms._common import _field, _with_basis
from tack.dataset.algorithms._extract import _keep_cells
from tack.dataset.spaces import Constant

__all__ = ['threshold']


@tack.kernel
def _keep_by_corners(cells, u, lower, upper, all_points, keep):
    for c in cells:
        inside = 0
        for j in range(cells.NUM_POINTS):
            v = u.corner_value(c, j)
            if lower <= v and v <= upper:
                inside += 1
        ok = 0
        if all_points == 1:
            if inside == cells.NUM_POINTS:
                ok = 1
        elif inside > 0:
            ok = 1
        keep[cells.entity_id(c)] = ok


def threshold(data, field, lower, upper, all_points=True, fields=None):
    """The cells whose ``field`` lies in ``[lower, upper]``, with only the points they use.

    ``field`` is a scalar field or its name. A cell field (``Constant``, values
    on cells) keeps a cell by its value; a field with a basis (``H1``, ``L2``
    of any orders) or values on points by its values at the cell's corners:
    all of them in range, or with ``all_points=False`` any (VTK's
    ``AllScalars``). The result's cells keep their shapes and order; its points
    are the used ones, renumbered in order. Every field it can carry comes
    along: point fields gathered to the kept points, cell fields to the kept
    cells, ``L2`` fields (and an ``L2`` geometry) block by block, keeping each
    cell's order. Fields on faces and edges, which the new topology derives
    afresh, are left behind.

    On a polyhedral topology the kept cells keep their faces, stored once and
    numbered in their old order, so values on faces come along. A face whose
    side-0 cell is dropped is turned around for its remaining cell, and its
    oriented values (``Values(..., "faces", oriented=True)``) are negated. A
    polygonal topology's kept polygons keep their loops, and a polyline
    topology's polylines their chains; their facets' values, whose numbering is
    derived afresh, are left behind.
    """
    field = _field(data, field)
    if arrays.width_of(field.values):
        raise TypeError("threshold needs a scalar field")
    field = _with_basis(data, field)
    space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"threshold needs a field on the points or cells, not {space!r}")
    return _keep_cells(data, _threshold_flags(data, field, lower, upper, all_points), fields)


def _threshold_flags(data, field, lower, upper, all_points):
    """Each cell's 1 or 0: whether ``threshold`` keeps it."""
    field = _field(data, field)
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_flags(data, field, lower, upper, all_points)
    keep = tack.empty((data.num_cells,), tack.i32)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _keep_by_corners(data.domain_view("cells", group), field.view(group), lower,
                             upper, 1 if all_points else 0, keep)
    return keep


@tack.kernel
def _polyhedral_keep(cells, values, lower, upper, by_cells, all_points, keep):
    for c in cells:
        e = cells.entity_id(c)
        ok = 0
        if by_cells == 1:
            v = values[e]
            if lower <= v and v <= upper:
                ok = 1
        else:
            inside = 0
            n = cells.num_points(c)
            for j in range(n):
                v = values[cells.point_id(c, j)]
                if lower <= v and v <= upper:
                    inside += 1
            if all_points == 1:
                if inside == n:
                    ok = 1
            elif inside > 0:
                ok = 1
        keep[e] = ok


def _polyhedral_flags(data, field, lower, upper, all_points):
    """``threshold``'s cells of a polyhedral or polygonal topology (whole cells; their
    faces are kept once and numbered in their old order, points compacted)."""

    by_cells = isinstance(field.space, Constant)
    values = arrays.materialize(field.values)
    n = data.num_cells
    keep = tack.empty((n,), tack.i32)
    for group in data.launch_groups("cells", []):
        if group.count:
            _polyhedral_keep(data.domain_view("cells", group), values, lower, upper,
                             1 if by_cells else 0, 1 if all_points else 0, keep)
    return keep

"""``interpolate``: a field whose values are a function of position, evaluated at a space's nodes -- each node once, where the geometry puts it -- with no array of node positions stored."""

import tack
from tack.dataset import ids
from tack.dataset.buckets import bucket_order
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, Values, reference_cells
from tack.dataset.topology import corner_layout
from tack.lang.func import Func
from tack.runtime.dispatch import on_inputs_device

__all__ = ['interpolate']


@tack.device_class
class _Of:
    """The function as a template attribute: a kernel takes no callable arguments.
    ``at`` gives it world positions: the geometry's value plus the dataset's
    origin, added in the geometry's type, where the origin is not zero (compiled
    in, so a dataset at zero runs the function on its values as they are)."""

    SHIFTED = 0

    def __init__(self, function, origin):
        self.f = function
        self.ox, self.oy, self.oz = origin
        if any(origin):
            self.SHIFTED = 1

    @tack.func
    def at(self, p):
        if self.SHIFTED:
            return self.f([p[0] + self.ox, p[1] + self.oy, p[2] + self.oz])
        return self.f(p)


@tack.kernel
def _at_space_nodes(nodes, fn, values):
    # Each of the space's nodes once, where the geometry puts it (node_views).
    for i in nodes:
        for j in range(nodes.node_count(i)):
            values[nodes.node_dof(i, j)] = fn.at(nodes.node_position(i, j))


@tack.kernel
def _corner_keys(cells, start, keys):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            keys[first + j] = cells.point_id(c, j)


@tack.kernel
def _node_keys(cells, t, start, keys):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_QUADRATIC
        for k in range(cells.NUM_QUADRATIC):
            keys[first + k] = t.dof_index(c, k)


@tack.kernel
def _first_of_each(order, offsets, writers, n):
    # The first (cell, node) entry of each node, in entry order, writes it.
    for d in range(n):
        if offsets[d + 1] > offsets[d]:
            writers[order[offsets[d]]] = 1


@tack.kernel
def _at_corners(cells, fn, writers, start, values):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            if writers[first + j] == 1:
                values[cells.point_id(c, j)] = fn.at(cells.position(c, cells.parametric_point(j)))


@tack.kernel
def _at_nodes(cells, t, fn, writers, start, first_node, values):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_QUADRATIC
        for k in range(first_node, cells.NUM_QUADRATIC):
            if writers[first + k] == 1:
                values[t.dof_index(c, k)] = fn.at(cells.position(c, cells.quadratic_node(k)))


def _writers(data, space, target):
    """Which (cell, node) entry writes each of ``space``'s nodes: an i32 flag per
    entry, entries group by group, a cell's nodes together; and where each
    group's start. The first entry of a node writes it, so each is written once.
    Kept on the space: it is addressing, which the geometry does not change."""
    if "_writers" not in space.__dict__:
        topology = data.topology
        groups = topology.groups()
        if space.order == 1:
            starts, total = corner_layout(topology)
        else:
            starts, total = {}, 0
            for group in groups:
                starts[group] = total
                total += group.count * group.shape.NUM_QUADRATIC
        keys = tack.empty((total,), ids.at_least(data.id_dtype, space.size))
        for group in groups:
            if group.count:
                if space.order == 1:
                    _corner_keys(group.view(), starts[group], keys)
                else:
                    _node_keys(group.view(), target.view(group), starts[group], keys)
        writers = tack.zeros((total,), tack.i32)
        if total:
            order, offsets = bucket_order(keys, space.size, ids.at_least(data.id_dtype, total))
            _first_of_each(order, offsets, writers, space.size)
        space.__dict__["_writers"] = (starts, writers)
    return space.__dict__["_writers"]


@on_inputs_device
def interpolate(data, function, space, dtype=None):
    """A ``Field`` in ``space`` whose values are ``function`` at each of the space's
    nodes: ``function`` is a ``@tack.func`` of a world position (a 3-vector of
    the geometry's type) returning a scalar or a vector, and ``dtype`` its
    result's type -- ``data.dtype`` by default, ``tack.vec3()`` for 3-vectors.

    The nodes are where the geometry puts them, evaluated as needed and never
    stored: ``H1``'s points (and at order 2 its edge middles, quad face centers
    and cell centers), ``L2``'s cell corners (order 1) or parametric centers
    (order 0), ``Constant``'s and ``Values(data, "cells")``' parametric
    centers, ``Values(data, "points")``' points. A node shared by several
    cells is evaluated once, by the first cell that has it, so a discontinuous
    (``L2``) geometry gives a shared node the first cell's position. A point no
    cell uses takes ``function`` at its own position when the geometry is
    ``H1``, otherwise zero.

    A function of position is what the space can represent exactly when it is
    a polynomial of the space's order on straight cells; otherwise this is the
    space's nodal interpolant. On polyhedra, polygons and polylines -- no
    reference element -- only point values (``H1`` of order 1 and
    ``Values(data, "points")``) of an ``H1`` geometry interpolate.
    """
    if not isinstance(function, Func):
        raise TypeError(f"interpolate takes a @tack.func of position, not {function!r}")
    if space.topology is not data.topology:
        raise ValueError("interpolate takes a space of the dataset's own topology")
    if isinstance(space, Values) and space.on not in ("points", "cells"):
        raise TypeError(f"interpolate gives values at points or cells, not on {space.on}")
    if not isinstance(space, (H1, L2, Constant, Values)):
        raise TypeError(f"interpolate takes H1, L2, Constant or Values, not {space!r}")
    fn = _Of(function, data.origin)
    values = tack.zeros((space.size,), dtype or data.dtype)
    out = Field(space, values)
    shared = isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")
    if not (shared and not isinstance(data.geometry.space, H1)):
        # Every node once, from the space's node views.
        for_each(_at_space_nodes, data, space, fn, values)
        return out
    # An L2 geometry gives a shared node no position of its own: the first cell
    # that has it decides, through its corners.
    reference_cells(data.topology, f"interpolate into {space!r}")
    if isinstance(space, Values):
        raise ValueError("an L2 geometry has positions per cell corner, not per point")
    starts, writers = _writers(data, space, out)
    for group in data.launch_groups("cells", [out]):
        if not group.count:
            continue
        cells = data.domain_view("cells", group)
        if space.order == 1:
            _at_corners(cells, fn, writers, starts[group.root], values)
        else:
            _at_nodes(cells, out.view(group), fn, writers, starts[group.root], 0, values)
    return out

"""``gradients``: the gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector -- at each cell's parametric center (``at="cells"``) or at the points (``at="points"``: each cell's gradient at the point, averaged over the cells around it, as VTK's vtkGradientFilter and Viskores' point gradients do)."""

import tack
from tack.data import arrays, shapes
from tack.data.algorithms._common import _field, _vectors
from tack.data.algorithms._links import _average_links
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import Field, for_each
from tack.data.spaces import Values, reference_cells
from tack.data.topology import corner_layout

__all__ = ['gradients']


@tack.kernel
def _gradients(cells, u, out):
    for c in cells:
        pc = cells.parametric_center()
        if cells.DIMENSION == 3:
            # d(u)/d(x) = J^-T d(u)/d(pc), J from the geometry, d(u)/d(pc) from u.
            out[cells.entity_id(c)] = (cells.geometry_jacobian(c, pc).inverse().transpose()
                                       @ u.parametric_gradient(c, pc))
        else:
            out[cells.entity_id(c)] = [0.0, 0.0, 0.0]


@tack.kernel
def _corner_gradients(cells, u, start, out):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            if cells.DIMENSION != 3:
                out[first + j] = [0.0, 0.0, 0.0]
            elif cells.ID == shapes.PYRAMID and j == 4:
                # The Jacobian is singular at a pyramid's apex. Extrapolate from
                # two points just below it on the axis, as vtkPyramid does.
                a = tack.Vector([0.5, 0.5, 0.998])
                b = tack.Vector([0.5, 0.5, 0.996])
                ga = (cells.geometry_jacobian(c, a).inverse().transpose()
                      @ u.parametric_gradient(c, a))
                gb = (cells.geometry_jacobian(c, b).inverse().transpose()
                      @ u.parametric_gradient(c, b))
                out[first + j] = 2.0 * ga - gb
            else:
                pc = cells.parametric_point(j)
                out[first + j] = (cells.geometry_jacobian(c, pc).inverse().transpose()
                                  @ u.parametric_gradient(c, pc))


@tack.kernel
def _component(values, k, out):
    for i in range(out.shape[0]):
        out[i] = values[i][k]


@tack.kernel
def _place_rows(rows, k, out):
    for i in range(out.shape[0]):
        g = rows[i]
        out[i][3 * k] = g[0]
        out[i][3 * k + 1] = g[1]
        out[i][3 * k + 2] = g[2]


def _scalar_gradients(data, field, at):
    if at == "cells":
        out = _vectors(data.num_cells, data.dtype)
        for_each(_gradients, data, "cells", field, out)
        return out
    # At the points: each cell's gradient at each of its corners, averaged over
    # the cells around each point, in layout order (VTK's and Viskores' point
    # gradients). Lower-dimensional cells contribute zero.
    start_of, total = corner_layout(data.topology)
    corners = _vectors(total, data.dtype)
    for group in data.launch_groups("cells", [field]):
        if group.count:
            _corner_gradients(data.domain_view("cells", group), field.view(group),
                              start_of[id(group.root)], corners)
    out = tack.Vector.field(3, data.dtype, shape=(data.num_points,))
    out.fill(0)
    if total:
        offsets, entries = data.topology.point_links()
        _average_links(offsets, entries, corners, out)
    return out


def gradients(data, field, at="cells"):
    """The gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector --
    at each cell's parametric center (``at="cells"``) or at the points
    (``at="points"``: each cell's gradient at the point, averaged over the cells
    around it, as VTK's vtkGradientFilter and Viskores' point gradients do).

    A scalar gives a 3-vector per entity; a vector of ``k`` components ``3k``
    values, row by row -- ``d(u_i)/d(x_j)`` at ``3 * i + j``, the layout of
    VTK's and Viskores' gradient tensors. The field's basis gives its
    derivative in parametric coordinates and the geometry's Jacobian turns that
    into a world gradient, so the two need not share a space. Cells below three
    dimensions get zero, and a cell of no volume NaN (its Jacobian is singular). ``flow_quantities`` derives divergence, vorticity and
    the Q-criterion from a vector field's gradient."""
    field = _field(data, field)
    reference_cells(data.topology, "gradients")
    if at not in ("cells", "points"):
        raise ValueError(f"gradients are at 'cells' or 'points', not {at!r}")
    space = Values(data, at)
    width = width_of(field.values)
    if not width:
        return Field(space, _scalar_gradients(data, field, at))
    n = data.num_cells if at == "cells" else data.num_points
    out = tack.Vector.field(3 * width, data.dtype, shape=(n,))
    values = materialize(field.values)
    for k in range(width):
        part = tack.field(arrays.dtype_of(field.values), shape=(size_of(field.values),))
        if part.shape[0]:
            _component(values, k, part)
        rows = _scalar_gradients(data, Field(field.space, part), at)
        if n:
            _place_rows(rows, k, out)
    return Field(space, out)

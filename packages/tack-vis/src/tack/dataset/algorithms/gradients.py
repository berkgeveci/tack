"""``gradients``: the gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector -- at each cell's parametric center (``at="cells"``) or at the points (``at="points"``: each cell's gradient at the point, averaged over the cells around it, as VTK's vtkGradientFilter and Viskores' point gradients do). On polyhedra, which have no reference element, each cell's gradient is Green–Gauss over its faces fanned from their points' means."""

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import shapes
from tack.dataset.algorithms._common import _field, _vectors, _with_basis
from tack.dataset.algorithms._links import _average_links
from tack.dataset.carry import _take
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import corner_layout
from tack.runtime.dispatch import on_inputs_device

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
                a = tack.vec3(0.5, 0.5, 0.998)
                b = tack.vec3(0.5, 0.5, 0.996)
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
def _polyhedral_gradients(cells, u, out):
    for c in cells:
        # Green-Gauss: (1/V) times the sum over the cell's boundary of each triangle's
        # outward area vector times the mean of its three values, the faces walked
        # outward and fanned from their points' means -- in position and in value,
        # so a linear field's gradient is exact. The triangles are cell_geometry's,
        # so V is its volume. Positions are taken from the cell's first point and
        # values from its value there (the area vectors of a closed surface sum to
        # zero), so cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        six = 0.0                                  # six times the volume
        total = tack.vec3(0.0, 0.0, 0.0)           # 2 A_t times 3 mean(u) on each triangle
        for k in range(cells.num_faces(c)):
            size = cells.side_num_points(c, k)
            apex = tack.vec3(0.0, 0.0, 0.0)
            at_apex = 0.0
            for j in range(size):
                apex += cells.side_position(c, k, j) - origin
                at_apex += u[cells.side_point_id(c, k, j)] - u0
            apex = apex / size
            at_apex = at_apex / size
            for j in range(size):
                nxt = j + 1 if j + 1 < size else 0
                a = cells.side_position(c, k, j) - origin
                b = cells.side_position(c, k, nxt) - origin
                six += a.dot(b.cross(apex))
                total += (b - a).cross(apex - a) * (
                    (u[cells.side_point_id(c, k, j)] - u0)
                    + (u[cells.side_point_id(c, k, nxt)] - u0) + at_apex)
        # (1/V) sum A_t mean(u)_t = sum (2 A_t)(3 mean(u)_t) / 6V. No volume: NaN.
        out[cells.entity_id(c)] = total / six


@tack.kernel
def _zero_gradients(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = [0.0, 0.0, 0.0]


@tack.kernel
def _entry_cells(offsets, cells):
    # Which cell each of cell_points()' entries belongs to.
    for c in range(offsets.shape[0] - 1):
        for k in range(offsets[c], offsets[c + 1]):
            cells[k] = c


def _polyhedral_scalar_gradients(data, field, at):
    """A scalar field's gradient on a topology without reference elements: per 3D
    cell Green-Gauss over the fanned faces (exact for a linear field), zero for
    polygons and polylines and for cell data (constant in each cell, as on the
    shape path), and at the points the mean of the cells' around each point."""
    cells = _vectors(data.num_cells, data.dtype)
    on_points = isinstance(field.space, H1)
    values = materialize(field.values) if on_points else None
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        if on_points and group.shape.DIMENSION == 3:
            _polyhedral_gradients(view, values, cells)
        else:
            _zero_gradients(view, cells)
    if at == "cells":
        return cells
    out = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    out.fill(0)
    point_offsets, point_ids = data.topology.cell_points()
    entries = point_ids.shape[0]
    if entries:
        owners = tack.empty((entries,), data.id_dtype)
        _entry_cells(point_offsets, owners)
        offsets, links = data.topology.point_links()
        _average_links(offsets, links, _take(cells, owners), out)
    return out


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
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_scalar_gradients(data, field, at)
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
                              start_of[group.root], corners)
    out = tack.empty((data.num_points,), tack.vec(3, data.dtype))
    out.fill(0)
    if total:
        offsets, entries = data.topology.point_links()
        _average_links(offsets, entries, corners, out)
    return out


@on_inputs_device
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
    the Q-criterion from a vector field's gradient.

    On a polyhedral topology there is no reference element. A cell's gradient
    of point data is then Green-Gauss over its boundary: the faces walked
    outward and fanned from their points' means (the triangles ``cell_geometry``
    measures the volume by), each triangle's area vector times the mean of its
    values, over the volume. That is exact for a linear field and needs no
    interpolant. Cell data, constant in each cell, has a zero gradient there,
    as on shapes; polygons and polylines get zero. At the points, the cells'
    gradients are averaged over the cells around each point."""
    field = _with_basis(data, _field(data, field))
    if (not getattr(data.topology, "reference_cells", True)
            and not isinstance(field.space, (H1, Constant))):
        raise NotImplementedError(f"gradients of a field on {field.space!r} on a mesh "
                                  "without reference elements: point or cell data only")
    if at not in ("cells", "points"):
        raise ValueError(f"gradients are at 'cells' or 'points', not {at!r}")
    space = Values(data, at)
    width = width_of(field.values)
    if not width:
        return Field(space, _scalar_gradients(data, field, at))
    n = data.num_cells if at == "cells" else data.num_points
    out = tack.empty((n,), tack.vec(3 * width, data.dtype))
    values = materialize(field.values)
    for k in range(width):
        part = tack.empty((size_of(field.values),), arrays.dtype_of(field.values))
        if part.shape[0]:
            _component(values, k, part)
        rows = _scalar_gradients(data, Field(field.space, part), at)
        if n:
            _place_rows(rows, k, out)
    return Field(space, out)

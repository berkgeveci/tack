"""``gradients``: the gradient of ``field`` -- ``H1``, ``L2`` or ``Constant``, scalar or vector -- at each cell's parametric center (``at="cells"``) or at the points (``at="points"``: each cell's gradient at the point, averaged over the cells around it, as VTK's vtkGradientFilter and Viskores' point gradients do). On polyhedra, which have no reference element, each cell's gradient is Green–Gauss over its faces fanned from their points' means."""

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import shapes, units
from tack.dataset.algorithms._common import _field, _vectors, _with_basis
from tack.dataset.algorithms._links import _average_links
from tack.dataset.carry import _take
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.dataset.topology import corner_layout
from tack.runtime.dispatch import on_inputs_device

__all__ = ['gradients', 'world_gradient']


@tack.func
def world_gradient(cells, u, c, pc):
    """`u`'s gradient in the world at parametric `pc` of cell `c`: its
    parametric gradient through the geometry's Jacobian. For kernels that need a
    field's gradient where `gradients` does not put it."""
    # d(u)/d(x) from d(u)/d(pc), through the geometry's Jacobian J. A solid's is
    # J^-T d(u)/d(pc). A 2D cell's lies in its tangent plane, spanned by J's
    # columns a and b: g = alpha a + beta b with g . a and g . b the parametric
    # derivatives, VTK's and Viskores' in-plane gradient (no area: NaN, as a
    # solid of no volume). Lines and vertices: zero.
    g = tack.vec3(0.0, 0.0, 0.0)
    if cells.DIMENSION == 3:
        g = cells.geometry_jacobian(c, pc).inverse().transpose() @ u.parametric_gradient(c, pc)
    elif cells.DIMENSION == 2:
        m = cells.geometry_jacobian(c, pc)
        a = tack.vec3(m[0, 0], m[1, 0], m[2, 0])
        b = tack.vec3(m[0, 1], m[1, 1], m[2, 1])
        d = u.parametric_gradient(c, pc)
        aa = a.dot(a)
        ab = a.dot(b)
        bb = b.dot(b)
        det = aa * bb - ab * ab
        g = ((bb * d[0] - ab * d[1]) * a + (aa * d[1] - ab * d[0]) * b) / det
    return g


_world_gradient = world_gradient


@tack.func
def world_hessian(cells, u, c, pc):
    """`u`'s second derivatives in the world at parametric `pc` of solid cell `c`, a
    symmetric 3x3 matrix: by the chain rule through the geometry, the parametric
    Hessian less the geometry's curvature along the world gradient, taken to world
    axes on both sides by the inverse Jacobian. Analytic from the basis on every
    shape's order 1 and 2 and on tensor shapes and simplices of any order;
    fourth-order differences of the parametric gradient on wedges above order 2.
    Zero on surfaces, lines and vertices."""
    _, _, h = world_jet(cells, u, c, pc)
    return h


@tack.func
def world_jet(cells, u, c, pc):
    """Scalar `u`'s value, world gradient and world Hessian at parametric `pc` of solid
    cell `c`, from the field view's `parametric_jet`, one pass over the cell's values:
    what a kernel needing several of them reads once. As `world_hessian`; on surfaces,
    lines and vertices the gradient and Hessian are zero."""
    v, pg, ph = u.parametric_jet(c, pc)
    g = tack.vec3(0.0, 0.0, 0.0)
    h = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    if cells.DIMENSION == 3:
        inverse = cells.geometry_jacobian(c, pc).inverse()
        g = inverse.transpose() @ pg
        h = inverse.transpose() @ (ph - cells.geometry_curvature(c, pc, g)) @ inverse
    return v, g, h


@tack.kernel
def _gradients(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = _world_gradient(cells, u, c, cells.parametric_center())


@tack.kernel
def _node_gradients(cells, u, out):
    # The gradient at each of the cell's nodes of the target space.
    for c in cells:
        for j in range(cells.node_count(c)):
            out[cells.node_dof(c, j)] = _world_gradient(cells, u, c, cells.node_parametric_point(c, j))


@tack.kernel
def _corner_gradients(cells, u, start, out):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            if cells.ID == shapes.PYRAMID and j == 4:
                # The Jacobian is singular at a pyramid's apex. Extrapolate from
                # two points just below it on the axis, as vtkPyramid does.
                ga = _world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.998))
                gb = _world_gradient(cells, u, c, tack.vec3(0.5, 0.5, 0.996))
                out[first + j] = 2.0 * ga - gb
            else:
                out[first + j] = _world_gradient(cells, u, c, cells.parametric_point(j))


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
            # One walk round the face. With F its points' mean, triangle j's
            # 2 A_j = p_j x p_j+1 + (p_j+1 - p_j) x F, so the face's share is
            # S + W x F + u_F N, with S = sum (p_j x p_j+1)(u_j + u_j+1),
            # W = sum (p_j+1 - p_j)(u_j + u_j+1) and N = sum p_j x p_j+1 (the
            # Newell vector, sum 2 A_j); its share of 6V is F . N. F and u_F are
            # known only at the end, and none of the sums needs them.
            size = cells.side_num_points(c, k)
            first = cells.side_position(c, k, 0) - origin
            u_first = u[cells.side_point_id(c, k, 0)] - u0
            a = first
            ua = u_first
            points = first
            values = u_first
            newell = tack.vec3(0.0, 0.0, 0.0)
            spun = tack.vec3(0.0, 0.0, 0.0)
            along = tack.vec3(0.0, 0.0, 0.0)
            for j in range(1, size + 1):
                b = first
                ub = u_first
                if j < size:
                    b = cells.side_position(c, k, j) - origin
                    ub = u[cells.side_point_id(c, k, j)] - u0
                    points += b
                    values += ub
                n = a.cross(b)
                both = ua + ub
                newell += n
                spun += n * both
                along += (b - a) * both
                a = b
                ua = ub
            apex = points / size
            six += apex.dot(newell)
            total += spun + along.cross(apex) + newell * (values / size)
        # (1/V) sum A_t mean(u)_t = sum (2 A_t)(3 mean(u)_t) / 6V. No volume: NaN.
        out[cells.entity_id(c)] = total / six


@tack.kernel
def _polygon_gradients(cells, u, out):
    for c in cells:
        # In-plane Green-Gauss: (1/A) times the sum over the edges of each edge's
        # outward in-plane normal (its length long) times its mean value, with N
        # the Newell vector (2A along the normal): sum (u_k + u_k+1) ((p_k+1 - p_k)
        # x N) / |N|^2. Exact for a linear field on a planar polygon. Positions
        # and values from the first point, as the polyhedral gradient takes them.
        origin = cells.side_position(c, 0, 0)
        u0 = u[cells.side_point_id(c, 0, 0)]
        newell = tack.vec3(0.0, 0.0, 0.0)
        total = tack.vec3(0.0, 0.0, 0.0)
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            newell += a.cross(b)
            total += (u[cells.side_point_id(c, k, 0)] + u[cells.side_point_id(c, k, 1)]
                      - 2.0 * u0) * (b - a)
        out[cells.entity_id(c)] = total.cross(newell) / newell.dot(newell)


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
    cell Green-Gauss over the fanned faces, per polygon in its plane over its
    edges (both exact for a linear field), zero for polylines and for cell data
    (constant in each cell, as on the shape path), and at the points the mean of
    the cells' around each point."""
    cells = _vectors(data.num_cells, data.dtype)
    on_points = isinstance(field.space, H1)
    values = materialize(field.values) if on_points else None
    for group in data.launch_groups("cells", []):
        if not group.count:
            continue
        view = data.domain_view("cells", group)
        if on_points and group.shape.DIMENSION == 3:
            _polyhedral_gradients(view, values, cells)
        elif on_points and group.shape.DIMENSION == 2:
            _polygon_gradients(view, values, cells)
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
    if isinstance(at, L2):
        # At an L2 space's nodes: each cell's own, from the cell alone.
        out = _vectors(at.size, data.dtype)
        groups = data.launch_groups("cells", [field], keys=(at,) if at.varies else ())
        for group, view in data._cell_node_views(at, groups):
            _node_gradients(view, field.view(group), out)
        return out
    if not getattr(data.topology, "reference_cells", True):
        return _polyhedral_scalar_gradients(data, field, at)
    if at == "cells":
        out = _vectors(data.num_cells, data.dtype)
        for_each(_gradients, data, "cells", field, out)
        return out
    # At the points: each cell's gradient at each of its corners, averaged over
    # the cells around each point, in layout order (VTK's and Viskores' point
    # gradients). Lines and vertices contribute zero.
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
    VTK's and Viskores' gradient tensors.

    ``at`` may instead be an ``L2`` space of the dataset: the gradient at each of
    its nodes in each cell -- a discontinuous field, exact at the nodes, so an
    ``H1`` field of order ``p`` keeps its order's detail in ``L2`` of order ``p``,
    where a gradient at the points averages it away. The field's basis gives its
    derivative in parametric coordinates and the geometry's Jacobian turns that
    into a world gradient, so the two need not share a space. A 2D cell's
    gradient lies in its plane (VTK's and Viskores' in-plane gradient), lines
    and vertices get zero, and a cell of no volume or area NaN (its Jacobian is
    singular). ``flow_quantities`` derives divergence, vorticity and
    the Q-criterion from a vector field's gradient.

    On a polyhedral topology there is no reference element. A cell's gradient
    of point data is then Green-Gauss over its boundary: the faces walked
    outward and fanned from their points' means (the triangles ``cell_geometry``
    measures the volume by), each triangle's area vector times the mean of its
    values, over the volume. That is exact for a linear field and needs no
    interpolant. Cell data, constant in each cell, has a zero gradient there,
    as on shapes. A polygon's is the same in its plane, over its edges; polylines
    get zero. At the points, the cells' gradients are averaged over the cells
    around each point."""
    field = _with_basis(data, _field(data, field))
    if (not getattr(data.topology, "reference_cells", True)
            and not isinstance(field.space, (H1, Constant))):
        raise NotImplementedError(f"gradients of a field on {field.space!r} on a mesh "
                                  "without reference elements: point or cell data only")
    if isinstance(at, L2):
        if at.topology is not data.topology:
            raise ValueError("gradients at the nodes of a space of another topology")
        if not getattr(data.topology, "reference_cells", True):
            raise NotImplementedError("gradients at an L2 space's nodes take cells with a "
                                      "reference element")
        space = at
    elif at not in ("cells", "points"):
        raise ValueError(f"gradients are at 'cells' or 'points', or an L2 space's nodes, "
                         f"not {at!r}")
    else:
        space = Values(data, at)
    # Per unit length; a difference of degrees Celsius is in kelvin.
    unit = units.derived(units.delta(field.unit), (data.unit, -1))
    width = width_of(field.values)
    if not width:
        return Field(space, _scalar_gradients(data, field, at), unit)
    n = space.size if isinstance(at, L2) else data.num_cells if at == "cells" \
        else data.num_points
    out = tack.empty((n,), tack.vec(3 * width, data.dtype))
    values = materialize(field.values)
    for k in range(width):
        part = tack.empty((size_of(field.values),), arrays.dtype_of(field.values))
        if part.shape[0]:
            _component(values, k, part)
        rows = _scalar_gradients(data, Field(field.space, part), at)
        if n:
            _place_rows(rows, k, out)
    return Field(space, out, unit)

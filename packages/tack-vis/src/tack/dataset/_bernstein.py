"""Bernstein spaces' coefficients from values at nodes, and back.

A Bernstein space's values are coefficients, not values at nodes, so what fills a
space at its nodes (``interpolate``, ``project``) fills the equispaced Lagrange
space of the same order -- the same layout, value for value -- and converts: each
coefficient is the 1D Lagrange-to-Bernstein matrix (``bases.bernstein``) applied
along each axis. A corner's coefficient is its value. An edge's or face's depends
on that entity's values alone (the trace of a polynomial on it), so it is formed
from them in a frame their corners' positions fix (``subdivide._edge_step``,
``_face_frame``), the same in every cell and every piece holding the entity, whose
writes then agree to the bit; the inside of a cell, and an ``L2`` cell's values,
in the cell's own frame.
"""

import numpy as np

import tack
from tack import arrays
from tack.dataset import bases
from tack.dataset.algorithms.subdivide import _edge_step, _face_frame, _unit
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, L2


def lagrange_companion(space):
    """The equispaced Lagrange space with ``space``'s layout: its order, and for
    ``L2`` its layout."""
    if isinstance(space, H1):
        return H1(space.topology, order=space.order, basis="equispaced")
    return L2(space.topology, order=space.order, basis="equispaced",
              layout=space.params.get("layout", "auto"))


def is_bernstein(space):
    return isinstance(space, (H1, L2)) and space.basis == bases.BERNSTEIN


@tack.func
def _cell_coefficient(w, c, matrix, q, n):
    # Node n's coefficient from the whole cell's values, in the cell's frame: the
    # matrix applied along each axis (its end rows pick the end values).
    i = n % q
    total = w.dof(c, 0) - w.dof(c, 0)
    if w.DIMENSION == 1:
        for a in range(q):
            total += matrix[i * q + a] * w.dof(c, a)
    elif w.DIMENSION == 2:
        j = n // q
        for b in range(q):
            row = w.dof(c, 0) - w.dof(c, 0)
            for a in range(q):
                row += matrix[i * q + a] * w.dof(c, a + q * b)
            total += matrix[j * q + b] * row
    else:
        j = (n // q) % q
        k = n // (q * q)
        for cc in range(q):
            plane = w.dof(c, 0) - w.dof(c, 0)
            for b in range(q):
                row = w.dof(c, 0) - w.dof(c, 0)
                for a in range(q):
                    row += matrix[i * q + a] * w.dof(c, a + q * (b + q * cc))
                plane += matrix[j * q + b] * row
            total += matrix[k * q + cc] * plane
    return total


@tack.kernel
def _l2_coefficients(cells, w, matrix, q, out):
    # Each cell's values to its coefficients, in its own frame.
    for c in cells:
        for n in range(q ** w.DIMENSION):
            out[w.dof_index(c, n)] = _cell_coefficient(w, c, matrix, q, n)


@tack.kernel
def _h1_coefficients(cells, w, matrix, q, corner_node, node_kind, edge_nodes, face_nodes,
                     stride, out):
    # Corners copied; each edge's and face's coefficients from its own values in the
    # frame its corners' positions fix (every cell holding it writes the same bits);
    # the inside's in the cell's frame.
    for c in cells:
        p = q - 1
        for j in range(cells.NUM_POINTS):
            n = corner_node[j]
            out[w.dof_index(c, n)] = w.dof(c, n)
        if w.DIMENSION >= 2:
            for m in range(cells.NUM_EDGES):
                _, flip = _edge_step(cells, c, m, 0, p)
                for i in range(1, p):
                    total = w.dof(c, 0) - w.dof(c, 0)
                    for a in range(q):
                        total += matrix[i * q + a] * \
                            w.dof(c, edge_nodes[m * q + (p - a if flip == 1 else a)])
                    out[w.dof_index(c, edge_nodes[m * q + (p - i if flip == 1 else i)])] = total
        if w.DIMENSION == 3:
            for m in range(cells.NUM_FACES):
                ka, kb, kd = _face_frame(cells, c, m)
                ax, ay = _unit(ka)
                bx, by = _unit(kb)
                dx, dy = _unit(kd)
                for i in range(1, p):
                    for j in range(1, p):
                        total = w.dof(c, 0) - w.dof(c, 0)
                        for b in range(q):
                            row = w.dof(c, 0) - w.dof(c, 0)
                            for a in range(q):
                                fa = ax * p + a * (bx - ax) + b * (dx - ax)
                                fb = ay * p + a * (by - ay) + b * (dy - ay)
                                row += matrix[i * q + a] * \
                                    w.dof(c, face_nodes[m * stride + fa + q * fb])
                            total += matrix[j * q + b] * row
                        fa = ax * p + i * (bx - ax) + j * (dx - ax)
                        fb = ay * p + i * (by - ay) + j * (dy - ay)
                        out[w.dof_index(c, face_nodes[m * stride + fa + q * fb])] = total
        for n in range(q ** w.DIMENSION):
            if node_kind[n] == 3:
                out[w.dof_index(c, n)] = _cell_coefficient(w, c, matrix, q, n)


def from_lagrange(data, field, space):
    """``field``, in ``lagrange_companion(space)``, as ``space``'s coefficients."""
    return Field(space, _converted(data, field.values, space,
                                   bases.bernstein("equispaced", space.order)), field.unit)


def to_lagrange(data, field):
    """``field``, of a Bernstein space, as its values at the equispaced nodes: a field
    of ``lagrange_companion(field.space)``, the same polynomials."""
    from math import comb

    p = field.space.order
    x = bases.nodes("equispaced", p)
    at = np.array([[comb(p, j) * t ** j * (1.0 - t) ** (p - j) for j in range(p + 1)]
                   for t in x])
    return Field(lagrange_companion(field.space),
                 _converted(data, field.values, field.space, at), field.unit)


def with_lagrange_geometry(data):
    """``data``, or a stand-in on its topology whose geometry is a curved Bernstein
    one's values at the equispaced nodes, the same cells: where a filter places
    nodes by the geometry's values there (``interpolate``), control points are no
    positions."""
    if not (is_bernstein(data.geometry.space) and data.geometry.space.order > 1):
        return data
    from tack.dataset.dataset import DataSet

    return DataSet(data.topology, to_lagrange(data, data.geometry), **data._frame())


def _converted(data, values, space, matrix):
    """``values`` of ``space``'s layout (read through ``space``'s view) with ``matrix``
    applied along each axis, in each entity's frame."""
    from tack.dataset.algorithms.subdivide import _entity_tables, _face_stride, _typed

    values = arrays.materialize(values)
    staged = Field(space, values)                  # the values, read through space's view
    out = tack.empty_like(values)
    order = space.order
    q = order + 1
    like = tack.empty((1,), arrays.dtype_of(values))
    matrix = _typed(np.ascontiguousarray(matrix).reshape(-1), like)
    for group in data.launch_groups("cells", [staged]):
        if not group.count:
            continue
        cells, w = data.domain_view("cells", group), staged.view(group)
        if isinstance(space, L2):
            _l2_coefficients(cells, w, matrix, q, out)
            continue
        kind, _, _, _, corner = bases.dof_tables(group.shape, order)
        edge_nodes, face_nodes = _entity_tables(group.shape, order)
        _h1_coefficients(cells, w, matrix, q, tack.array(np.asarray(corner, np.int32)),
                         tack.array(np.asarray(kind, np.int32)), edge_nodes, face_nodes,
                         _face_stride(group.shape, order), out)
    return out

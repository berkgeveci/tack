"""Lagrange bases of any order on the tensor shapes: segments, quads, pixels,
hexahedra and voxels.

A basis is a shape, an order ``p`` and a family of 1D nodes on ``[0, 1]`` (the
shapes' parametric range):

- ``"gauss_lobatto"``: the ends and the roots of the derivative of the Legendre
  polynomial of degree ``p`` -- MFEM's default for ``H1``, Nek5000's and
  deal.II's; better conditioned than equispaced nodes at high order;
- ``"equispaced"``: ``j / p`` -- VTK's Lagrange cells, Exodus, gmsh;
- ``"gauss_legendre"``: the roots of the Legendre polynomial of degree ``p + 1``,
  inside the segment -- MFEM's default for ``L2``; for ``L2`` only, since
  shared values need nodes at the ends.

At order 2 or less the two closed families are the same nodes. A cell's
nodes are the products of the 1D nodes along its axes, numbered
lexicographically, the first axis fastest: node ``n`` is ``(i, j, k)`` with
``n = i + (p + 1) * (j + (p + 1) * k)``, at parametric coordinates
``(x[i], x[j], x[k])``. Its basis function is the product of the 1D Lagrange
polynomials, evaluated in barycentric form from the nodes and their weights
(``weights``), which kernels read as small arrays in the field's own float type.

``H1`` shares values: a node at a corner is the point's, one inside an edge,
quad face or cell belongs to that entity (``dof_tables`` classifies each node,
and the field views orient edges from their lower point id and faces in their
own row's frame). ``L2`` keeps each cell's ``(p + 1)^d`` values in node order.
"""

import functools

import numpy as np

from tack.dataset import shapes

__all__ = ["FAMILIES", "dof_tables", "nodes", "tensor_shape", "weights"]

FAMILIES = ("gauss_lobatto", "equispaced", "gauss_legendre")
#: The families with nodes at the ends, which shared values need.
CLOSED = ("gauss_lobatto", "equispaced")

#: Node kinds in ``dof_tables``.
CORNER, EDGE, FACE, INTERIOR = 0, 1, 2, 3


def _legendre_roots(polynomial, guesses):
    """``guesses`` polished by Newton's method on ``polynomial``, a NumPy Legendre series."""
    x = np.asarray(guesses, np.float64)
    derivative = polynomial.deriv()
    for _ in range(100):
        step = polynomial(x) / derivative(x)
        x = x - step
        if np.max(np.abs(step), initial=0.0) < 1e-16:
            break
    return x


@functools.cache
def nodes(family, order):
    """The ``order + 1`` nodes of ``family`` on ``[0, 1]``, increasing, as an f64 array,
    symmetric about ``0.5`` to the last bit."""
    if family not in FAMILIES:
        raise ValueError(f"nodes are one of {', '.join(FAMILIES)}, not {family!r}")
    p = int(order)
    if p < 0:
        raise ValueError(f"an order is at least 0, not {order}")
    if p == 0:
        x = np.array([0.0])                     # one node, at the middle
    elif family == "equispaced":
        x = np.linspace(-1.0, 1.0, p + 1)
    elif family == "gauss_lobatto":
        legendre = np.polynomial.legendre.Legendre.basis(p)
        inner = np.cos(np.pi * np.arange(p - 1, 0, -1) / p)       # Chebyshev-Lobatto guesses
        x = np.concatenate([[-1.0], _legendre_roots(legendre.deriv(), inner), [1.0]])
    else:
        legendre = np.polynomial.legendre.Legendre.basis(p + 1)
        x = _legendre_roots(legendre, np.polynomial.legendre.leggauss(p + 1)[0])
    x = 0.5 * (np.sort(x) + 1.0)
    half = (p + 1) // 2
    x[p - np.arange(half)] = 1.0 - x[:half]     # the upper half mirrors the lower
    if p % 2 == 0:
        x[p // 2] = 0.5
    x.setflags(write=False)
    return x


@functools.cache
def weights(family, order):
    """The barycentric weights of ``nodes(family, order)``: ``1 / prod(x_j - x_m)``
    over ``m != j``, so the Lagrange polynomial of node ``j`` at ``t`` is
    ``weights[j] * prod(t - x_m)``."""
    x = nodes(family, order)
    difference = x[:, None] - x[None, :]
    np.fill_diagonal(difference, 1.0)
    w = 1.0 / np.prod(difference, axis=1)
    w.setflags(write=False)
    return w


# ── Shapes: their corners, edges and faces on the host ──────────────

#: Each tensor shape's corners as integer parametric coordinates (0 or 1), in
#: its own point order, and its edge and face tables (``shapes``' constants).
_SHAPES = {
    shapes.Line: ([(0,), (1,)], (), ()),
    shapes.Quad: ([(0, 0), (1, 0), (1, 1), (0, 1)], shapes._QUAD_EDGES, ()),
    shapes.Pixel: ([(0, 0), (1, 0), (0, 1), (1, 1)], shapes._PIXEL_EDGES, ()),
    shapes.Hexahedron: ([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                        shapes._HEXAHEDRON_EDGES, shapes._HEXAHEDRON_FACES),
    shapes.Voxel: ([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                   shapes._VOXEL_EDGES, shapes._VOXEL_FACES),
}


def tensor_shape(shape):
    """Whether ``shape`` has bases of any order here."""
    return shape in _SHAPES


def _pairs(table, k):
    flat = [int(v) for v in np.asarray(table).reshape(-1)]
    return [tuple(flat[i:i + k]) for i in range(0, len(flat), k)]


@functools.cache
def dof_tables(shape, order):
    """For each of a ``shape`` cell's ``(order + 1)^d`` nodes, in node order, where its
    value lives: four i32 arrays ``(kind, entity, a, b)``.

    - ``CORNER``: ``entity`` is the cell's corner (its point).
    - ``EDGE``: ``entity`` is the cell's local edge and ``a`` (1 to ``order - 1``)
      the node's place along it from the edge's corner 0.
    - ``FACE``: ``entity`` is the cell's local face, ``(a, b)`` the node's place
      from the face's corner 0, ``a`` toward its corner 1 and ``b`` toward its
      corner 3.
    - ``INTERIOR``: ``a`` is the node's place among the cell's interior nodes,
      lexicographic, the first axis fastest.

    Also ``corner_node``: corner ``j``'s node, an i32 array of ``NUM_POINTS``."""
    if shape not in _SHAPES:
        raise NotImplementedError(f"bases of any order are on segments, quads, pixels, "
                                  f"hexahedra and voxels, not {shape.__name__}")
    corners, edge_table, face_table = _SHAPES[shape]
    p = int(order)
    d = len(corners[0])
    edges = _pairs(edge_table, 2) if len(edge_table) else []
    faces = _pairs(face_table, 4) if len(face_table) else []
    q = p + 1
    count = q ** d
    kind = np.zeros(count, np.int32)
    entity = np.zeros(count, np.int32)
    a = np.zeros(count, np.int32)
    b = np.zeros(count, np.int32)
    corner_node = np.zeros(len(corners), np.int32)

    def place(index, start, toward):
        """The node's steps from corner ``start`` toward corner ``toward``."""
        axis = next(i for i in range(d) if corners[start][i] != corners[toward][i])
        return index[axis] if corners[start][axis] == 0 else p - index[axis]

    for n in range(count):
        index = [(n // q ** axis) % q for axis in range(d)]
        ends = [i in (0, p) for i in index]
        if all(ends):
            at = tuple(i // p for i in index) if p else tuple(0 for _ in index)
            kind[n], entity[n] = CORNER, corners.index(at)
            corner_node[corners.index(at)] = n
        elif not any(ends):
            interior = [i - 1 for i in index]
            kind[n] = INTERIOR
            a[n] = sum(interior[axis] * (p - 1) ** axis for axis in range(d))
        elif d - sum(ends) == 1:
            # On an edge: the corners it runs between agree with the node's ends.
            free = ends.index(False)
            for e, (c0, c1) in enumerate(edges):
                if all(corners[c0][i] * p == index[i] for i in range(d) if i != free) \
                        and corners[c0][free] != corners[c1][free]:
                    kind[n], entity[n], a[n] = EDGE, e, place(index, c0, c1)
                    break
            else:
                raise AssertionError(f"node {n} of {shape.__name__} is on no edge")
        else:
            # On a face of a solid: the face whose corners share the node's end.
            fixed = ends.index(True)
            for f, row in enumerate(faces):
                if all(corners[c][fixed] * p == index[fixed] for c in row):
                    kind[n], entity[n] = FACE, f
                    a[n], b[n] = place(index, row[0], row[1]), place(index, row[0], row[3])
                    break
            else:
                raise AssertionError(f"node {n} of {shape.__name__} is on no face")
    for array in (kind, entity, a, b, corner_node):
        array.setflags(write=False)
    return kind, entity, a, b, corner_node

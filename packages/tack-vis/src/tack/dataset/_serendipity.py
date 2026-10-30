"""Serendipity fields read as the full order-2 space.

``H1(order=2, basis="serendipity")`` stores the points' values, then one value per
edge -- exactly the first values of ``H1(order=2)``, numbered alike -- and none for
quad faces' centres or cells' insides, which the serendipity functions determine
from the corners and edges. A view of a serendipity field reads ``full_values``: the
stored values followed by those, so every kernel sees an ordinary order-2 field (the
serendipity space lies in the full one, so the full basis at these values is the
serendipity function). Writes through a view's ``dof_index`` reach the stored values,
whose indices are the same in both spaces.

A quad face's centre is half its edges' values less a quarter of its corners',
summed smallest first so every cell sharing the face, in every piece, writes the
same bits; a quad's inside the same over its own edges and corners, a hexahedron's
a quarter of its 12 edges less a quarter of its 8 corners.
"""

import tack
from tack import arrays


@tack.func
def _sorted_sum4(a, b, c, d):
    # a + b + c + d summed smallest first (componentwise for vectors): the same bits
    # in whatever order a cell lists them.
    lo1 = min(a, b)
    hi1 = max(a, b)
    lo2 = min(c, d)
    hi2 = max(c, d)
    m1 = max(lo1, lo2)
    m2 = min(hi1, hi2)
    return ((min(lo1, lo2) + min(m1, m2)) + max(m1, m2)) + max(hi1, hi2)


@tack.func
def _edge_between(cells, a, b):
    # The cell's local edge joining its corners a and b.
    found = 0
    for e in range(cells.NUM_EDGES):
        p = cells.edge_corner(e, 0)
        q = cells.edge_corner(e, 1)
        if (p == a and q == b) or (p == b and q == a):
            found = e
    return found


@tack.kernel
def _derived(cells, w, out):
    # Each quad face's centre and each quad's or hexahedron's inside, from the
    # stored corners and edges already in out (w: the full order-2 space's view).
    for c in cells:
        points = cells.NUM_POINTS
        edges = cells.NUM_EDGES
        for i in range(cells.QUADRATIC_FACES):
            f = cells.NUM_FACES - cells.QUADRATIC_FACES + i
            k0 = cells.face_corner(f, 0)
            k1 = cells.face_corner(f, 1)
            k2 = cells.face_corner(f, 2)
            k3 = cells.face_corner(f, 3)
            corners = _sorted_sum4(out[w.dof_index(c, k0)], out[w.dof_index(c, k1)],
                                   out[w.dof_index(c, k2)], out[w.dof_index(c, k3)])
            middles = _sorted_sum4(
                out[w.dof_index(c, points + _edge_between(cells, k0, k1))],
                out[w.dof_index(c, points + _edge_between(cells, k1, k2))],
                out[w.dof_index(c, points + _edge_between(cells, k2, k3))],
                out[w.dof_index(c, points + _edge_between(cells, k3, k0))])
            out[w.dof_index(c, points + edges + i)] = 0.5 * middles - 0.25 * corners
        if cells.QUADRATIC_INTERIOR == 1:
            weight = 0.5 if cells.DIMENSION == 2 else 0.25
            total = out[w.dof_index(c, 0)] * -0.25
            for j in range(1, points):
                total += out[w.dof_index(c, j)] * -0.25
            for e in range(edges):
                total += out[w.dof_index(c, points + e)] * weight
            out[w.dof_index(c, points + edges + cells.QUADRATIC_FACES)] = total


def full_values(space, values):
    """A serendipity field's ``values`` as the full order-2 space's: the stored ones,
    then each quad face's centre and each cell's inside. Kept on the space for the
    last values asked for, until Tack writes them."""
    from tack.dataset.dataset import Field
    from tack.dataset.spaces import H1

    token = arrays._revision(values)
    kept = space.__dict__.get("_full_values")
    if kept is not None and kept[0] is values and kept[1] == token:
        return kept[2]
    full = H1(space.topology, order=2)
    held = arrays.materialize(values)
    width = arrays.width_of(held)
    dtype = arrays.dtype_of(held)
    out = tack.empty((full.size,), tack.vec(width, dtype) if width else dtype,
                     device=space.topology.device)
    stored = arrays.size_of(held)
    if stored:
        _copy_prefix(held, out)
    target = Field(full, out)
    for group in space.topology.groups():
        if group.count and (group.shape.QUADRATIC_FACES or group.shape.QUADRATIC_INTERIOR):
            _derived(group.view(), target.view(group), out)
    space.__dict__["_full_values"] = (values, token, out)
    return out


@tack.kernel
def _copy_prefix(values, out):
    for i in range(values.shape[0]):
        out[i] = values[i]

"""``glyphs``: an arrow at each point along a vector field, VTK's vtkGlyph3D with its
arrow source."""

import math

import tack
from tack import arrays
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms.tubes import _across, _on_points
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['glyphs']

#: An arrow of length 1: the shaft's radius and length, the head's radius.
SHAFT_RADIUS = tack.constant(0.03)
SHAFT_LENGTH = tack.constant(0.65)
HEAD_RADIUS = tack.constant(0.1)


@tack.kernel
def _arrows(positions, vectors, scale, by, scales, res, step, out, source, connectivity):
    # Point i's arrow: a shaft (two rings), the head's base ring, its tip and the
    # shaft's base centre, 3 res + 2 points; as triangles, the shaft's sides, the
    # head's base, its cone and the shaft's base, 6 res, wound outward.
    for i in range(positions.shape[0]):
        v = vectors[i]
        length = v.norm()
        t = tack.vec3(0.0, 0.0, 1.0)
        if length > 0.0:
            t = v / length
        size = length * scale
        if by == 1:
            size = scales[i] * scale
        n = _across(t)
        b = t.cross(n)
        p = positions[i]
        first = i * (3 * res + 2)
        for s in range(res):
            angle = step * s
            d = tack.cos(angle) * n + tack.sin(angle) * b
            out[first + s] = p + size * SHAFT_RADIUS * d
            out[first + res + s] = p + size * (SHAFT_LENGTH * t + SHAFT_RADIUS * d)
            out[first + 2 * res + s] = p + size * (SHAFT_LENGTH * t + HEAD_RADIUS * d)
        out[first + 3 * res] = p + size * t
        out[first + 3 * res + 1] = p
        for k in range(3 * res + 2):
            source[first + k] = i
        tip = first + 3 * res
        centre = tip + 1
        tri = i * 6 * res
        for s in range(res):
            nxt = (s + 1) % res
            c = 3 * (tri + 6 * s)
            connectivity[c] = first + s                       # shaft side
            connectivity[c + 1] = first + nxt
            connectivity[c + 2] = first + res + nxt
            connectivity[c + 3] = first + s
            connectivity[c + 4] = first + res + nxt
            connectivity[c + 5] = first + res + s
            connectivity[c + 6] = first + res + s             # the head's base, facing back
            connectivity[c + 7] = first + res + nxt
            connectivity[c + 8] = first + 2 * res + nxt
            connectivity[c + 9] = first + res + s
            connectivity[c + 10] = first + 2 * res + nxt
            connectivity[c + 11] = first + 2 * res + s
            connectivity[c + 12] = first + 2 * res + s        # cone
            connectivity[c + 13] = first + 2 * res + nxt
            connectivity[c + 14] = tip
            connectivity[c + 15] = first + nxt                # the shaft's base
            connectivity[c + 16] = first + s
            connectivity[c + 17] = centre


@on_inputs_device
def glyphs(data, vectors, scale=1.0, scale_by=None, resolution=8, fields=None):
    """An arrow at each of ``data``'s points along ``vectors`` (a 3-vector field on the
    points, or its name), as VTK's ``vtkGlyph3D`` places its arrow source: from the
    point toward the vector, ``scale`` times its length long, or ``scale`` times
    ``scale_by`` (a scalar point field, or its name) where given. An arrow of
    length 1 has a shaft 0.65 long and 0.03 in radius and a head 0.1 in radius,
    ``resolution`` points round, as triangles wound outward. Its points carry
    the point's fields; cell fields are left behind. A zero vector gives an
    arrow of no size. Glyph a subset (``mask_points``) or a slice to see
    anything in a dense field."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("glyphs places arrows at the points: an H1 order-1 geometry")
    res = int(resolution)
    if res < 3:
        raise ValueError(f"an arrow has a resolution of at least 3, not {res}")
    field = _field(data, vectors)
    if arrays.width_of(field.values) != 3 or not _on_points(field):
        raise TypeError("glyphs takes a 3-vector field on the points")
    sizes = None
    if scale_by is not None:
        sizes = _field(data, scale_by)
        if arrays.width_of(sizes.values) or not _on_points(sizes):
            raise TypeError("scale_by is a scalar field on the points")
    n = data.num_points
    total = n * (3 * res + 2)
    idt = id_types.at_least(data.id_dtype, total)
    out = tack.empty((total,), tack.vec(3, data.dtype))
    source = tack.empty((total,), data.id_dtype)
    connectivity = tack.empty((n * 18 * res,), idt)
    if n:
        by = sizes.values if sizes is not None else tack.empty((1,), data.dtype)
        _arrows(arrays.materialize(data.geometry.values), arrays.materialize(field.values),
                float(scale), int(sizes is not None), arrays.materialize(by), res,
                2.0 * math.pi / res, out, source, connectivity)
    types, offsets = uniform_cells(int(shapes.TRIANGLE), n * 6 * res, idt)
    topology = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    return carry(data, topology, points=Take(source),
                 geometry=Field(H1(topology), out, data.geometry.unit), fields=fields)

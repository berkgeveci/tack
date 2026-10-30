"""``tubes``: lines as tubes of quads round them, VTK's vtkTubeFilter."""

import math

import tack
from tack import arrays
from tack.arrays import size_of
from tack.dataset import ids as id_types
from tack.dataset import shapes
from tack.dataset.algorithms._common import _field, _length
from tack.dataset.carry import Take, carry
from tack.dataset.dataset import Field
from tack.dataset.polyhedra import PolylineTopology
from tack.dataset.spaces import H1, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['tubes']


@tack.func
def _unit_or(v, fallback):
    length = v.norm()
    if length > 0.0:
        return v / length
    return fallback


@tack.func
def _across(t):
    # A unit vector across t: the axis t leans on least, less its part along t.
    e = tack.vec3(1.0, 0.0, 0.0)
    if abs(t[1]) < abs(t[0]) and abs(t[1]) <= abs(t[2]):
        e = tack.vec3(0.0, 1.0, 0.0)
    elif abs(t[2]) < abs(t[0]) and abs(t[2]) < abs(t[1]):
        e = tack.vec3(0.0, 0.0, 1.0)
    return _unit_or(e - t.dot(e) * t, tack.vec3(0.0, 0.0, 1.0))


@tack.kernel
def _rings(offsets, chain, lines, positions, radius, radii, by, sides, step, out, normals,
           source, connectivity, line_of):
    # One chain at a time, in order along it: each point's ring in the plane across
    # the chain, its frame the last one's carried along (its part along the new
    # direction taken out), so the tube does not twist; then the quads between
    # consecutive rings, wound outward.
    for c in range(lines):
        a = offsets[c]
        m = offsets[c + 1] - a
        t = tack.vec3(0.0, 0.0, 1.0)
        n = tack.vec3(1.0, 0.0, 0.0)
        for k in range(m):
            lo = k - 1 if k > 0 else 0
            hi = k + 1 if k + 1 < m else m - 1
            p = positions[chain[a + k]]
            t = _unit_or(positions[chain[a + hi]] - positions[chain[a + lo]], t)
            if k == 0:
                n = _across(t)
            else:
                n = _unit_or(n - t.dot(n) * t, _across(t))
            b = t.cross(n)
            r = radius
            if by == 1:
                r = radius * radii[chain[a + k]]
            for s in range(sides):
                angle = step * s
                d = tack.cos(angle) * n + tack.sin(angle) * b
                at = (a + k) * sides + s
                out[at] = p + r * d
                normals[at] = d
                source[at] = chain[a + k]
            if k + 1 < m:
                for s in range(sides):
                    q = ((a - c) + k) * sides + s
                    nxt = (s + 1) % sides
                    connectivity[4 * q] = (a + k) * sides + s
                    connectivity[4 * q + 1] = (a + k) * sides + nxt
                    connectivity[4 * q + 2] = (a + k + 1) * sides + nxt
                    connectivity[4 * q + 3] = (a + k + 1) * sides + s
                    line_of[q] = c


def _chains(topology):
    """``(offsets, points)``: each line's points in order, a polyline's chain or a
    line cell's two points."""
    if isinstance(topology, PolylineTopology):
        return topology.line_offsets, topology.line_points
    if isinstance(topology, UnstructuredTopology) and all(
            g.shape is shapes.Line for g in topology.groups() if g.count):
        return topology.offsets, topology.connectivity
    raise TypeError("tubes takes lines: a polyline topology (streamlines') or line cells "
                    "(extract_edges', feature_edges')")


@on_inputs_device
def tubes(data, radius, sides=12, radius_by=None, fields=None):
    """``data``'s lines -- polylines (``streamlines``' paths) or line cells
    (``extract_edges``', ``feature_edges``') -- as tubes of ``radius`` (in the
    spatial unit, or a quantity), times ``radius_by`` (a scalar point field, or
    its name) where given: as VTK's ``vtkTubeFilter``. Each point of a line gets a
    ring of ``sides`` points in the plane across the line, and consecutive rings
    are joined by quads, wound outward. The ring's frame is carried along the
    line, its part along each new direction taken out, so a tube does not twist
    about its line. The output's point fields are each ring's line point's, its
    cell fields each quad's line's, and ``normals`` (dimensionless, Values on
    points) is each ring point's outward direction, for shading. A line of one
    point gets a ring and no quads; tubes are open at their ends."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    if not isinstance(data.geometry.space, H1) or data.geometry.space.order != 1:
        raise TypeError("tubes takes lines with their positions at the points")
    sides = int(sides)
    if sides < 3:
        raise ValueError(f"a tube has at least 3 sides, not {sides}")
    radius = _length(data, radius, "tubes' radius")
    offsets, chain = _chains(data.topology)
    radii = None
    if radius_by is not None:
        radii = _field(data, radius_by)
        if arrays.width_of(radii.values) or not _on_points(radii):
            raise TypeError("radius_by is a scalar field on the points")
    lines = data.num_cells
    entries = size_of(chain)
    rings = entries * sides
    quads = (entries - lines) * sides
    idt = id_types.at_least(data.id_dtype, rings)
    vec = tack.vec(3, data.dtype)
    out = tack.empty((rings,), vec)
    normals = tack.empty((rings,), vec)
    source = tack.empty((rings,), data.id_dtype)
    connectivity = tack.empty((4 * quads,), idt)
    line_of = tack.empty((quads,), data.id_dtype)
    if lines:
        positions = arrays.materialize(data.geometry.values)
        by = radii.values if radii is not None else tack.empty((1,), data.dtype)
        _rings(offsets, chain, lines, positions, float(radius), by, int(radii is not None),
               sides, 2.0 * math.pi / sides, out, normals, source, connectivity, line_of)
    types, cell_offsets = uniform_cells(int(shapes.QUAD), quads, idt)
    topology = UnstructuredTopology(types, cell_offsets, connectivity, num_points=rings)
    geometry = Field(H1(topology), out, data.geometry.unit)
    result = carry(data, topology, points=Take(source), cells=Take(line_of),
                   geometry=geometry, fields=fields)
    result.fields["normals"] = Field(Values(result, "points"), normals, "dimensionless")
    return result


def _on_points(field):
    space = field.space
    return (isinstance(space, H1) and space.order == 1) or \
        (isinstance(space, Values) and space.on == "points")

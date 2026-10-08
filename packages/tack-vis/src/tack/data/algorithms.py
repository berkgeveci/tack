"""A few algorithms over the dataset API, one per idea it introduces.

- ``cell_centers``: the geometry evaluated at each cell's parametric center.
- ``face_geometry``: each face's area and unit normal, out of its side 0,
  as fields on faces.
- ``edge_lengths``: a field on edges.
- ``boundary_faces`` / ``extract_surface``: a side set, and a surface
  dataset made of it.
- ``jump``: a cell field's difference across each face, from side 0 to side
  1 -- data on the two sides of a face.
- ``divergence``: each cell's outward sum of a face field, through the cell
  -> face incidence and which side of each face the cell is.
- ``to_points``: a cell or DG field averaged onto points (projection to
  H1), reading both through ``u.dof(c, j)``.
- ``values_at_centers``: any field with a basis evaluated at cell centers.
- ``gradients``: a field's gradient at cell centers, from its own basis and
  the geometry field's Jacobian -- two fields, each through its own space.
- ``discontinuous``: an ``L2`` field from a function per cell -- each cell
  evaluates it at its own corners, so values at shared points can differ.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.algorithms.sort import _run_offsets, sort_by_key
from tack.data.arrays import materialize, width_of
from tack.data.dataset import DataSet, Field, for_each
from tack.data.spaces import H1, L2, Constant, Values
from tack.data.topology import UnstructuredTopology

__all__ = [
    "boundary_faces",
    "cell_centers",
    "discontinuous",
    "divergence",
    "edge_lengths",
    "extract_surface",
    "face_geometry",
    "gradients",
    "jump",
    "to_points",
    "values_at_centers",
]


def _vectors(n, dtype):
    return tack.Vector.field(3, dtype, shape=(n,))


# ── Cells ───────────────────────────────────────────────────────────

@tack.kernel
def _centers(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.position(c, cells.parametric_center())


def cell_centers(data):
    """Each cell's center, where the geometry maps its parametric center: a field on cells."""
    out = _vectors(data.num_cells, data.dtype)
    for_each(_centers, data, "cells", out)
    return Field(Values(data, "cells"), out)


@tack.kernel
def _at_centers(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = u.value(c, cells.parametric_center())


def values_at_centers(data, field):
    """``field`` evaluated at each cell's parametric center, through its basis."""
    out = _like(field.values, data.num_cells)
    for_each(_at_centers, data, "cells", field, out)
    return Field(Values(data, "cells"), out)


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


def gradients(data, field):
    """The gradient of ``field`` (a scalar ``H1``, ``L2`` or ``Constant`` field) at each
    cell's parametric center: a field of 3-vectors on cells. The field's basis gives
    its derivative in parametric coordinates, and the geometry field's Jacobian
    turns that into a world gradient, so the two need not share a space. Cells
    below three dimensions get zero."""
    if width_of(field.values):
        raise TypeError("gradients takes a scalar field")
    out = _vectors(data.num_cells, data.dtype)
    for_each(_gradients, data, "cells", field, out)
    return Field(Values(data, "cells"), out)


@tack.kernel
def _corner_values(cells, u, offsets, out):
    for c in cells:
        first = offsets[cells.entity_id(c)]
        for j in range(cells.NUM_POINTS):
            out[first + j] = u.value(c, cells.parametric_point(j))


def discontinuous(data, field):
    """``field`` (``H1`` or ``Constant``) as an ``L2`` field: each cell's value at each
    of its corners, stored per cell. The same function, in the DG layout; edit the
    values and cells disagree where they meet."""
    space = L2(data)
    out = _like(field.values, space.size)
    for_each(_corner_values, data, "cells", field, space.offsets, out)
    return Field(space, out)


# ── Faces and edges ─────────────────────────────────────────────────

@tack.kernel
def _face_geometry(faces, normals, areas):
    for f in faces:
        n = tack.Vector([0.0, 0.0, 0.0])
        for j in range(faces.NUM_POINTS):
            k = j + 1 if j + 1 < faces.NUM_POINTS else 0
            n += faces.point(f, j).cross(faces.point(f, k))      # Newell's normal
        length = n.norm()
        normals[faces.entity_id(f)] = n / length
        areas[faces.entity_id(f)] = 0.5 * length


def face_geometry(data):
    """``(normal, area)``: fields on faces. A face's normal points out of its side 0."""
    faces = data.topology.faces()
    dtype = data.dtype
    normals = _vectors(faces.num_faces, dtype)
    areas = tack.field(dtype, shape=(faces.num_faces,))
    for_each(_face_geometry, data, "faces", normals, areas)
    return Field(Values(data, "faces"), normals), Field(Values(data, "faces"), areas)


@tack.kernel
def _edge_lengths(edges, out):
    for e in edges:
        out[edges.entity_id(e)] = (edges.point(e, 1) - edges.point(e, 0)).norm()


def edge_lengths(data):
    """Each edge's length: a field on edges."""
    out = tack.field(data.dtype, shape=(data.topology.edges().num_edges,))
    for_each(_edge_lengths, data, "edges", out)
    return Field(Values(data, "edges"), out)


def boundary_faces(data, name="boundary"):
    """The faces with one side, kept in ``data.sets[name]``; returns their ids."""
    data.sets[name] = data.topology.faces().boundary()
    return data.sets[name]


@tack.kernel
def _surface_sizes(ids, kinds, sizes):
    for i in range(ids.shape[0]):
        sizes[i] = 3 if kinds[ids[i]] == 5 else 4


@tack.kernel
def _surface_cells(ids, kinds, rows, starts, types, offsets, connectivity, length):
    for i in range(ids.shape[0]):
        f = ids[i]
        at = starts[i]
        types[i] = tack.u8(kinds[f])
        offsets[i] = at
        row = rows[f]
        for j in range(3 if kinds[f] == 5 else 4):
            connectivity[at + j] = row[j]
        if i == 0:
            offsets[ids.shape[0]] = length


def extract_surface(data, name="boundary"):
    """The faces of set ``name`` as a surface dataset: triangles and quads on the same
    points, each in its side 0's outward order. Face fields become cell fields."""
    faces = data.topology.faces()
    ids = data.sets[name]
    n = ids.shape[0]
    sizes = tack.field(tack.i32, shape=(n,))
    starts = tack.field(tack.i32, shape=(n,))
    if n:
        _surface_sizes(ids, faces.kinds, sizes)
    length = exclusive_scan(sizes, starts, n) if n else 0
    types = tack.field(tack.u8, shape=(n,))
    offsets = tack.zeros(tack.i32, (n + 1,))
    connectivity = tack.field(tack.i32, shape=(length,))
    if n:
        _surface_cells(ids, faces.kinds, faces.rows, starts, types, offsets, connectivity,
                       length)
    surface = UnstructuredTopology(types, offsets, connectivity, num_points=data.num_points)
    on_faces = Values(data, "faces")
    fields = {key: Field(Values(surface, "cells"), _take(f.values, ids))
              for key, f in data.fields.items() if f.space is on_faces}
    # The same points: the geometry's values, on the surface's H1 space.
    return DataSet(surface, Field(H1(surface), data.geometry.values), fields=fields)


# ── Two-sided faces and incidence ───────────────────────────────────

@tack.kernel
def _jump(faces, values, out):
    for f in faces:
        c0 = faces.side_cell(f, 0)
        c1 = faces.side_cell(f, 1)
        out[faces.entity_id(f)] = values[c1] - values[c0] if c1 >= 0 else values[c0] * 0.0


def jump(data, field):
    """A cell field's difference across each face, side 1 minus side 0 (zero on the
    boundary): a field on faces."""
    if not isinstance(field.space, Constant):
        raise TypeError("jump takes a field of one value per cell")
    out = _like(field.values, data.topology.faces().num_faces)
    for_each(_jump, data, "faces", materialize(field.values), out)
    return Field(Values(data, "faces"), out)


@tack.kernel
def _outward_sums(cells, flux, out):
    for c in cells:
        total = flux[0] * 0.0
        for f in range(cells.NUM_FACES):
            sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
            total += sign * flux[cells.face_id(c, f)]
        out[cells.entity_id(c)] = total


def divergence(data, flux):
    """Each cell's outward sum of ``flux``, a field on faces oriented out of side 0:
    a face's value counts plus for its side-0 cell and minus for its side-1 cell."""
    if flux.space is not Values(data, "faces"):
        raise TypeError("divergence sums a field on faces")
    data.topology.faces()                          # the incidence the cell views need
    out = _like(flux.values, data.num_cells)
    for_each(_outward_sums, data, "cells", materialize(flux.values), out)
    return Field(Values(data, "cells"), out)


@tack.kernel
def _incidences(cells, u, start, points, contributions):
    for c in cells:
        first = start + cells.index(c) * cells.NUM_POINTS
        for j in range(cells.NUM_POINTS):
            points[first + j] = cells.point_id(c, j)
            contributions[first + j] = u.dof(c, j)


@tack.kernel
def _average_runs(points, values, offsets, out, count):
    for r in range(count):
        begin = offsets[r]
        end = offsets[r + 1]
        total = values[begin]
        for i in range(begin + 1, end):
            total += values[i]
        out[points[begin]] = total / (end - begin)


def to_points(data, field):
    """A ``Constant`` or ``L2`` field averaged onto the points: an ``H1`` field. Each
    point averages the values the cells around it give it -- a cell's value, or for
    a DG field the cell's own value at that corner. The sums run in a fixed order."""
    if not isinstance(field.space, (Constant, L2)):
        raise TypeError("to_points projects a cell or L2 field")
    groups = data.topology.groups()
    starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_POINTS
                                             for g in groups])]).astype(int)
    total = int(starts[-1])
    points = tack.field(tack.i32, shape=(total,))
    contributions = tack.field(field.values.dtype, shape=(total,))
    for group, start in zip(groups, starts[:-1]):
        if group.count:
            view = data.domain_view("cells", group)
            _incidences(view, field.view(group), int(start), points, contributions)
    out = tack.zeros(field.values.dtype, (data.num_points,))
    if total:
        keys, values = sort_by_key(points, contributions)
        offsets, count = _run_offsets(keys, total)
        _average_runs(keys, values, offsets, out, count)
    return Field(H1(data), out)


# ── Helpers ─────────────────────────────────────────────────────────

def _like(values, n):
    width = width_of(values)
    if width:
        return tack.Vector.field(width, values.dtype, shape=(n,))
    return tack.field(values.dtype, shape=(n,))


@tack.kernel
def _take_rows(values, ids, out):
    for i in range(ids.shape[0]):
        out[i] = values[ids[i]]


def _take(values, ids):
    out = _like(values, ids.shape[0])
    if ids.shape[0]:
        _take_rows(materialize(values), ids, out)
    return out

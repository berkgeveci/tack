"""A few algorithms over the dataset API, one per idea it introduces.

- ``cell_centers``: the geometry evaluated at each cell's parametric center.
- ``face_geometry``: each face's area and unit normal, out of its side 0,
  as fields on faces.
- ``edge_lengths``: a field on edges.
- ``boundary_faces`` / ``extract_surface``: a side set, and a surface
  dataset made of it.
- ``traces``: any field with a basis, as each cell has it on each of its
  faces -- values on the two sides of a face, point by point.
- ``jump``: a field's difference across each face, from side 0 to side 1,
  from its traces, so a DG field's own jumps.
- ``upwind_flux``: a DG-style advective flux through each face, from the
  upwind side's trace.
- ``face_centers``, ``cell_geometry``: face area centroids, and cell volumes
  and centroids by the divergence theorem, for shapes and polyhedra alike.
- ``perot``: a cell vector from face-normal components (C-grid models).

The face-based ones -- face geometry, divergence, jump and upwind flux of
cell data, cell geometry, Perot -- read only the entity methods both paths'
views share (``face_size``, ``num_faces``, ``side_position``...), so they
run on shape-based and polyhedral topologies from one source.
- ``divergence``: each cell's outward sum of a face field, through the cell
  -> face incidence and which side of each face the cell is.
- ``to_points``: a cell or DG field averaged onto points (projection to
  H1), reading both through ``u.dof(c, j)``.
- ``to_cells``: any field with a basis averaged over each cell's corners
  (projection to Constant), point data as VTK's vtkPointDataToCellData.
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
from tack.data import shapes
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import DataSet, Field, for_each, traces
from tack.data.spaces import H1, L2, Constant, Values
from tack.data.topology import UnstructuredTopology

__all__ = [
    "boundary_faces",
    "cell_centers",
    "cell_geometry",
    "discontinuous",
    "divergence",
    "edge_lengths",
    "extract_surface",
    "face_centers",
    "face_geometry",
    "gradients",
    "jump",
    "perot",
    "to_cells",
    "to_points",
    "traces",
    "upwind_flux",
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
        size = faces.face_size(f)                # the shape's constant, or a polygon's
        origin = faces.point(f, 0)
        n = tack.Vector([0.0, 0.0, 0.0])
        for j in range(1, size - 1):
            # Newell's normal, from the face's first point: a fan of triangles.
            n += (faces.point(f, j) - origin).cross(faces.point(f, j + 1) - origin)
        length = n.norm()
        normals[faces.entity_id(f)] = n / length
        areas[faces.entity_id(f)] = 0.5 * length


def face_geometry(data):
    """``(normal, area)``: fields on faces, of any size of polygon. A face's normal
    points out of its side 0; on a non-planar face it is the mean normal."""
    faces = data.topology.faces()
    dtype = data.dtype
    normals = _vectors(faces.num_faces, dtype)
    areas = tack.field(dtype, shape=(faces.num_faces,))
    for_each(_face_geometry, data, "faces", normals, areas)
    return Field(Values(data, "faces"), normals), Field(Values(data, "faces"), areas)


@tack.kernel
def _face_centers(faces, out):
    for f in faces:
        size = faces.face_size(f)
        origin = faces.point(f, 0)
        apex = tack.Vector([0.0, 0.0, 0.0])
        for j in range(size):
            apex += faces.point(f, j) - origin
        apex = apex / size
        total = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for j in range(size):
            a = faces.point(f, j) - origin
            b = faces.point(f, j + 1 if j + 1 < size else 0) - origin
            area = (a - apex).cross(b - apex).norm()
            total += area
            weighted += area * (a + b + apex)
        out[faces.entity_id(f)] = origin + weighted / (3.0 * total)


def face_centers(data):
    """Each face's area centroid: a field on faces. The face is fanned from its
    points' mean, so both its cells -- which store it once -- see the same fan."""
    out = _vectors(data.topology.faces().num_faces, data.dtype)
    for_each(_face_centers, data, "faces", out)
    return Field(Values(data, "faces"), out)


@tack.kernel
def _cell_geometry(cells, volumes, centroids):
    for c in cells:
        # The divergence theorem over the cell's faces, each walked outward and
        # fanned from its points' mean, taken from the cell's first point so
        # cells far from the origin keep their precision.
        origin = cells.side_position(c, 0, 0)
        six = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            size = cells.side_size(c, k)
            apex = tack.Vector([0.0, 0.0, 0.0])
            for j in range(size):
                apex += cells.side_position(c, k, j) - origin
            apex = apex / size
            for j in range(size):
                a = cells.side_position(c, k, j) - origin
                b = cells.side_position(c, k, j + 1 if j + 1 < size else 0) - origin
                t = a.dot(b.cross(apex))
                six += t
                weighted += t * (a + b + apex)
        volumes[cells.entity_id(c)] = six / 6.0
        centroids[cells.entity_id(c)] = origin + weighted / (4.0 * six)


@tack.kernel
def _polygon_geometry(cells, areas, centroids):
    for c in cells:
        # Newell's area vector over the polygon's edges, walked its way, then a fan
        # from its first point weighted by each triangle's area along the normal.
        origin = cells.side_position(c, 0, 0)
        n = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            n += (cells.side_position(c, k, 0) - origin).cross(
                cells.side_position(c, k, 1) - origin)
        unit = n / n.norm()
        twice = 0.0
        weighted = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            a = cells.side_position(c, k, 0) - origin
            b = cells.side_position(c, k, 1) - origin
            t = a.cross(b).dot(unit)
            twice += t
            weighted += t * (a + b)
        areas[cells.entity_id(c)] = 0.5 * twice
        centroids[cells.entity_id(c)] = origin + weighted / (3.0 * twice)


def cell_geometry(data):
    """``(measure, centroid)`` of each cell: fields on cells. A 3D cell's volume, by the
    divergence theorem over its faces -- polyhedral or of a shape -- with faces
    taken as fans from their points' means, so a cell with non-planar faces is
    measured as that polyhedron (a hexahedron with warped faces is not quite its
    trilinear volume); its faces must be wound out of it. A polygon's area, along
    its own normal (Newell's), for a polygonal topology."""
    data.topology.faces()                          # the incidence the cell views need
    n = data.num_cells
    volumes = tack.field(data.dtype, shape=(n,))
    centroids = _vectors(n, data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == 3:
            _cell_geometry(data.domain_view("cells", group), volumes, centroids)
        elif group.count and group.shape is shapes.Polygon:
            _polygon_geometry(data.domain_view("cells", group), volumes, centroids)
    return Field(Values(data, "cells"), volumes), Field(Values(data, "cells"), centroids)


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


@tack.kernel
def _face_cells(ids, sides, out):
    for i in range(ids.shape[0]):
        out[i] = sides[ids[i]][0]


def extract_surface(data, name="boundary"):
    """The faces of set ``name`` as a surface dataset: triangles and quads on the same
    points, each in its side 0's outward order. Fields on faces become the surface's
    cell fields, and so do cell fields (``Constant``, values on cells), each face
    taking its side 0 cell's value; point fields (``H1``, values on points) stay on
    the same points -- an order-2 field's or geometry's values at the points,
    the surface being made of linear faces. The geometry must be ``H1``: an
    ``L2`` one has no shared points for the faces to stand on."""
    if not isinstance(data.geometry.space, H1):
        raise TypeError("extract_surface keeps the points, so needs an H1 geometry")
    if not getattr(data.topology, "reference_cells", True):
        return _polygon_surface(data, name)
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
    cells = tack.field(tack.i32, shape=(n,))
    if n:
        _face_cells(ids, faces.sides, cells)
    fields = {}
    for key, f in data.fields.items():
        space = f.space
        if key == "shape":
            continue
        if _on_faces(data, f):
            fields[key] = Field(Values(surface, "cells"), _take(f.values, ids))
        elif isinstance(space, Constant) or space is Values(data, "cells"):
            fields[key] = Field(Values(surface, "cells"), _take(f.values, cells))
        elif isinstance(space, H1):
            fields[key] = Field(H1(surface), _point_values(f.values, data.num_points))
        elif space is Values(data, "points"):
            fields[key] = Field(Values(surface, "points"), f.values)
    # The same points: the geometry's values, on the surface's H1 space.
    return DataSet(surface, Field(H1(surface), _point_values(data.geometry.values,
                                                             data.num_points)),
                   fields=fields)


@tack.kernel
def _surface_loop_sizes(ids, face_offsets, sizes):
    for i in range(ids.shape[0]):
        f = ids[i]
        sizes[i] = face_offsets[f + 1] - face_offsets[f]


@tack.kernel
def _surface_loops(ids, face_offsets, face_points, starts, loops):
    for i in range(ids.shape[0]):
        f = ids[i]
        for j in range(face_offsets[f + 1] - face_offsets[f]):
            loops[starts[i] + j] = face_points[face_offsets[f] + j]


@tack.kernel
def _close_last(offsets, n, total):
    for i in range(1):
        offsets[n] = total


def _polygon_surface(data, name):
    """``extract_surface`` of a polyhedral topology: the set's faces, each a polygon in
    its side 0's outward order, as a ``PolygonalTopology`` on the same points."""
    from tack.data.polyhedra import PolygonalTopology

    topology = data.topology
    faces = topology.faces()
    ids = data.sets[name]
    n = ids.shape[0]
    sizes = tack.field(tack.i32, shape=(n,))
    offsets = tack.field(tack.i32, shape=(n + 1,))
    if n:
        _surface_loop_sizes(ids, topology.face_offsets, sizes)
    total = exclusive_scan(sizes, offsets, n) if n else 0
    _close_last(offsets, n, total)
    loops = tack.field(tack.i32, shape=(total,))
    if n:
        _surface_loops(ids, topology.face_offsets, topology.face_points, offsets, loops)
    surface = PolygonalTopology(offsets, loops, num_points=data.num_points)
    cells = tack.field(tack.i32, shape=(n,))
    if n:
        _face_cells(ids, faces.sides, cells)
    fields = {}
    for key, f in data.fields.items():
        space = f.space
        if key == "shape":
            continue
        if _on_faces(data, f):
            fields[key] = Field(Values(surface, "cells"), _take(f.values, ids))
        elif isinstance(space, Constant) or space is Values(data, "cells"):
            fields[key] = Field(Values(surface, "cells"), _take(f.values, cells))
        elif isinstance(space, H1):
            fields[key] = Field(H1(surface), _point_values(f.values, data.num_points))
        elif space is Values(data, "points"):
            fields[key] = Field(Values(surface, "points"), f.values)
    return DataSet(surface, Field(H1(surface), _point_values(data.geometry.values,
                                                             data.num_points)),
                   fields=fields)


def _point_values(values, n):
    """An H1 field's values at the points: all of an order-1 field's, the first ``n``
    of an order-2 field's (which go on to its edges, faces and cells)."""
    if size_of(values) == n:
        return values
    return _take(values, tack.arange(n, tack.i32))


# ── Two-sided faces and incidence ───────────────────────────────────

@tack.kernel
def _jump(faces, t, out):
    for f in faces:
        pc = faces.parametric_center()
        inside = t.value(f, 0, pc)
        out[faces.entity_id(f)] = (t.value(f, 1, pc) - inside if faces.num_sides(f) == 2
                                   else inside * 0.0)


@tack.kernel
def _cell_jump(faces, values, out):
    for f in faces:
        inside = values[faces.side_cell(f, 0)]
        out[faces.entity_id(f)] = (values[faces.side_cell(f, 1)] - inside
                                   if faces.num_sides(f) == 2 else inside * 0.0)


def _on_faces(data, field):
    """Whether ``field`` is values on ``data``'s faces, oriented or not."""
    space = field.space
    return (isinstance(space, Values) and space.on == "faces"
            and space.topology is data.topology)


def _on_cells(data, field):
    space = field.space
    return isinstance(space, Constant) or space is Values(data, "cells")


def jump(data, field):
    """A field's difference across each face at the face's center, side 1 minus side
    0 (zero on the boundary): a field on faces. Each side is the field as its own
    cell has it -- ``traces`` -- so a DG field's jumps are its own, not its cells'
    averages; a continuous field's are zero. Cell data (``Constant``, values on
    cells) is read from each side's cell directly, on any topology, polyhedral
    included."""
    out = _like(field.values, data.topology.faces().num_faces)
    if _on_cells(data, field):
        for_each(_cell_jump, data, "faces", materialize(field.values), out)
    else:
        for_each(_jump, data, "faces", traces(data, field), out)
    return Field(Values(data, "faces"), out)


@tack.kernel
def _upwind(faces, t, normals, areas, vx, vy, vz, out):
    for f in faces:
        e = faces.entity_id(f)
        pc = faces.parametric_center()
        vn = normals[e].dot(tack.Vector([vx, vy, vz]))
        upwind = t.value(f, 0, pc)
        if vn < 0.0 and faces.num_sides(f) == 2:
            upwind = t.value(f, 1, pc)
        out[e] = vn * areas[e] * upwind


def upwind_flux(data, field, velocity):
    """The flux of ``field`` carried by a constant ``velocity`` through each face, out
    of side 0: ``(v . n) * area * u``, ``u`` taken from the upwind side at the face's
    center -- the side the flow leaves -- as a DG advection scheme does. A boundary
    face takes its one side's value whichever way the flow goes. ``divergence`` of
    the result is each cell's net outflow."""
    normals, areas = face_geometry(data)
    out = _like(field.values, data.topology.faces().num_faces)
    vx, vy, vz = (float(v) for v in velocity)
    if _on_cells(data, field):
        for_each(_cell_upwind, data, "faces", materialize(field.values), normals.values,
                 areas.values, vx, vy, vz, out)
    else:
        for_each(_upwind, data, "faces", traces(data, field), normals.values, areas.values,
                 vx, vy, vz, out)
    return Field(Values(data, "faces", oriented=True), out)


@tack.kernel
def _cell_upwind(faces, values, normals, areas, vx, vy, vz, out):
    for f in faces:
        e = faces.entity_id(f)
        vn = normals[e].dot(tack.Vector([vx, vy, vz]))
        cell = faces.side_cell(f, 0)
        if vn < 0.0 and faces.num_sides(f) == 2:
            cell = faces.side_cell(f, 1)
        out[e] = vn * areas[e] * values[cell]


@tack.kernel
def _perot(cells, flux, areas, centers, centroids, volumes, out):
    for c in cells:
        e = cells.entity_id(c)
        r = centroids[e]
        u = tack.Vector([0.0, 0.0, 0.0])
        for k in range(cells.num_faces(c)):
            f = cells.face_id(c, k)
            outward = 1.0 - 2.0 * cells.face_side(c, k)
            u += (outward * areas[f] * flux[f]) * (centers[f] - r)
        out[e] = u / volumes[e]


def perot(data, flux):
    """A vector per cell from each face's normal component -- Perot's reconstruction,
    ``u_c = (1/V) sum_f A_f u_f (r_f - r_c)``, as C-grid models (MPAS, ICON) need to
    show their face-normal velocity at cells.

    ``flux`` is a scalar field on faces, the component along each face's normal out
    of its side 0; the side each cell is on gives the sign. Face centers, areas
    and cell centroids and volumes are this module's, so a uniform field's normal
    components give it back exactly on cells with planar faces. First order, as
    Perot's method is."""
    if not _on_faces(data, flux):
        raise TypeError("perot reconstructs from a field on faces")
    _, areas = face_geometry(data)
    centers = face_centers(data)
    volumes, centroids = cell_geometry(data)
    out = _vectors(data.num_cells, data.dtype)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == 3:
            _perot(data.domain_view("cells", group), materialize(flux.values), areas.values,
                   centers.values, centroids.values, volumes.values, out)
    return Field(Values(data, "cells"), out)


@tack.kernel
def _outward_sums(cells, flux, out):
    for c in cells:
        total = flux[0] * 0.0
        for f in range(cells.num_faces(c)):     # the shape's count, or a polyhedron's
            sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
            total += sign * flux[cells.face_id(c, f)]
        out[cells.entity_id(c)] = total


def divergence(data, flux):
    """Each cell's outward sum of ``flux``, values on faces measured out of side 0
    (``Values(data, "faces", oriented=True)``, or plain face values taken so): a
    face's value counts plus for its side-0 cell and minus for its side-1 cell."""
    if not _on_faces(data, flux):
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
    # One entry per (cell, corner), laid out by the topology's groups; a
    # subgroup's cells sit at their positions in their group (index(c)).
    groups = data.topology.groups()
    starts = np.concatenate([[0], np.cumsum([g.count * g.shape.NUM_POINTS
                                             for g in groups])]).astype(int)
    start_of = {id(g): int(s) for g, s in zip(groups, starts[:-1])}
    total = int(starts[-1])
    points = tack.field(tack.i32, shape=(total,))
    contributions = tack.field(field.values.dtype, shape=(total,))
    for group in data.launch_groups("cells", [field]):
        if group.count:
            view = data.domain_view("cells", group)
            _incidences(view, field.view(group), start_of[id(group.root)], points,
                        contributions)
    out = tack.zeros(field.values.dtype, (data.num_points,))
    if total:
        keys, values = sort_by_key(points, contributions)
        offsets, count = _run_offsets(keys, total)
        _average_runs(keys, values, offsets, out, count)
    return Field(H1(data), out)


@tack.kernel
def _corner_averages(cells, u, out):
    for c in cells:
        total = u.corner_value(c, 0)
        for j in range(1, cells.NUM_POINTS):
            total += u.corner_value(c, j)
        out[cells.entity_id(c)] = total / cells.NUM_POINTS


def to_cells(data, field):
    """``field`` averaged over each cell's corners: a ``Constant`` field.

    For point data -- ``H1`` of order 1, or values on points -- each cell gets
    the mean of its points' values, as VTK's vtkPointDataToCellData does. Any
    field with a basis goes through its view's corner values: a DG field
    averages each cell's own, an order-2 field its values at the corners (as
    the filters read it), and a cell constant comes back as itself. Scalars or
    vectors; the values must be floating point.
    """
    space = field.space
    if isinstance(space, Values) and space.on == "points":
        field = Field(H1(data), field.values)
        space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"to_cells averages a field on the points or cells, not {space!r}")
    if field.values.dtype not in (tack.f32, tack.f64):
        raise TypeError(f"to_cells needs floating-point values, not {field.values.dtype.name}")
    out = _like(field.values, data.num_cells)
    for_each(_corner_averages, data, "cells", field, out)
    return Field(Constant(data), out)


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

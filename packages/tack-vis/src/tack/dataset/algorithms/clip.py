"""``clip``: the part of ``data`` where ``by`` -- point data (a field or its name) or an implicit function (``tack.dataset.implicit``) -- is at or above ``value``, or with ``inside`` below it (inside an implicit function's region): Viskores' ClipWithField and (``value`` 0) ClipWithImplicitFunction, by its case tables. Cells wholly kept keep their shape; cut cells become the pieces their case lists. Points are the kept corners, the crossings on cut edges (merged across cells) and centroid points; point data is interpolated onto them, cell data goes to each piece."""


import tack
from tack import arrays
from tack.algorithms.scan import exclusive_scan
from tack.dataset import _clip_tables as tables
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import (
    _close,
    _converted,
    _crossing,
    _field,
    _first_end,
    _floating_dtype,
    _in_unit_of,
    _table,
)
from tack.dataset.algorithms.implicit_values import implicit_values
from tack.dataset.buckets import key_order, run_offsets
from tack.dataset.carry import Interpolate, Pieces, _point_values, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, Values
from tack.dataset.topology import UnstructuredTopology
from tack.runtime.dispatch import on_inputs_device

__all__ = ['clip']


# A table read through a corner order: a pixel's and a voxel's corners run
# x-fastest; and the tables' wedge is VisIt's, VTK's mirrored (each triangle the
# other way round), for the wedge cut as for the wedges it makes.
_TABLE_OF = {int(shapes.PIXEL): (9, [0, 1, 3, 2]),
             int(shapes.VOXEL): (12, [0, 1, 3, 2, 4, 5, 7, 6]),
             int(shapes.WEDGE): (13, [0, 2, 1, 3, 5, 4])}


_KINDS = 5            # cells, connectivity, records, centroids, centroid members


# Entries of any kind one cell can make, at most: a bound for the output's ids.
_MOST = 64

# Cells up to which one thread scans every kind's counts (``_scan_kinds``).
_SERIAL_KINDS = 2048


@tack.kernel
def _scan_kinds(counts, starts, sums, n):
    # Each kind's exclusive scan of the cells' counts, in one thread, as a
    # vector of the kinds, and their totals.
    for i in range(1):
        running = starts[0] * 0
        for c in range(n):
            starts[c] = running
            running += counts[c]
        sums[0] = running


def _shape_tables(shape_id, invert):
    """``(data, cases, edges, remap, whole_type, whole_order)`` for one shape: the
    tables, the corner order to read them through, and the type and corner order
    of a cell kept whole -- a pixel or voxel kept as a quad or hexahedron, as
    VTK's clip keeps them; a wedge in its own (VTK's) order."""
    table, remap = _TABLE_OF.get(shape_id, (shape_id, None))
    if table not in tables.EDGES:
        return None
    n = len(remap) if remap else {1: 1, 3: 2, 5: 3, 9: 4, 10: 4, 12: 8, 13: 6, 14: 5}[table]
    remap = remap or list(range(n))
    whole_order = list(range(n)) if shape_id == int(shapes.WEDGE) else remap
    return (_table(("clip data", invert), tables.DATA[invert]),
            _table(("clip cases", invert, table), tables.CASES[invert][table]),
            _table(("clip edges", table), tables.EDGES[table] or [0, 0]),
            _table(("clip remap", shape_id), remap), table,
            _table(("clip whole", shape_id), whole_order))


@tack.kernel
def _clip_counts(cells, values, value, invert, data, cases, remap, counts):
    for c in cells:
        e = cells.entity_id(c)
        full = (1 << cells.NUM_POINTS) - 1
        case = 0
        for k in range(cells.NUM_POINTS):
            if values[cells.point_id(c, remap[k])] >= value:
                case = case | (1 << k)
        n_cells = 0
        n_conn = 0
        n_records = 0
        n_centroids = 0
        n_members = 0
        whole = case == full if invert == 0 else case == 0
        dropped = case == 0 if invert == 0 else case == full
        if whole:
            n_cells = 1
            n_conn = cells.NUM_POINTS
            n_records = cells.NUM_POINTS
        elif not dropped:
            at = cases[case]
            shapes_left = data[at]
            at += 1
            for s in range(shapes_left):
                kind = data[at]
                count = data[at + 1]
                at += 2
                if kind == 0:
                    n_centroids += 1
                    n_members += count
                    n_records += count
                else:
                    n_cells += 1
                    n_conn += count
                    for k in range(count):
                        if data[at + k] != 20:
                            n_records += 1
                at += count
        counts[e] = [n_cells, n_conn, n_records, n_centroids, n_members]


@tack.kernel
def _clip_emit(cells, values, value, invert, data, cases, edges, remap, whole_type,
               whole_order, starts, types, offsets, connectivity, sources, keys, slots,
               centroid_offsets, centroid_cells):
    for c in cells:
        e = cells.entity_id(c)
        full = (1 << cells.NUM_POINTS) - 1
        case = 0
        for k in range(cells.NUM_POINTS):
            if values[cells.point_id(c, remap[k])] >= value:
                case = case | (1 << k)
        first = starts[e]
        cell = first[0]
        conn = first[1]
        record = first[2]
        centroid = first[3]
        member = first[4]
        whole = case == full if invert == 0 else case == 0
        dropped = case == 0 if invert == 0 else case == full
        if whole:
            types[cell] = tack.u8(whole_type)
            offsets[cell] = conn
            sources[cell] = e
            for k in range(cells.NUM_POINTS):
                p = cells.point_id(c, whole_order[k])
                keys[record] = [p, p]
                slots[record] = conn
                record += 1
                conn += 1
        elif not dropped:
            at = cases[case]
            shapes_left = data[at]
            at += 1
            for s in range(shapes_left):
                kind = data[at]
                count = data[at + 1]
                at += 2
                if kind == 0:
                    centroid_offsets[centroid] = member
                    centroid_cells[centroid] = e
                    for k in range(count):
                        entry = data[at + k]
                        pa = 0
                        pb = 0
                        if entry < 8:
                            pa = cells.point_id(c, remap[entry])
                            pb = pa
                        else:
                            pa = cells.point_id(c, remap[edges[2 * (entry - 8)]])
                            pb = cells.point_id(c, remap[edges[2 * (entry - 8) + 1]])
                        keys[record] = [min(pa, pb), max(pa, pb)]
                        slots[record] = -(member + 1)
                        record += 1
                        member += 1
                else:
                    types[cell] = tack.u8(kind)
                    offsets[cell] = conn
                    sources[cell] = e
                    cell += 1
                    for kk in range(count):
                        # The tables (VisIt's) wind wedges the other way from VTK's
                        # order: each triangle is taken reversed, 0 2 1 3 5 4.
                        k = kk
                        if kind == 13:
                            k = 3 * (kk // 3) + (3 - kk % 3) % 3
                        if data[at + k] == 20:
                            connectivity[conn] = -(centroid + 1)     # resolved later
                        else:
                            entry = data[at + k]
                            pa = 0
                            pb = 0
                            if entry < 8:
                                pa = cells.point_id(c, remap[entry])
                                pb = pa
                            else:
                                pa = cells.point_id(c, remap[edges[2 * (entry - 8)]])
                                pb = cells.point_id(c, remap[edges[2 * (entry - 8) + 1]])
                            keys[record] = [min(pa, pb), max(pa, pb)]
                            slots[record] = conn
                            record += 1
                        conn += 1
                at += count


@tack.kernel
def _resolve_points(keys, order, run_offsets, slots, values, positions, value, connectivity,
                    members, ends, weights, count):
    # Each run of equal keys is one output point: a corner, or an edge crossing
    # interpolated from its end first by position, as contour's are, so every
    # piece holding the edge computes it alike.
    for r in range(count):
        key = keys[order[run_offsets[r]]]
        a = key[0]
        b = key[1]
        if a != b and not _first_end(positions[a], positions[b], a, b):
            a, b = b, a
        ends[r] = [a, b]
        weights[r] = 0.0
        if a != b:
            weights[r] = _crossing(value, values[a], values[b])
        for i in range(run_offsets[r], run_offsets[r + 1]):
            slot = slots[order[i]]
            if slot >= 0:
                connectivity[slot] = r
            else:
                members[-slot - 1] = r


@tack.kernel
def _centroid_entries(connectivity, first_centroid):
    for i in range(connectivity.shape[0]):
        if connectivity[i] < 0:
            connectivity[i] = first_centroid - connectivity[i] - 1


@tack.kernel
def _split_counts(counts, kind, out):
    for i in range(out.shape[0]):
        out[i] = counts[i][kind]


@tack.kernel
def _join_starts(s0, s1, s2, s3, s4, out):
    for i in range(out.shape[0]):
        out[i] = [s0[i], s1[i], s2[i], s3[i], s4[i]]


@on_inputs_device
def clip(data, by, value=0.0, inside=False, fields=None, subdivision=None, tolerance=None):
    """The part of ``data`` where ``by`` -- point data (a field or its name) or an
    implicit function (``tack.dataset.implicit``) -- is at or above ``value``, or
    with ``inside`` below it: Viskores' ClipWithField and (``value`` 0)
    ClipWithImplicitFunction, by its case tables. Cells wholly kept keep their
    shape; cut cells become the pieces their case lists. Points are the kept
    corners, the crossings on cut edges (merged across cells) and centroid
    points; point data is interpolated onto them, cell data goes to each piece.

    A polyhedral topology is clipped cell by cell into polyhedra: each cut cell
    becomes the polyhedron of its kept part, its faces clipped (each face once,
    so faces two cells share stay one face) and its iso-polygons -- as
    ``contour`` traces them, by López's method -- as caps wound out of it; the
    output is a ``PolyhedralTopology``. A point at exactly ``value`` is kept on
    either side (``inside`` keeps at or below). Polygons and polylines are not
    clipped yet.

    A curved geometry or a field above order 1 (or ``subdivision`` above 1, or a
    ``tolerance``) is clipped through subdivision, as ``contour`` cuts it, and the
    result is ``PartitionedDataSet([whole, part])``: ``whole`` the cells the field's
    Bernstein bound keeps entirely, as they are (curved, their fields of any order),
    and ``part`` the clip of the others that the bound does not drop, cut into
    ``subdivision`` lattice steps (by default the largest order of the field and the
    geometry; with ``tolerance``, in the field's unit, the fewest whose estimated
    error is within it). The two meet at the lattice points of the faces between
    them, where a linear sub-face lies against a curved face."""
    from tack.dataset.ghosts import visible

    data, by = visible(data, by)
    from tack.dataset.polyhedra import PolygonalTopology, PolyhedralTopology

    polyhedral = isinstance(data.topology, PolyhedralTopology) and \
        not isinstance(data.topology, PolygonalTopology)
    if not polyhedral and not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("clip takes cells of the linear shapes or polyhedra, not "
                                  "polygons or polylines")
    implicit = hasattr(by, "value") and not isinstance(by, (str, Field))
    field = implicit_values(data, by) if implicit else _field(data, by)
    # In the field's unit; an implicit function's values have none.
    value = _in_unit_of(field, value, "clip's value")
    if arrays.width_of(field.values):
        raise TypeError("clip needs a scalar field")
    from tack.dataset.algorithms.subdivide import split_by_range, subdivides

    if not polyhedral and (subdivides(data, field) or tolerance is not None
                           or (subdivision is not None and subdivision > 1)):
        if tolerance is not None:
            tolerance = _in_unit_of(field, tolerance, "clip's tolerance")
        bound = float("inf")
        lower, upper = (-bound, value) if inside else (value, bound)
        return split_by_range(data, field, lower, upper, subdivision, tolerance, fields,
                              lambda refined, name: clip(refined, name, value, inside, fields))
    if not isinstance(data.geometry.space, H1):
        # Clip's points are merged across cells, so each needs one position.
        raise TypeError("clip needs an H1 geometry: one position per point")
    space = field.space
    if not implicit and not (isinstance(space, H1)
                             or (isinstance(space, Values) and space.on == "points")):
        raise TypeError(f"clip takes point data or an implicit function, not {space!r}")
    precision = _floating_dtype(field.values)
    values = _converted(_point_values(field.values, data.num_points), precision)
    if polyhedral:
        from tack.dataset.algorithms._clip_polyhedra import clip_polyhedra

        return clip_polyhedra(data, arrays.materialize(values), float(value), inside, fields)
    invert = 1 if inside else 0           # the tables' name: keep what is below
    n = data.num_cells

    groups = []
    for group in data.topology.groups():
        if group.count:
            t = _shape_tables(int(group.shape.ID), bool(invert))
            if t is None:
                raise NotImplementedError(f"clip has no table for {group.shape.__name__} cells")
            groups.append((group, t))
    counts = tack.empty((n,), tack.vec(_KINDS, tack.i32))
    if n:
        counts.fill(0)
    for group, (table, cases, _, remap, _, _) in groups:
        _clip_counts(group.view(), values, float(value), invert, table, cases, remap, counts)
    idt = data.id_dtype
    odt = ids.for_output(data, _MOST * n)
    starts = tack.empty((n,), tack.vec(_KINDS, odt))
    if n <= _SERIAL_KINDS:
        # A small mesh: every kind's scan in one thread, one launch and one read,
        # where the parallel scans below are eleven launches.
        sums = tack.empty((1,), tack.vec(_KINDS, odt))
        if n:
            _scan_kinds(counts, starts, sums, n)
        totals = [int(v) for v in sums.numpy().reshape(-1)] if n else [0] * _KINDS
    else:
        totals, parts = [], []
        for kind in range(_KINDS):
            part = tack.empty((n,), tack.i32)
            kind_starts = tack.empty((n,), odt)
            _split_counts(counts, kind, part)
            totals.append(exclusive_scan(part, kind_starts, n))
            parts.append(kind_starts)
        _join_starts(*parts, starts)
    n_cells, n_conn, n_records, n_centroids, n_members = totals

    types = tack.empty((n_cells,), tack.u8)
    offsets = tack.zeros((n_cells + 1,), odt)
    connectivity = tack.empty((n_conn,), odt)
    sources = tack.empty((n_cells,), idt)
    keys = tack.empty((n_records,), tack.vec(2, idt))
    slots = tack.empty((n_records,), odt)
    centroid_offsets = tack.empty((n_centroids + 1,), odt)
    centroid_cells = tack.empty((n_centroids,), idt)
    for group, (table, cases, edges, remap, whole_type, whole_order) in groups:
        _clip_emit(group.view(), values, float(value), invert, table, cases, edges, remap,
                   whole_type, whole_order, starts, types, offsets, connectivity, sources,
                   keys, slots, centroid_offsets, centroid_cells)
    _close(offsets, n_cells, n_conn)
    _close(centroid_offsets, n_centroids, n_members)

    order = key_order(keys, data.num_points, odt)
    point_runs, count = run_offsets(keys, order)
    members = tack.empty((n_members,), odt)
    ends = tack.empty((count,), tack.vec(2, idt))
    # Weights are fractions even when the values are integers (Viskores interpolates
    # an integer field's crossings the same way).
    weights = tack.empty((count,), precision)
    if count:
        _resolve_points(keys, order, point_runs, slots, values,
                        arrays.materialize(data.geometry.values), float(value), connectivity,
                        members, ends, weights, count)
    if n_conn:
        _centroid_entries(connectivity, count)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=count + n_centroids)
    points = Interpolate(ends, weights, averages=(centroid_offsets, members),
                         average_cells=centroid_cells)
    return carry(data, out, points=points, cells=Pieces(sources), fields=fields)

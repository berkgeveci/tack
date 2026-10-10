"""Clip: the part of each cell on one side of a value, cells cut along the way.

Viskores' ClipWithField and ClipWithImplicitFunction, by Viskores' case tables
(``_clip_tables``, VisIt's): a cell whose corners are all kept is kept whole,
one with none is dropped, and any other is replaced by the pieces its case
lists -- tetrahedra, pyramids, wedges, hexahedra (triangles and quads for 2D
cells, lines and vertices below) -- on its kept corners, on points where its
edges cross the value, and on a centroid point some cases add. A corner is
kept when its value is at or above the clip value, or with ``invert`` below
it, as in Viskores. A pixel or voxel kept whole becomes a quad or hexahedron,
as in VTK's clip.

The tables (VisIt's, as Viskores has them) number a wedge's corners as VisIt
does, VTK's order mirrored: a wedge being cut is read through that order, and
every wedge piece is written back in VTK's. Taken as VTK wedges, as Viskores
takes them, the pieces came out inside out.

Count, scan, emit: every corner or edge point a piece uses is recorded under a
key -- a corner ``p`` as ``(p, p)``, an edge as its ``(low, high)`` point ids
-- and the records are merged by bucketing, so a shared edge's point is one
point, interpolated from its lower id as every cell sees it. Centroid points
follow, one per case that has one, the average of the points it lists.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.data import _clip_tables as tables
from tack.data import arrays, ids, shapes
from tack.data.buckets import bucket_order, run_offsets
from tack.data.carry import Interpolate, Pieces, _point_values, carry
from tack.data.dataset import Field
from tack.data.spaces import H1, Values
from tack.data.topology import UnstructuredTopology, _as_field

__all__ = ["clip"]

# A table read through a corner order: a pixel's and a voxel's corners run
# x-fastest; and the tables' wedge is VisIt's, VTK's mirrored (each triangle the
# other way round), for the wedge cut as for the wedges it makes.
_TABLE_OF = {int(shapes.PIXEL): (9, [0, 1, 3, 2]),
             int(shapes.VOXEL): (12, [0, 1, 3, 2, 4, 5, 7, 6]),
             int(shapes.WEDGE): (13, [0, 2, 1, 3, 5, 4])}
_KINDS = 5            # cells, connectivity, records, centroids, centroid members
# Entries of any kind one cell can make, at most: a bound for the output's ids.
_MOST = 64


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
    return (_as_field(np.asarray(tables.DATA[invert], np.int32), tack.i32),
            _as_field(np.asarray(tables.CASES[invert][table], np.int32), tack.i32),
            _as_field(np.asarray(tables.EDGES[table] or [0, 0], np.int32), tack.i32),
            _as_field(np.asarray(remap, np.int32), tack.i32), table,
            _as_field(np.asarray(whole_order, np.int32), tack.i32))


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
               centroid_offsets):
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
def _resolve_points(keys, order, run_offsets, slots, values, value, connectivity, members,
                    ends, weights, count):
    # Each run of equal keys is one output point: a corner, or an edge crossing
    # interpolated from its lower point id.
    for r in range(count):
        key = keys[order[run_offsets[r]]]
        a = key[0]
        b = key[1]
        ends[r] = key
        weights[r] = 0.0
        if a != b:
            weights[r] = (value - values[a]) / (values[b] - values[a])
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
def _close(offsets, n, total):
    for i in range(1):
        offsets[n] = total


@tack.kernel
def _split_counts(counts, kind, out):
    for i in range(out.shape[0]):
        out[i] = counts[i][kind]


@tack.kernel
def _join_starts(s0, s1, s2, s3, s4, out):
    for i in range(out.shape[0]):
        out[i] = [s0[i], s1[i], s2[i], s3[i], s4[i]]


def clip(data, by, value=0.0, invert=False, fields=None):
    """The part of ``data`` where ``by`` -- point data (a field or its name) or an
    implicit function (``tack.data.implicit``) -- is at or above ``value``, or
    with ``invert`` below it: Viskores' ClipWithField and (``value`` 0)
    ClipWithImplicitFunction, by its case tables. Cells wholly kept keep their
    shape; cut cells become the pieces their case lists. Points are the kept
    corners, the crossings on cut edges (merged across cells) and centroid
    points; point data is interpolated onto them, cell data goes to each piece.
    """
    from tack.data.algorithms import implicit_values
    from tack.data.filters import _field

    if not getattr(data.topology, "reference_cells", True):
        raise NotImplementedError("clip takes cells of the linear shapes, not polyhedra")
    if hasattr(by, "value") and not isinstance(by, (str, Field)):
        if not isinstance(data.geometry.space, H1):
            raise TypeError("clip by a function needs an H1 geometry: one position per point")
        field = implicit_values(data, by)
    else:
        field = _field(data, by)
        space = field.space
        if not (isinstance(space, H1) or (isinstance(space, Values) and space.on == "points")):
            raise TypeError(f"clip takes point data or an implicit function, not {space!r}")
    if arrays.width_of(field.values):
        raise TypeError("clip needs a scalar field")
    values = arrays.materialize(_point_values(field.values, data.num_points))
    invert = 1 if invert else 0
    n = data.num_cells

    groups = []
    for group in data.topology.groups():
        if group.count:
            t = _shape_tables(int(group.shape.ID), bool(invert))
            if t is None:
                raise NotImplementedError(f"clip has no table for {group.shape.__name__} cells")
            groups.append((group, t))
    counts = tack.Vector.field(_KINDS, tack.i32, shape=(n,))
    if n:
        counts.from_numpy(np.zeros((n, _KINDS), np.int32))
    for group, (table, cases, _, remap, _, _) in groups:
        _clip_counts(group.view(), values, float(value), invert, table, cases, remap, counts)
    idt = data.id_dtype
    odt = ids.for_output(data, _MOST * n)
    totals, parts = [], []
    for kind in range(_KINDS):
        part = tack.field(tack.i32, shape=(n,))
        starts = tack.field(odt, shape=(n,))
        if n:
            _split_counts(counts, kind, part)
        totals.append(exclusive_scan(part, starts, n) if n else 0)
        parts.append(starts)
    n_cells, n_conn, n_records, n_centroids, n_members = totals
    starts = tack.Vector.field(_KINDS, odt, shape=(n,))
    if n:
        _join_starts(*parts, starts)

    types = tack.field(tack.u8, shape=(n_cells,))
    offsets = tack.zeros(odt, (n_cells + 1,))
    connectivity = tack.field(odt, shape=(n_conn,))
    sources = tack.field(idt, shape=(n_cells,))
    keys = tack.Vector.field(2, idt, shape=(n_records,))
    slots = tack.field(odt, shape=(n_records,))
    centroid_offsets = tack.field(odt, shape=(n_centroids + 1,))
    for group, (table, cases, edges, remap, whole_type, whole_order) in groups:
        _clip_emit(group.view(), values, float(value), invert, table, cases, edges, remap,
                   whole_type, whole_order, starts, types, offsets, connectivity, sources,
                   keys, slots, centroid_offsets)
    _close(offsets, n_cells, n_conn)
    _close(centroid_offsets, n_centroids, n_members)

    order, _ = bucket_order(keys, data.num_points, odt)
    point_runs, count = run_offsets(keys, order)
    members = tack.field(odt, shape=(n_members,))
    ends = tack.Vector.field(2, idt, shape=(count,))
    weights = tack.field(arrays.dtype_of(field.values), shape=(count,))
    if count:
        _resolve_points(keys, order, point_runs, slots, values, float(value), connectivity,
                        members, ends, weights, count)
    if n_conn:
        _centroid_entries(connectivity, count)
    out = UnstructuredTopology(types, offsets, connectivity, num_points=count + n_centroids)
    points = Interpolate(ends, weights, averages=(centroid_offsets, members))
    return carry(data, out, points=points, cells=Pieces(sources), fields=fields)

"""``ghost_points``: each piece of a partitioned point cloud with copies of the other
pieces' points it needs, so that filters over neighbours run per piece and are
right on the points it owns, as ``ghost_cells`` does for meshes.

Two needs, two ways of finding the copies:

- **Within a radius** (``radius=``): each piece receives every other piece's
  points within ``radius`` of its box -- every point within ``radius`` of a
  point it owns -- found by the boxes alone, in one exchange.
- **The k nearest** (``k=``): each piece receives exactly the points among its
  owned points' ``k`` nearest that other pieces hold. Each piece finds its
  points' ``k`` nearest among its own, whose distance bounds the true one; a
  point whose ball of that radius reaches another piece's box asks that piece
  for its ``k`` nearest there; the ``k`` best of the answers are fetched.

In a periodic box the copies are images: a point copied across a face of the
box is moved by the box's side, so a piece is an ordinary, non-periodic cloud
with its neighbours around it -- run filters on it without ``periodic``. A
piece near two faces holds images of its own points too.

The matching runs on the host and the exchanges on the pieces' devices: it is
set-up work, as ``ghost_cells`` is. The result keeps the exchange that brings
each copy from its owner, which ``update_ghosts`` runs for named fields.
"""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.dataset import _matching, shapes
from tack.dataset._matching import INDEX_BITS
from tack.dataset.algorithms._common import _length
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.algorithms.ghost_cells import _carried, _joined, _kind
from tack.dataset.carry import _point_values, _take
from tack.dataset.dataset import Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.point_locator import MAX_K, PointLocator, periodic_box
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import UnstructuredTopology

__all__ = ["ghost_points"]

# An image's shift, in sides of the box along each axis, coded (sx + 1) + 3 (sy + 1)
# + 9 (sz + 1); _HOME is no shift.
_SHIFTS = np.array([(c % 3 - 1, c // 3 % 3 - 1, c // 9 - 1) for c in range(27)], np.float64)
_HOME = 13


def ghost_points(data, radius=None, k=None, periodic=None, around=None, fields=None,
                 world=None):
    """``data``'s pieces, point clouds (``point_cloud``: one vertex cell per point),
    each with copies of the other pieces' points it needs appended as ghosts: a
    new ``PartitionedDataSet``.

    With ``radius``, every point within ``radius`` of the piece's box: a filter
    over the points within ``radius`` (``outliers``' radius test,
    ``point_normals``, ``sph_density``, ``sph_interpolate`` with its kernel's
    reach) then gives each owned point the whole's answer. ``around`` (a
    ``PartitionedDataSet`` with a piece for each of ``data``'s) widens each
    piece's box to its own piece's points too: the targets ``sph_interpolate``
    resamples onto there. With ``k`` (at most 32), every point among an owned
    point's ``k`` nearest, wherever it is: ``point_normals`` and ``outliers``
    by the ``k`` nearest then see the whole's neighbours.

    Each copy is a vertex cell marked ``DUPLICATECELL`` at a point marked
    ``DUPLICATEPOINT``, owned by the piece it came from, after the piece's own
    points, which keep their order. Point and cell fields come along
    (``fields`` choosing, as filters do), sets keep their members, and global
    ids come along where every piece has them. ``update_ghosts`` refreshes a
    field's copies from their owners, after a per-piece filter that is right
    only on what a piece owns (``sph_density`` before ``sph_interpolate``).

    In a ``periodic`` box (as ``PointLocator`` takes it) a copy across a face is
    an image, moved by the box's side: run filters on the result without
    ``periodic``. ``data`` holds this rank's pieces, on any devices, in
    ``world`` (``data.world`` by default), on one origin, without ghosts."""
    radius = _length(data, radius, "ghost_points' radius")
    periodic = _length(data, periodic, "ghost_points' periodic box")
    ghosted = _ghosted(data, radius, k, periodic, around, fields, world)
    result = PartitionedDataSet(ghosted.pieces, world=ghosted.blocks.world)
    result._ghost_plans = {"points": ghosted.plan, "cells": ghosted.plan}
    return result


class _Ghosted:
    """What ``_ghosted`` makes: the pieces with their copies; each local piece's
    identity of every point, ``(owner << INDEX_BITS) | index there``, a host
    array; its count of own points; the plan bringing each copy from its owner;
    and the blocks."""

    def __init__(self, blocks, pieces, identity, owned, plan):
        self.blocks, self.pieces, self.identity = blocks, pieces, identity
        self.owned, self.plan = owned, plan


def _ghosted(data, radius, k, periodic, around, fields, world):
    if (radius is None) == (k is None):
        raise ValueError("ghost_points takes radius (every point within it) or k (the "
                         "k nearest)")
    if radius is not None and not radius >= 0:
        raise ValueError(f"radius must be at least zero, not {radius!r}")
    if k is not None and not (isinstance(k, (int, np.integer)) and 1 <= k <= MAX_K):
        raise ValueError(f"k is a whole number from 1 to {MAX_K}, not {k!r}")
    if around is not None and k is not None:
        raise ValueError("around widens the boxes a radius reaches from; k needs none")
    if not isinstance(data, PartitionedDataSet):
        raise TypeError(f"ghost_points takes a PartitionedDataSet, not {type(data).__name__}")
    world = data.world if world is None else world
    origin = common_origin(data, "ghost_points", world)
    pieces = list(data)
    for i, piece in enumerate(pieces):
        _check_cloud(i, piece)
    names = _carried(pieces, fields, "ghost_points")
    blocks = tc.Blocks([piece.device for piece in pieces], world)
    if blocks.count > 1 << (63 - INDEX_BITS):
        raise ValueError(f"ghost_points handles up to {1 << (63 - INDEX_BITS)} pieces")
    box = periodic_box(periodic)
    sides = np.zeros(3) if box is None else np.asarray(box[1]) - np.asarray(box[0])
    codes = range(27) if box is not None else (_HOME,)
    hosts = {g: piece.positions().reshape(-1, 3).astype(np.float64)
             for g, piece in zip(blocks.local, pieces)}
    # Distances are measured in the geometry's type; a copy is sent a few of its
    # roundings early, so the piece's test and the whole's agree.
    scale = blocks.all_reduce({g: float(np.abs(hosts[g]).max()) if len(hosts[g]) else 0.0
                               for g in blocks.local}, "max") if blocks.count else 0.0
    dtypes = {piece.dtype for piece in pieces}
    eps = max((float(np.finfo(d.numpy_dtype).eps) for d in dtypes), default=0.0)
    slack = 8.0 * eps * (scale + float(radius or 0.0) + float(sides.max()))
    if radius is not None:
        if box is not None and radius >= 0.5 * float(sides.min()):
            raise ValueError(f"a radius of {radius} reaches half the periodic box "
                             f"({float(sides.min())} across): ghost less far")
        boxes = _boxes(blocks, hosts, around, origin)
        sends = _within(blocks, hosts, boxes, float(radius) + slack, sides, codes)
    else:
        sends = _nearest(blocks, pieces, hosts, int(k), slack, sides, codes, box)
    incoming = blocks.exchange(_messages(blocks, pieces, hosts, sends, sides, names))
    numbered = {kind: bool(blocks.all_reduce({
        g: int(p.global_ids is not None and getattr(p.global_ids, kind) is not None)
        for g, p in zip(blocks.local, pieces)}, "min")) if blocks.count else False
        for kind in ("points", "cells")}
    out, identity, owned, receives = [], {}, {}, {}
    for g, piece in zip(blocks.local, pieces):
        result, identity[g], receives[g] = _appended(g, piece, incoming[g], names, numbered,
                                                     blocks)
        owned[g] = piece.num_points
        out.append(result)
    return _Ghosted(blocks, out, identity, owned, _matching.plan(blocks, receives))


def _check_cloud(i, piece):
    if not isinstance(piece.geometry.space, H1):
        raise TypeError(f"ghost_points needs an H1 geometry; partition {i}'s is "
                        f"{piece.geometry.space!r}")
    topology, n = piece.topology, piece.num_points
    cloud = isinstance(topology, UnstructuredTopology) and piece.num_cells == n
    if cloud and n:
        cloud = (np.all(np.asarray(arrays.to_host(topology.types)).reshape(-1)
                        == int(shapes.VERTEX))
                 and np.array_equal(np.asarray(arrays.to_host(topology.offsets)).reshape(-1),
                                    np.arange(n + 1))
                 and np.array_equal(np.asarray(arrays.to_host(topology.connectivity))
                                    .reshape(-1), np.arange(n)))
    if not cloud:
        raise TypeError(f"ghost_points takes point clouds, one vertex cell per point "
                        f"(point_cloud); partition {i} is not one")
    marks = piece.ghosts
    if marks is not None and marks.points is not None and \
            (marks.points.numpy() & Ghosts.DUPLICATEPOINT).any():
        raise ValueError(f"partition {i} already has ghost points; remove_ghosts first")


def _boxes(blocks, hosts, around, origin):
    """Every block's box (``(lower, upper)`` or None for no points), on every rank:
    its points', widened to its piece of ``around``."""
    if around is not None:
        if not isinstance(around, PartitionedDataSet) or len(around) != len(blocks.local):
            raise ValueError("around is a PartitionedDataSet with a piece for each of "
                             "data's pieces on this rank")
    mine = {}
    for i, g in enumerate(blocks.local):
        x = hosts[g]
        if around is not None and around[i].num_points:
            other = around[i].positions().reshape(-1, 3).astype(np.float64) + \
                (np.asarray(around[i].origin) - np.asarray(origin))
            x = np.concatenate([x, other])
        mine[g] = (x.min(axis=0), x.max(axis=0)) if len(x) else None
    return blocks.all_gather(mine)


def _distance2(x, box):
    """Squared distances of points ``x`` from ``box``, ``(lower, upper)``."""
    d = np.maximum(np.maximum(box[0] - x, 0.0), x - box[1])
    return (d * d).sum(axis=1)


def _within(blocks, hosts, boxes, reach, sides, codes):
    """``{g: {b: (indices, shift codes)}}``: each of this rank's pieces' points, or
    their images, within ``reach`` of another block's box (its own, for images)."""
    sends = {}
    for g in blocks.local:
        x = hosts[g]
        sends[g] = {}
        if not len(x):
            continue
        low, high = x.min(axis=0), x.max(axis=0)
        for b, target in enumerate(boxes):
            if target is None:
                continue
            found, shifts = [], []
            for code in codes:
                if b == g and code == _HOME:
                    continue
                shift = _SHIFTS[code] * sides
                if _distance2((low + shift)[None], (target[0] - (high - low), target[1]))[0] \
                        > reach * reach:
                    continue              # the boxes are too far apart
                hit = np.flatnonzero(_distance2(x + shift, target) <= reach * reach)
                found.append(hit)
                shifts.append(np.full(len(hit), code, np.int64))
            if found and sum(map(len, found)):
                sends[g][b] = (np.concatenate(found), np.concatenate(shifts))
    return sends


def _nearest(blocks, pieces, hosts, k, slack, sides, codes, box):
    """``{g: {b: (indices, shift codes)}}``: the points of this rank's pieces among
    some other piece's owned points' ``k`` nearest, as images where they are."""
    locators, local = {}, {}
    for g, piece in zip(blocks.local, pieces):
        with piece.device:
            locators[g] = PointLocator(piece)
            if piece.num_points:
                ids, d2s = locators[g].nearest(k)
                local[g] = (ids.numpy().astype(np.int64), d2s.numpy().astype(np.float64))
            else:
                local[g] = (np.zeros((0, k), np.int64), np.zeros((0, k)))
    # A point's k nearest among its piece's bound the true distance: it asks the
    # pieces whose boxes (or their images) are no farther.
    boxes = blocks.all_gather({g: (hosts[g].min(axis=0), hosts[g].max(axis=0))
                               if len(hosts[g]) else None for g in blocks.local})
    asked, queries = {}, {}
    for g, piece in zip(blocks.local, pieces):
        x = hosts[g]
        last = local[g][1][:, k - 1] if len(x) else np.zeros(0)
        reach = np.where(last >= 0, np.sqrt(np.maximum(last, 0.0)), np.inf) + slack
        asked[g], queries[g] = {}, {}
        for b, target in enumerate(boxes):
            if target is None or not len(x):
                continue
            rows, shifts = [], []
            for code in codes:
                if b == g and code == _HOME:
                    continue
                hit = np.flatnonzero(_distance2(x + _SHIFTS[code] * sides, target)
                                     <= reach * reach)
                rows.append(hit)
                shifts.append(np.full(len(hit), code, np.int64))
            if rows and sum(map(len, rows)):
                rows, shifts = np.concatenate(rows), np.concatenate(shifts)
                asked[g][b] = (rows, shifts)
                at = (x[rows] + _SHIFTS[shifts] * sides).astype(piece.dtype.numpy_dtype)
                queries[g][b] = {"x": tack.array(at, tack.vec3(piece.dtype),
                                                 device=blocks.device(g))}
    received = blocks.exchange(queries)
    answers = {}
    for g, piece in zip(blocks.local, pieces):
        answers[g] = {}
        for source, message in received[g].items():
            with piece.device:
                x = message["x"]
                if x.dtype != tack.vec3(piece.dtype):
                    x = x.astype(tack.vec3(piece.dtype))
                ids, d2s = locators[g].nearest(k, x)
            # Distances travel as f64 bits in u64, which every device holds (a
            # device without f64, as Metal, cannot hold them as f64).
            bits = np.ascontiguousarray(d2s.numpy().reshape(-1).astype(np.float64)
                                        .view(np.uint64))
            answers[g][source] = {"ids": ids.reshape((ids.size,)).astype(tack.i64),
                                  "d2": tack.array(bits, tack.u64, device=blocks.device(g))}
    replies = blocks.exchange(answers)

    # The k best of each owned point's candidates, by distance, then piece, image
    # and index; those of other pieces (or images) are fetched.
    requests = {}
    for g in blocks.local:
        n = len(hosts[g])
        ids, d2s = local[g]
        point = [np.repeat(np.arange(n), k)]
        piece_of, code_of = [np.full(n * k, g, np.int64)], [np.full(n * k, _HOME, np.int64)]
        index, distance = [ids.reshape(-1)], [d2s.reshape(-1)]
        for b, message in replies[g].items():
            rows, shifts = asked[g][b]
            point.append(np.repeat(rows, k))
            piece_of.append(np.full(len(rows) * k, b, np.int64))
            code_of.append(np.repeat(shifts, k))
            index.append(message["ids"].numpy())
            distance.append(message["d2"].numpy().view(np.float64))
        point, piece_of, code_of = map(np.concatenate, (point, piece_of, code_of))
        index, distance = np.concatenate(index), np.concatenate(distance)
        real = index >= 0
        point, piece_of, code_of = point[real], piece_of[real], code_of[real]
        index, distance = index[real], distance[real]
        order = np.lexsort((index, code_of, piece_of, distance, point))
        point = point[order]
        rank = np.arange(len(point)) - np.searchsorted(point, point, side="left")
        best = order[rank < k]
        if box is not None and len(best) and \
                float(np.sqrt(distance[best].max())) >= 0.5 * float(sides.min()):
            raise ValueError("an owned point's k nearest reach half the periodic box: "
                             "take fewer, or a bigger box")
        away = best[(piece_of[best] != g) | (code_of[best] != _HOME)]
        requests[g] = {}
        for b in np.unique(piece_of[away]).tolist():
            chosen = away[piece_of[away] == b]
            # A point found near this one moved by s is this one's neighbour moved
            # by -s: the opposite code.
            pairs = np.unique(np.stack([26 - code_of[chosen], index[chosen]], axis=1), axis=0)
            requests[g][int(b)] = {
                "index": tack.array(pairs[:, 1].copy(), tack.i64, device=blocks.device(g)),
                "shift": tack.array(pairs[:, 0].copy(), tack.i64, device=blocks.device(g))}
    wanted = blocks.exchange(requests)
    return {g: {asker: (message["index"].numpy(), message["shift"].numpy())
                for asker, message in wanted[g].items()} for g in blocks.local}


def _messages(blocks, pieces, hosts, sends, sides, names):
    """Each piece's copies for each other piece: positions (images moved), the
    copies' identities, marks, fields and global ids."""
    outgoing = {}
    for g, piece in zip(blocks.local, pieces):
        device = blocks.device(g)
        marks = piece.ghosts
        n = piece.num_points
        point_marks = marks.points.numpy() if marks is not None and marks.points is not None \
            else np.zeros(n, np.uint8)
        cell_marks = marks.cells.numpy() if marks is not None and marks.cells is not None \
            else np.zeros(n, np.uint8)
        geometry = arrays.materialize(_point_values(piece.geometry.values, n))
        outgoing[g] = {}
        for b, (index, shifts) in sends[g].items():
            chosen = tack.array(index.astype(np.int64), tack.i64, device=device)
            positions = _take(geometry, chosen)
            moved = shifts != _HOME
            if moved.any():
                host = positions.numpy().reshape(-1, 3).astype(np.float64)
                host[moved] = hosts[g][index[moved]] + _SHIFTS[shifts[moved]] * sides
                positions = tack.array(host.astype(piece.dtype.numpy_dtype),
                                       tack.vec3(piece.dtype), device=device)
            message = {
                "positions": positions,
                "identity": tack.array((np.int64(g) << INDEX_BITS) | index.astype(np.int64),
                                       tack.i64, device=device),
                "index": chosen,
                "point_marks": tack.array(point_marks[index], tack.u8, device=device),
                "cell_marks": tack.array(cell_marks[index], tack.u8, device=device),
            }
            for name in names:
                message["field." + name] = _take(piece.fields[name].values, chosen)
            ids = piece.global_ids
            for kind in ("points", "cells"):
                if ids is not None and getattr(ids, kind) is not None:
                    message["ids." + kind] = _take(getattr(ids, kind), chosen)
            outgoing[g][int(b)] = message
    return outgoing


def _appended(g, piece, received, names, numbered, blocks):
    """The piece with its copies appended, in order of the block they came from;
    its points' identities; where its copies come from, ``{owner: (owner's
    indices, local indices)}``."""
    n = piece.num_points
    sources = list(received.items())
    counts = [int(message["identity"].size) for _, message in sources]
    starts = np.concatenate(([n], n + np.cumsum(counts))).astype(np.int64)
    marks = piece.ghosts
    own_points = marks.points.numpy() if marks is not None and marks.points is not None \
        else np.zeros(n, np.uint8)
    own_cells = marks.cells.numpy() if marks is not None and marks.cells is not None \
        else np.zeros(n, np.uint8)
    identity = np.concatenate([(np.int64(g) << INDEX_BITS) | np.arange(n, dtype=np.int64)]
                              + [message["identity"].numpy() for _, message in sources])
    point_marks = np.concatenate([own_points] + [message["point_marks"].numpy()
                                                 | Ghosts.DUPLICATEPOINT
                                                 for _, message in sources])
    cell_marks = np.concatenate([own_cells] + [message["cell_marks"].numpy()
                                               | Ghosts.DUPLICATECELL
                                               for _, message in sources])
    receives = {int(source): (message["index"].numpy(),
                              np.arange(starts[m], starts[m + 1], dtype=np.int64))
                for m, (source, message) in enumerate(sources)}
    with blocks.device(g):
        geometry = arrays.materialize(_point_values(piece.geometry.values, n))
        positions = _joined([geometry] + [message["positions"] for _, message in sources])
        out = _vertices(positions, piece)
        out.ghosts = Ghosts(cells=cell_marks, points=point_marks)
        for name in names:
            field = piece.fields[name]
            values = _joined([field.values] + [message["field." + name]
                                               for _, message in sources])
            space = field.space
            if _kind(field) == "points":
                space = H1(out) if isinstance(space, H1) else Values(out, "points")
            else:
                space = Constant(out) if isinstance(space, Constant) else Values(out, "cells")
            out.fields[name] = Field(space, values, field.unit)
        for name, members in piece.sets.items():
            if members.on in ("points", "cells"):
                out.sets[name] = Set(out, members.on, members.ids)
        ids = {kind: _joined([getattr(piece.global_ids, kind)] +
                             [message["ids." + kind] for _, message in sources])
               for kind in ("points", "cells") if numbered[kind]}
        if ids:
            out.global_ids = GlobalIds(cells=ids.get("cells"), points=ids.get("points"))
    return out, identity, receives

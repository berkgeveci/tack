"""Ghost cells for the pieces of a partitioned dataset, and refreshing their values.

``ghost_cells`` gives each piece a layer of its neighbours' cells, as VTK's
``vtkGhostCellsGenerator`` does, through ``tack.communication``: the same code
in one process (pieces on any devices) and across MPI ranks.

1. Each piece keys its boundary points by their exact position (a hash of
   the coordinates' bits) and the keys are sorted over every block
   (``tack.communication.sort_by_key``), which brings every copy of a point
   to one block.
2. There, points at one position on several pieces are matched, and each
   piece hears which other pieces hold each of its boundary points, and at
   which index. A point's owner is the lowest piece holding it, as ``split``
   decides; the point's identity everywhere is (owner, index there).
3. Each piece sends each neighbour its cells that use a point they share,
   with those cells' points (their identities, positions and point fields)
   and cell fields.
4. Each piece appends what it received: the cells as ``DUPLICATECELL``,
   points it has not already got -- one per identity, wherever it came from
   -- as ``DUPLICATEPOINT``, as are its own shared points another piece owns.

The result keeps the two exchanges a refresh needs (each ghost cell from the
piece that sent it, each ghost point from its owner) as ``Plan``s, which
``update_ghosts`` runs for named fields.

The matching and assembly run on the host, and the exchanges on the pieces'
devices: ghost generation is set-up work, done once per decomposition.
"""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.dataset.carry import _take
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import UnstructuredTopology

__all__ = ["ghost_cells", "update_ghosts"]

# A point's identity, (owner piece, index there), as one integer.
_INDEX_BITS = 40


def ghost_cells(data, layers=1, world=None, fields=None):
    """``data``'s pieces, each with a layer of its neighbours' cells added as ghosts:
    a new ``PartitionedDataSet``, as VTK's ``vtkGhostCellsGenerator`` makes.

    ``data`` holds this rank's pieces, unstructured (``UnstructuredTopology``)
    and without ghost cells, on any devices; ``world`` is the
    ``tack.communication.World`` they live in (``data.world``, this process
    alone, by default). Pieces neighbour where they share points at exactly
    the same position. Each piece gains the cells of other pieces that use a
    point it shares with them (``DUPLICATECELL``), and their points it lacks
    (``DUPLICATEPOINT``); a point several pieces share belongs to the lowest
    of them, and is ``DUPLICATEPOINT`` in the others. Input marks come along:
    ``HIDDENCELL`` and the other bits are copied with each cell.

    Point and cell fields (``H1`` of order 1, ``Constant``, values on points
    or cells) are carried, ``fields`` choosing them as filters do (``None``:
    all); point and cell sets keep their members. ``update_ghosts`` refreshes
    a field's ghost values from their owners afterwards, after a filter that
    is right only where it sees every neighbour.

    One layer, which every filter needs to give each piece its share of the
    whole's output; more are not made yet. Structured pieces get theirs from
    ``split(..., ghost_layers=n)``.
    """
    if layers != 1:
        raise NotImplementedError(f"ghost_cells makes one layer of ghost cells, not "
                                  f"{layers!r}; structured pieces get more from "
                                  f"split(..., ghost_layers=n)")
    if not isinstance(data, PartitionedDataSet):
        raise TypeError(f"ghost_cells takes a PartitionedDataSet, not {type(data).__name__}")
    world = data.world if world is None else world
    pieces = list(data)
    names = _carried(pieces, fields)
    for i, piece in enumerate(pieces):
        _check_piece(i, piece)
    blocks = tc.Blocks([piece.device for piece in pieces], world)
    if blocks.count > 1 << (63 - _INDEX_BITS):
        raise ValueError(f"ghost_cells handles up to {1 << (63 - _INDEX_BITS)} pieces")

    hosts = {g: _host(piece) for g, piece in zip(blocks.local, pieces)}
    partners = _match(blocks, pieces, hosts)
    owners = {g: _owners(g, hosts[g]["num_points"], partners[g]) for g in blocks.local}
    sent = _ghost_parts(blocks, pieces, hosts, partners, owners, names)
    incoming = blocks.exchange(sent)

    out, receive_points, receive_cells = [], {}, {}
    for g, piece in zip(blocks.local, pieces):
        result, points_from, cells_from = _appended(g, piece, hosts[g], owners[g],
                                                    incoming[g], names, blocks)
        out.append(result)
        receive_points[g], receive_cells[g] = points_from, cells_from
    point_plan = _plan(blocks, receive_points)
    cell_plan = _plan(blocks, receive_cells)
    result = PartitionedDataSet(out, world=world)
    result._ghost_plans = {"points": point_plan, "cells": cell_plan}
    return result


def update_ghosts(data, *fields):
    """Copy the named fields' values on ghost points and cells from the pieces that
    own them, in place, for a ``PartitionedDataSet`` that ``ghost_cells`` made.

    A filter that needs each entity's neighbours -- ``project`` to points, point
    gradients -- is right on what a piece owns and wrong on the ghost layer's
    rim; refreshing the ghosts makes a second such filter right too. Each named
    field is a point or cell field of explicit values on every piece. One
    exchange moves all the point fields, one all the cell fields."""
    plans = getattr(data, "_ghost_plans", None)
    if plans is None:
        raise ValueError("update_ghosts needs the PartitionedDataSet ghost_cells made, "
                         "which keeps the exchanges")
    pieces = list(data)
    blocks = plans["points"].blocks
    by_kind = {"points": {g: {} for g in blocks.local}, "cells": {g: {} for g in blocks.local}}
    for name in fields:
        for g, piece in zip(blocks.local, pieces):
            if name not in piece.fields:
                raise KeyError(f"partition {g - blocks.local.start} has no field {name!r}")
            field = piece.fields[name]
            kind = _kind(field)
            if kind is None:
                raise TypeError(f"update_ghosts refreshes point and cell fields; {name!r} "
                                f"is in {field.space!r}")
            if not isinstance(field.values, tack.Array):
                raise TypeError(f"update_ghosts writes the values in place; {name!r} holds "
                                f"a {type(field.values).__name__}, not a Tack array")
            by_kind[kind][g][name] = field.values
    for kind, values in by_kind.items():
        if any(values.values()):
            plans[kind].run(values)


# -- the pieces ------------------------------------------------------------

def _kind(field):
    space = field.space
    if (isinstance(space, H1) and space.order == 1) or \
            (isinstance(space, Values) and space.on == "points"):
        return "points"
    if isinstance(space, Constant) or (isinstance(space, Values) and space.on == "cells"):
        return "cells"
    return None


def _carried(pieces, fields):
    if not pieces:
        return []
    names = list(pieces[0].fields) if fields is None else \
        ([fields] if isinstance(fields, str) else list(fields))
    for i, piece in enumerate(pieces):
        for name in names:
            if name not in piece.fields:
                raise KeyError(f"partition {i} has no field {name!r}")
            if _kind(piece.fields[name]) is None:
                raise TypeError(f"ghost_cells carries point and cell fields; {name!r} is in "
                                f"{piece.fields[name].space!r} (choose others with fields=)")
    return names


def _check_piece(i, piece):
    if not isinstance(piece.topology, UnstructuredTopology):
        raise TypeError(f"ghost_cells takes unstructured pieces; partition {i} is "
                        f"{type(piece.topology).__name__} (split(..., ghost_layers=n) adds "
                        f"them to structured pieces)")
    marks = piece.ghosts
    if marks is not None and marks.cells is not None and \
            (marks.cells.numpy() & Ghosts.DUPLICATECELL).any():
        raise ValueError(f"partition {i} already has ghost cells; remove_ghosts first")


def _host(piece):
    """What the matching and assembly read of a piece, on the host."""
    topology = piece.topology
    positions = piece.positions()
    faces = topology.faces()
    rows = arrays.to_host(faces.rows).reshape(-1, 4)
    boundary = arrays.to_host(faces.boundary()).astype(np.int64)
    candidates = np.unique(rows[boundary].ravel())
    marks = piece.ghosts
    return {
        "num_points": piece.num_points,
        "num_cells": piece.num_cells,
        "positions": positions,
        "geometry_dtype": arrays.dtype_of(piece.geometry.values),
        "candidates": candidates[candidates >= 0].astype(np.int64),
        "types": np.asarray(arrays.to_host(topology.types), np.uint8).reshape(-1),
        "offsets": np.asarray(arrays.to_host(topology.offsets), np.int64).reshape(-1),
        "connectivity": np.asarray(arrays.to_host(topology.connectivity),
                                   np.int64).reshape(-1),
        "cell_marks": (np.zeros(piece.num_cells, np.uint8)
                       if marks is None or marks.cells is None else marks.cells.numpy()),
        "point_marks": (np.zeros(piece.num_points, np.uint8)
                        if marks is None or marks.points is None else marks.points.numpy()),
    }


def _position_keys(positions):
    """A 64-bit hash of each position's exact coordinates (-0 as 0)."""
    exact = np.ascontiguousarray(positions + 0.0)
    bits = exact.view(np.uint32 if exact.dtype == np.float32 else np.uint64).astype(np.uint64)
    bits = bits.reshape(len(positions), -1)
    key = np.zeros(len(positions), np.uint64)
    with np.errstate(over="ignore"):
        for column in bits.T:
            key = _mix(key ^ _mix(column + np.uint64(0x9E3779B97F4A7C15)))
    return key


def _mix(x):
    # splitmix64's finalizer
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


# -- matching --------------------------------------------------------------

def _match(blocks, pieces, hosts):
    """For each piece on this rank, its boundary points other pieces hold:
    ``(index, other piece, index there)`` rows, as three host arrays."""
    keys, values = {}, {}
    for g, piece in zip(blocks.local, pieces):
        host = hosts[g]
        chosen = host["candidates"]
        positions = np.asarray(host["positions"][chosen], np.float64) + 0.0   # -0 as 0
        device = blocks.device(g)
        keys[g] = tack.array(_position_keys(positions), tack.u64, device=device)
        values[g] = {
            "piece": tack.array(np.full(len(chosen), g, np.int64), tack.i64, device=device),
            "index": tack.array(chosen, tack.i64, device=device),
        }
        # The coordinates' bits, as u64: every device stores those, f64 or not.
        bits = np.ascontiguousarray(positions).view(np.uint64)
        for axis, name in enumerate("xyz"):
            values[g][name] = tack.array(bits[:, axis].copy(), tack.u64, device=device)
    sorted_keys, sorted_values = tc.sort_by_key(blocks, keys, values)

    outgoing = {}
    for g in blocks.local:
        found = _pairs(sorted_keys[g].numpy(),
                       *(sorted_values[g][n].numpy() for n in ("piece", "index", "x", "y", "z")))
        device = blocks.device(g)
        outgoing[g] = {}
        for piece in np.unique(found[0]).tolist():
            rows = found[0] == piece
            outgoing[g][int(piece)] = {
                name: tack.array(column[rows], tack.i64, device=device)
                for name, column in zip(("index", "other", "other_index"), found[1:])}
    incoming = blocks.exchange(outgoing)
    partners = {}
    for g in blocks.local:
        parts = list(incoming[g].values())
        partners[g] = tuple(np.concatenate([part[name].numpy() for part in parts])
                            if parts else np.zeros(0, np.int64)
                            for name in ("index", "other", "other_index"))
    return partners


def _pairs(keys, piece, index, x, y, z):
    """Rows ``(piece, index, other piece, other index)`` for every two points at one
    position on different pieces, from sorted keys and their points."""
    if not len(keys):
        return tuple(np.zeros(0, np.int64) for _ in range(4))
    table = np.stack([keys, x, y, z], axis=1).view(np.int64)
    _, group, sizes = np.unique(table, axis=0, return_inverse=True, return_counts=True)
    group = group.reshape(-1)
    order = np.argsort(group, kind="stable")
    group, piece, index = group[order], piece[order], index[order]
    starts = np.concatenate(([0], np.cumsum(sizes)))[group]
    size = sizes[group]
    first = np.repeat(np.arange(len(group)), size)
    second = np.repeat(starts, size) + (np.arange(size.sum()) - np.repeat(np.cumsum(size) - size,
                                                                          size))
    keep = piece[first] != piece[second]
    first, second = first[keep], second[keep]
    return piece[first], index[first], piece[second], index[second]


def _owners(g, num_points, partners):
    """Each point's identity, ``(owner << _INDEX_BITS) | index there``: the lowest
    piece holding it, and the point's index on that piece."""
    if num_points >= 1 << _INDEX_BITS:
        raise ValueError(f"ghost_cells handles pieces of fewer than 2**{_INDEX_BITS} points")
    identity = (np.int64(g) << _INDEX_BITS) | np.arange(num_points, dtype=np.int64)
    index, other, other_index = partners
    if len(index):
        theirs = (other.astype(np.int64) << _INDEX_BITS) | other_index
        np.minimum.at(identity, index, theirs)
    return identity


# -- what each piece sends -------------------------------------------------

def _ghost_parts(blocks, pieces, hosts, partners, owners, names):
    """Each piece's message to each neighbour: its cells using a point they share."""
    outgoing = {}
    for g, piece in zip(blocks.local, pieces):
        host = hosts[g]
        index, other, _ = partners[g]
        offsets, connectivity = host["offsets"], host["connectivity"]
        counts = np.diff(offsets)
        cell_of = np.repeat(np.arange(host["num_cells"]), counts)
        device = blocks.device(g)
        outgoing[g] = {}
        for neighbour in np.unique(other).tolist():
            shared = np.zeros(host["num_points"], bool)
            shared[index[other == neighbour]] = True
            cells = np.unique(cell_of[shared[connectivity]])
            used = np.concatenate([connectivity[offsets[c]:offsets[c + 1]] for c in cells])
            points, local = np.unique(used, return_inverse=True)
            sizes = counts[cells]
            message = {
                "types": tack.array(host["types"][cells], tack.u8, device=device),
                "sizes": tack.array(sizes.astype(np.int64), tack.i64, device=device),
                "connectivity": tack.array(local.astype(np.int64), tack.i64, device=device),
                "cells": tack.array(cells.astype(np.int64), tack.i64, device=device),
                "cell_marks": tack.array(host["cell_marks"][cells], tack.u8, device=device),
                "identity": tack.array(owners[g][points], tack.i64, device=device),
                "positions": tack.array(host["positions"][points],
                                        tack.vec3(host["geometry_dtype"]), device=device),
            }
            point_ids = tack.array(points, tack.i64, device=device)
            cell_ids = tack.array(cells.astype(np.int64), tack.i64, device=device)
            for name in names:
                field = piece.fields[name]
                chosen = point_ids if _kind(field) == "points" else cell_ids
                message["field." + name] = _take(field.values, chosen)
            outgoing[g][int(neighbour)] = message
    return outgoing


# -- what each piece makes of what it received -----------------------------

def _appended(g, piece, host, identity, received, names, blocks):
    """The piece with what its neighbours sent appended, and where its ghost points
    and cells come from: ``{owner: (owner's indices, local indices)}`` for each."""
    n_points, n_cells = host["num_points"], host["num_cells"]
    owner = identity >> _INDEX_BITS
    point_marks = host["point_marks"].copy()
    point_marks[owner != g] |= Ghosts.DUPLICATEPOINT

    # Received points, one per identity, after the piece's own: each mapped to
    # the piece's point of that identity, or to a new point, numbered in order
    # of first arrival (messages in order of source block).
    sources = list(received.items())
    sent = [message["identity"].numpy() for _, message in sources]
    starts = np.concatenate(([0], np.cumsum([len(ids) for ids in sent]))).astype(np.int64)
    arrived = np.concatenate(sent) if sent else np.zeros(0, np.int64)
    local = np.full(len(arrived), -1, np.int64)
    mine = (arrived >> _INDEX_BITS) == g
    local[mine] = arrived[mine] & ((1 << _INDEX_BITS) - 1)
    elsewhere = np.flatnonzero(owner != g)
    if len(elsewhere) and len(arrived):
        order = np.argsort(identity[elsewhere])
        table = identity[elsewhere][order]
        at = np.minimum(np.searchsorted(table, arrived), len(table) - 1)
        hit = ~mine & (table[at] == arrived)
        local[hit] = elsewhere[order][at[hit]]
    unknown = np.flatnonzero(local < 0)
    codes, first, which = np.unique(arrived[unknown], return_index=True, return_inverse=True)
    rank = np.empty(len(codes), np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(codes))
    local[unknown] = n_points + rank[which.reshape(-1)]
    new_from = unknown[np.sort(first)]                 # rows of the new points, in order
    new_identity = arrived[new_from]
    maps = [local[starts[m]:starts[m + 1]] for m in range(len(sources))]

    types, sizes, connectivity, cell_marks, cells_from = [host["types"]], \
        [np.diff(host["offsets"])], [host["connectivity"]], [host["cell_marks"]], {}
    for m, (source, message) in enumerate(sources):
        types.append(message["types"].numpy())
        sizes.append(message["sizes"].numpy())
        connectivity.append(maps[m][message["connectivity"].numpy()])
        cell_marks.append(message["cell_marks"].numpy() | Ghosts.DUPLICATECELL)
        start = n_cells + sum(len(t) for t in types[1:-1])
        cells_from[source] = (message["cells"].numpy(),
                              np.arange(start, start + len(types[-1]), dtype=np.int64))
    types, sizes = np.concatenate(types), np.concatenate(sizes)
    connectivity = np.concatenate(connectivity)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    total_points = n_points + len(new_identity)
    sent_positions = [message["positions"].numpy().reshape(-1, 3) for _, message in sources]
    new_positions = (np.concatenate(sent_positions)[new_from] if sources
                     else np.zeros((0, 3)))
    geometry_dtype = arrays.dtype_of(piece.geometry.values)
    positions = np.concatenate([host["positions"],
                                new_positions.astype(geometry_dtype.numpy_dtype)])
    point_marks = np.concatenate([point_marks,
                                  np.full(len(new_identity), Ghosts.DUPLICATEPOINT, np.uint8)])
    cell_marks = np.concatenate(cell_marks)

    # Each ghost point from its owner: the piece's own shared points owned
    # elsewhere, then the new ones.
    every_identity = np.concatenate([identity, new_identity])
    ghosts = np.flatnonzero((every_identity >> _INDEX_BITS) != g)
    points_from = {}
    for source in np.unique(every_identity[ghosts] >> _INDEX_BITS).tolist():
        mine = ghosts[(every_identity[ghosts] >> _INDEX_BITS) == source]
        points_from[int(source)] = (every_identity[mine] & ((1 << _INDEX_BITS) - 1), mine)

    with blocks.device(g):
        topology = UnstructuredTopology(types, offsets, connectivity, num_points=total_points,
                                        id_dtype=piece.topology.id_dtype)
        out = DataSet(topology, positions, dtype=geometry_dtype,
                      ghosts=Ghosts(cells=cell_marks, points=point_marks))
        for name in names:
            field = piece.fields[name]
            kind = _kind(field)
            parts = [field.values]
            if kind == "points" and len(new_from):
                # Each new point's values from the first message that brought it.
                sent = [message["field." + name] for _, message in sources]
                parts.append(_take(_joined(sent), tack.array(new_from, tack.i64)))
            elif kind == "cells":
                parts += [message["field." + name] for _, message in sources]
            values = _joined(parts)
            space = field.space
            if kind == "points":
                out.fields[name] = Field(H1(out) if isinstance(space, H1)
                                         else Values(out, "points"), values)
            else:
                out.fields[name] = Field(Constant(out) if isinstance(space, Constant)
                                         else Values(out, "cells"), values)
        for name, members in piece.sets.items():
            if members.on in ("points", "cells"):
                out.sets[name] = Set(out, members.on, members.ids)
    return out, points_from, cells_from


def _joined(parts):
    """``parts`` end to end: the first as it is when the others are empty."""
    kept = [arrays.materialize(p) for p in parts[1:] if arrays.size_of(p)]
    if not kept:
        return parts[0]
    return tack.concat([arrays.materialize(parts[0]), *kept])


def _plan(blocks, receives):
    """The plan giving each piece's ghosts the values at ``receives``:
    ``{gid: {source: (source's indices, local indices)}}``. Each source hears which
    of its indices to send, in order."""
    outgoing = {g: {source: {"wanted": tack.array(np.asarray(theirs, np.int64), tack.i64,
                                                  device=blocks.device(g))}
                    for source, (theirs, _) in receives[g].items() if len(theirs)}
                for g in blocks.local}
    asked = blocks.exchange(outgoing)
    sends = {g: {asker: message["wanted"] for asker, message in asked[g].items()}
             for g in blocks.local}
    local = {g: {source: np.asarray(mine, np.int64) for source, (theirs, mine)
                 in receives[g].items() if len(theirs)} for g in blocks.local}
    return tc.Plan(blocks, sends, local)

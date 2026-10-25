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
from tack.dataset import _matching
from tack.dataset._matching import INDEX_BITS, INDEX_MASK
from tack.dataset.algorithms._structured_ghosts import structured_ghost_cells
from tack.dataset.carry import _take
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology, UnstructuredTopology

__all__ = ["ghost_cells", "update_ghosts"]


def ghost_cells(data, layers=1, world=None, fields=None):
    """``data``'s pieces, each with a layer of its neighbours' cells added as ghosts:
    a new ``PartitionedDataSet``, as VTK's ``vtkGhostCellsGenerator`` makes.

    ``data`` holds this rank's pieces, all unstructured
    (``UnstructuredTopology``) or all structured (below), without ghost cells,
    on any devices; ``world`` is the
    ``tack.communication.World`` they live in (``data.world``, this process
    alone, by default). Pieces neighbour where they share points at exactly
    the same position. Each piece gains the cells of other pieces that use a
    point it shares with them (``DUPLICATECELL``), and their points it lacks
    (``DUPLICATEPOINT``); a point several pieces share belongs to the lowest
    of them, and is ``DUPLICATEPOINT`` in the others. Input marks come along:
    ``HIDDENCELL`` and the other bits are copied with each cell.

    Point and cell fields (``H1`` of order 1, ``Constant``, values on points
    or cells) are carried, ``fields`` choosing them as filters do (``None``:
    all); point and cell sets keep their members, and global ids come along
    where every piece has them, points then matched by their ids rather than
    their positions. ``update_ghosts`` refreshes
    a field's ghost values from their owners afterwards, after a filter that
    is right only where it sees every neighbour.

    ``layers`` layers: the first, every filter's need, is the cells that use
    a point a piece shares; each further layer the cells, of any piece, that use
    a point of the layer before (a corner's diagonal neighbours included), as
    ``split``'s ghost layers and ``vtkGhostCellsGenerator``'s grow. Pieces
    thinner than ``layers`` cells are fine: a layer reaches past them to the
    pieces beyond.

    Structured pieces (``StructuredTopology``, each placed by its ``offset``,
    together tiling a box of cells, as ``split``'s and VTK's image pieces do)
    stay structured: each becomes its block grown by ``layers`` cells, as far
    as the box goes, with its values from the pieces that own them, and its
    implicit coordinates grown with it.

    An ``AMRDataSet``'s boxes each grow by ``layers`` cells within their level's
    domain, the ghosts marked ``DUPLICATECELL`` and ``DUPLICATEPOINT``: a ghost a
    sibling box holds is its value, copied (AMReX's ``FillBoundary``); one no
    sibling holds takes the coarser level's -- the coarse cell's value for a
    cell, the coarse cell's corners' by linear weights for a point
    (``FillPatchTwoLevels``) -- and one neither holds is hidden. A grid's ghost
    positions are filled the same way, so rectilinear and curvilinear boxes
    grow too. Ghosts a finer box covers are marked refined. ``update_ghosts``
    refreshes them, ``average_down`` puts fine data into coarse cells.
    """
    if isinstance(layers, bool) or not isinstance(layers, (int, np.integer)) or layers < 1:
        raise ValueError(f"ghost_cells makes one or more layers, not {layers!r}")
    from tack.dataset.amr import AMRDataSet

    if isinstance(data, AMRDataSet):
        from tack.dataset.algorithms.amr_ghosts import amr_ghost_cells

        return amr_ghost_cells(data, int(layers), fields)
    if not isinstance(data, PartitionedDataSet):
        raise TypeError(f"ghost_cells takes a PartitionedDataSet, not {type(data).__name__}")
    world = data.world if world is None else world
    common_origin(data, "ghost_cells", world)       # positions are matched as they are
    pieces = list(data)
    names = _carried(pieces, fields)
    for i, piece in enumerate(pieces):
        _check_piece(i, piece)
    blocks = tc.Blocks([piece.device for piece in pieces], world)
    if blocks.count > 1 << (63 - INDEX_BITS):
        raise ValueError(f"ghost_cells handles up to {1 << (63 - INDEX_BITS)} pieces")
    structured = {g: int(isinstance(p.topology, StructuredTopology))
                  for g, p in zip(blocks.local, pieces)}
    every = blocks.all_gather(structured)
    if blocks.count and all(every):
        out, plans = structured_ghost_cells(blocks, pieces, int(layers), names, _kind)
        result = PartitionedDataSet(out, world=world)
        result._ghost_plans = plans
        return result
    if any(every):
        raise TypeError("ghost_cells takes pieces all structured or all unstructured, "
                        "not a mix")

    hosts = {g: _host(piece) for g, piece in zip(blocks.local, pieces)}
    # Global ids come along where every piece has them, and points are matched
    # by them then, by position otherwise.
    numbered = {kind: bool(blocks.all_reduce({
        g: int(p.global_ids is not None and getattr(p.global_ids, kind) is not None)
        for g, p in zip(blocks.local, pieces)}, "min")) if blocks.count else False
        for kind in ("points", "cells")}

    def values(g, piece):
        chosen = hosts[g]["candidates"]
        if numbered["points"]:
            return [piece.global_ids.points.numpy()[chosen].astype(np.uint64)]
        return _matching.position_bits(hosts[g]["positions"][chosen])

    partners = _matching.match(blocks, {
        g: (hosts[g]["candidates"], values(g, piece),
            np.zeros(len(hosts[g]["candidates"]), np.uint8))
        for g, piece in zip(blocks.local, pieces)})
    owners = {g: _matching.owners(g, hosts[g]["num_points"], partners[g],
                                  np.zeros(hosts[g]["num_points"], bool))
              for g in blocks.local}
    needed = _needed(blocks, hosts, partners, owners, int(layers))
    sent = _ghost_parts(blocks, pieces, hosts, needed, owners, names, numbered)
    incoming = blocks.exchange(sent)

    out, receive_points, receive_cells = [], {}, {}
    for g, piece in zip(blocks.local, pieces):
        result, points_from, cells_from = _appended(g, piece, hosts[g], owners[g],
                                                    incoming[g], names, numbered, blocks)
        out.append(result)
        receive_points[g], receive_cells[g] = points_from, cells_from
    point_plan = _matching.plan(blocks, receive_points)
    cell_plan = _matching.plan(blocks, receive_cells)
    result = PartitionedDataSet(out, world=world)
    result._ghost_plans = {"points": point_plan, "cells": cell_plan}
    return result


def update_ghosts(data, *fields):
    """Copy the named fields' values on ghost points and cells from the pieces that
    own them, in place, for a ``PartitionedDataSet`` that ``ghost_cells`` or
    ``ghost_points`` made.

    A filter that needs each entity's neighbours -- ``project`` to points, point
    gradients -- is right on what a piece owns and wrong on the ghost layer's
    rim; refreshing the ghosts makes a second such filter right too. Each named
    field is a point or cell field of explicit values on every piece. One
    exchange moves all the point fields, one all the cell fields."""
    from tack.dataset.amr import AMRDataSet

    if isinstance(data, AMRDataSet):
        from tack.dataset.algorithms.amr_ghosts import amr_update_ghosts

        amr_update_ghosts(data, fields)
        return
    plans = getattr(data, "_ghost_plans", None)
    if plans is None:
        raise ValueError("update_ghosts needs the PartitionedDataSet ghost_cells or "
                         "ghost_points made, which keeps the exchanges")
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
    if not isinstance(piece.topology, (UnstructuredTopology, StructuredTopology)):
        raise TypeError(f"ghost_cells takes unstructured or structured pieces; partition "
                        f"{i} is {type(piece.topology).__name__}")
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


# -- what each piece sends -------------------------------------------------

def _needed(blocks, hosts, partners, owners, layers):
    """The cells each piece sends each other piece, ``{gid: {other: cells}}`` for
    this rank's pieces, in ``layers`` rounds of identities only.

    A piece asks for the cells around its frontier -- first its points other
    pieces share, then the points of the cells the last round brought that it
    had not seen -- by the points' identities. Every piece offers its cells
    within ``layers`` steps of its boundary, each under its points' identities.
    Requests and offers of one identity meet in a sort over every block
    (``_matching.match``); each offered cell a piece has not been sent goes to
    it, and it hears the identities of that cell's points for the next round."""
    needed = {g: {} for g in blocks.local}
    seen, frontier, offers = {}, {}, {}
    for g in blocks.local:
        host = hosts[g]
        seen[g] = np.unique(owners[g])
        frontier[g] = np.unique(owners[g][np.unique(partners[g][0])])
        offers[g] = _offers(host, owners[g], layers)
    for _ in range(layers):
        entries = {}
        for g in blocks.local:
            cells, ids = offers[g]
            ask = frontier[g]
            index = np.concatenate([cells, -1 - np.arange(len(ask), dtype=np.int64)])
            values = np.concatenate([ids, ask]).astype(np.int64).view(np.uint64)
            flags = np.concatenate([np.zeros(len(cells), np.uint8), np.ones(len(ask), np.uint8)])
            entries[g] = (index, [values], flags)
        found = _matching.match(blocks, entries)
        outgoing = {}
        for g in blocks.local:
            index, other, _, other_flag = found[g]
            offered = (index >= 0) & (other_flag == 1)
            outgoing[g] = {}
            host = hosts[g]
            for b in np.unique(other[offered]).tolist():
                cells = np.unique(index[offered & (other == b)])
                cells = np.setdiff1d(cells, needed[g].get(b, np.zeros(0, np.int64)))
                if not len(cells):
                    continue
                needed[g][b] = np.union1d(needed[g].get(b, np.zeros(0, np.int64)), cells)
                points = np.concatenate([host["connectivity"][host["offsets"][c]:
                                                              host["offsets"][c + 1]]
                                         for c in cells])
                outgoing[g][int(b)] = {"ids": tack.array(owners[g][np.unique(points)],
                                                         tack.i64, device=blocks.device(g))}
        incoming = blocks.exchange(outgoing)
        for g in blocks.local:
            ids = [m["ids"].numpy() for m in incoming[g].values()]
            new = np.setdiff1d(np.unique(np.concatenate(ids)) if ids else
                               np.zeros(0, np.int64), seen[g])
            frontier[g] = new
            seen[g] = np.union1d(seen[g], new)
    return needed


def _offers(host, identity, layers):
    """A piece's cells within ``layers`` steps of its boundary, one entry per
    (cell, point): ``(cells, point identities)``."""
    offsets, connectivity = host["offsets"], host["connectivity"]
    counts = np.diff(offsets)
    cell_of = np.repeat(np.arange(host["num_cells"]), counts)
    reached = np.zeros(host["num_points"], bool)
    reached[host["candidates"]] = True
    chosen = np.zeros(host["num_cells"], bool)
    for _ in range(layers):
        touching = np.zeros(host["num_cells"], bool)
        touching[cell_of[reached[connectivity]]] = True
        chosen |= touching
        reached[connectivity[chosen[cell_of]]] = True
    rows = chosen[cell_of]
    return cell_of[rows].astype(np.int64), identity[connectivity[rows]].astype(np.int64)


def _ghost_parts(blocks, pieces, hosts, needed, owners, names, numbered):
    """Each piece's message to each other piece: the cells it needs (``_needed``)."""
    outgoing = {}
    for g, piece in zip(blocks.local, pieces):
        host = hosts[g]
        offsets, connectivity = host["offsets"], host["connectivity"]
        counts = np.diff(offsets)
        device = blocks.device(g)
        outgoing[g] = {}
        for neighbour, cells in sorted(needed[g].items()):
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
            for kind, chosen in (("points", point_ids), ("cells", cell_ids)):
                if numbered[kind]:
                    message["ids." + kind] = _take(getattr(piece.global_ids, kind), chosen)
            outgoing[g][int(neighbour)] = message
    return outgoing


# -- what each piece makes of what it received -----------------------------

def _appended(g, piece, host, identity, received, names, numbered, blocks):
    """The piece with what its neighbours sent appended, and where its ghost points
    and cells come from: ``{owner: (owner's indices, local indices)}`` for each."""
    n_points, n_cells = host["num_points"], host["num_cells"]
    owner = identity >> INDEX_BITS
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
    mine = (arrived >> INDEX_BITS) == g
    local[mine] = arrived[mine] & INDEX_MASK
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
    ghosts = np.flatnonzero((every_identity >> INDEX_BITS) != g)
    points_from = {}
    for source in np.unique(every_identity[ghosts] >> INDEX_BITS).tolist():
        mine = ghosts[(every_identity[ghosts] >> INDEX_BITS) == source]
        points_from[int(source)] = (every_identity[mine] & INDEX_MASK, mine)

    with blocks.device(g):
        topology = UnstructuredTopology(types, offsets, connectivity, num_points=total_points,
                                        id_dtype=piece.topology.id_dtype)
        out = DataSet(topology, positions, dtype=geometry_dtype,
                      ghosts=Ghosts(cells=cell_marks, points=point_marks), origin=piece.origin, unit=piece.unit)
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
                                         else Values(out, "points"), values, field.unit)
            else:
                out.fields[name] = Field(Constant(out) if isinstance(space, Constant)
                                         else Values(out, "cells"), values, field.unit)
        for name, members in piece.sets.items():
            if members.on in ("points", "cells"):
                out.sets[name] = Set(out, members.on, members.ids)
        ids = {}
        for kind in ("points", "cells"):
            if not numbered[kind]:
                continue
            parts = [getattr(piece.global_ids, kind)]
            sent = [message["ids." + kind] for _, message in sources]
            if kind == "points" and len(new_from):
                parts.append(_take(_joined(sent), tack.array(new_from, tack.i64)))
            elif kind == "cells":
                parts += sent
            ids[kind] = _joined(parts)
        if ids:
            out.global_ids = GlobalIds(cells=ids.get("cells"), points=ids.get("points"))
    return out, points_from, cells_from


def _joined(parts):
    """``parts`` end to end: the first as it is when the others are empty."""
    kept = [arrays.materialize(p) for p in parts[1:] if arrays.size_of(p)]
    if not kept:
        return parts[0]
    return tack.concat([arrays.materialize(parts[0]), *kept])

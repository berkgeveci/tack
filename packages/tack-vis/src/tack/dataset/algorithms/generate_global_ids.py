# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Global ids for the points and cells of a partitioned dataset's pieces."""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.dataset import _matching
from tack.dataset._matching import INDEX_BITS
from tack.dataset.algorithms._common import _tolerance
from tack.dataset.dataset import DataSet
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.partitioned import PartitionedDataSet, common_origin
from tack.dataset.topology import StructuredTopology

__all__ = ["generate_global_ids"]


def generate_global_ids(data, world=None, tolerance=0.0):
    """``data``'s pieces numbered across every piece, as VTK's
    ``vtkGenerateGlobalIds``: a new ``PartitionedDataSet`` of the same pieces with
    ``global_ids`` (``tack.dataset.GlobalIds``), every point and cell one id,
    the same on every piece holding it.

    Each piece numbers what it owns in its own order, after the pieces before
    it -- piece 0's owned points from 0, then piece 1's -- and copies take their
    owner's ids. Owners come from the ghost marks where pieces have them (a
    copy marked ``DUPLICATEPOINT`` or ``DUPLICATECELL`` is another piece's);
    a point several pieces hold unmarked, as pieces without ghost cells share
    their boundary points, belongs to the lowest of them, and is marked
    ``DUPLICATEPOINT`` in the others, as VTK marks it. Points are copies of
    each other when their positions are exactly equal -- or, with
    ``tolerance``, at most that far apart (in the spatial unit, or a
    quantity), chains of them too, for pieces whose copies of a shared point
    differ in their last bits -- and cells when their points' global ids are;
    pieces from ``ghost_cells`` reuse the exchanges it kept instead of
    matching again. Points a piece repeats within itself stay apart:
    ``merge_points`` merges them first.

    A filter's output from pieces whose marks it could not carry (pieces
    without ghost cells; see ``owned_values``) is marked here exactly: every
    point is matched, so a point several pieces made belongs to the lowest of
    them and one only a single piece made is its own, and a cell several
    pieces made (from a ghost cell and its owner, or an edge they share) is
    matched by its points' ids, the lowest piece's again. The output then has
    marks, and reductions take it.

    ``data`` holds this rank's pieces, on any devices; ``world`` is the
    ``tack.communication.World`` they live in (``data.world`` by default).
    The matching runs on the host and the exchanges on the pieces' devices.
    """
    if not isinstance(data, PartitionedDataSet):
        raise TypeError(f"generate_global_ids takes a PartitionedDataSet, not "
                        f"{type(data).__name__}")
    world = data.world if world is None else world
    common_origin(data, "generate_global_ids", world)
    tolerance = _tolerance(data, tolerance, "generate_global_ids")
    pieces = list(data)
    blocks = tc.Blocks([piece.device for piece in pieces], world)
    plans = getattr(data, "_ghost_plans", None)
    # Pieces that lost their marks keep only their cells' (from source cells):
    # every point is matched, and every cell not marked a copy.
    lost = bool(blocks.all_reduce({g: int(bool(piece._ghosts_lost))
                                   for g, piece in zip(blocks.local, pieces)}, "max"))
    if lost:
        plans = None
    marks = {g: _marks(piece, lost) for g, piece in zip(blocks.local, pieces)}

    # Points: who owns each, then the owners' numbers, then the copies'.
    if plans is not None:
        point_plan = plans["points"]
        point_owned = {g: (marks[g][1] & Ghosts.DUPLICATEPOINT) == 0 for g in blocks.local}
        point_shared = None
    else:
        point_owned, point_plan, point_shared = _point_owners(blocks, pieces, marks,
                                                                 tolerance, every=lost)
    point_ids = _numbered(blocks, pieces, point_owned, point_plan, "points")

    # Cells: owned cells numbered, ghost cells from their owners.
    cell_owned = {g: (marks[g][0] & Ghosts.DUPLICATECELL) == 0 for g in blocks.local}
    copies = blocks.all_reduce({g: int((~cell_owned[g]).sum()) for g in blocks.local})
    if plans is not None:
        cell_plan = plans["cells"]
    elif lost:
        cell_owned, cell_plan = _matched_cells(blocks, pieces, marks, point_ids, point_shared)
    elif copies:
        cell_plan = _cell_owners(blocks, pieces, marks, point_ids, point_shared)
    else:
        cell_plan = None
    cell_ids = _numbered(blocks, pieces, cell_owned, cell_plan, "cells")

    out = []
    for g, piece in zip(blocks.local, pieces):
        points = marks[g][1] | np.where(point_owned[g], 0,
                                        Ghosts.DUPLICATEPOINT).astype(np.uint8)
        with blocks.device(g):
            ghosts = piece.ghosts
            if lost:                                        # every owner decided here
                cells = marks[g][0] | np.where(cell_owned[g], 0,
                                               Ghosts.DUPLICATECELL).astype(np.uint8)
                ghosts = Ghosts(cells=cells, points=points)
            elif not np.array_equal(points, marks[g][1]):   # copies newly marked
                ghosts = Ghosts(cells=None if ghosts is None else ghosts.cells, points=points)
            copy = DataSet(piece.topology, piece.geometry, fields=dict(piece.fields),
                           sets=dict(piece.sets), dtype=piece.dtype, ghosts=ghosts,
                           global_ids=GlobalIds(cells=cell_ids[g], points=point_ids[g]),
                           **piece._frame())
        copy._ghosts_lost = False if lost else piece._ghosts_lost
        out.append(copy)
    result = PartitionedDataSet(out, world=world)
    if plans is not None:
        result._ghost_plans = plans
    return result


def _marks(piece, lost=False):
    """A piece's cell and point marks on the host, zeros where it has none (or,
    when it ``lost`` them, its points')."""
    ghosts = piece.ghosts
    cells = None if ghosts is None else ghosts.cells
    points = None if ghosts is None or lost else ghosts.points
    return (np.zeros(piece.num_cells, np.uint8) if cells is None else cells.numpy(),
            np.zeros(piece.num_points, np.uint8) if points is None else points.numpy())


def _boundary_points(topology):
    """The points on ``topology``'s boundary faces, sorted: from the faces' rows (four
    ids a face, padded with -1), or a polyhedral topology's faces, of any size."""
    faces = topology.faces()
    boundary = arrays.to_host(faces.boundary()).astype(np.int64)
    if hasattr(faces, "rows"):
        chosen = arrays.to_host(faces.rows).reshape(-1, 4)[boundary]
    else:
        from tack.dataset.polyhedra import _ranges

        offsets = arrays.to_host(topology.face_offsets).astype(np.int64)
        chosen = arrays.to_host(topology.face_points)[
            _ranges(offsets[boundary], offsets[boundary + 1] - offsets[boundary])]
    chosen = np.unique(chosen)
    return chosen[chosen >= 0].astype(np.int64)


def _point_owners(blocks, pieces, marks, tolerance=0.0, every=False):
    """Each piece's owned points (a bool each), the plan bringing its other points'
    ids from their owners, and which of its points another piece also holds.
    Only boundary points are matched, unless pieces have copies marked (points,
    or cells, whose points are copies too) or ``every`` point is to be."""
    duplicates = every or blocks.all_reduce(
        {g: int((marks[g][1] & Ghosts.DUPLICATEPOINT).any()
                or (marks[g][0] & Ghosts.DUPLICATECELL).any()) for g in blocks.local}, "max")
    entries = {}
    for g, piece in zip(blocks.local, pieces):
        if duplicates:
            # Ghost layers: a piece's copies of other pieces' points are anywhere
            # in it, so every point is matched.
            chosen = np.arange(piece.num_points, dtype=np.int64)
        else:
            chosen = _boundary_points(piece.topology)
        flags = (marks[g][1][chosen] & Ghosts.DUPLICATEPOINT).astype(np.uint8)
        entries[g] = (chosen, piece.positions()[chosen], flags)
    partners = _matching.match_positions(blocks, entries, tolerance)
    identities = {g: _matching.owners(g, piece.num_points, partners[g],
                                      (marks[g][1] & Ghosts.DUPLICATEPOINT) != 0)
                  for g, piece in zip(blocks.local, pieces)}
    if tolerance:
        _matching.settled(blocks, identities, partners)
    owned, receives, shared = {}, {}, {}
    for g, piece in zip(blocks.local, pieces):
        identity = identities[g]
        owned[g] = (identity >> INDEX_BITS) == g
        receives[g] = _matching.from_owners(g, identity, np.flatnonzero(~owned[g]))
        shared[g] = np.zeros(piece.num_points, bool)
        shared[g][partners[g][0]] = True
    return owned, _matching.plan(blocks, receives), shared


def _cell_owners(blocks, pieces, marks, point_ids, shared):
    """The plan bringing each piece's ghost cells' ids from their owners, cells
    matched by their points' global ids."""
    ghosts = {g: (marks[g][0] & Ghosts.DUPLICATECELL) != 0 for g in blocks.local}
    partners = _matching.match(blocks, _cell_entries(blocks, pieces, ghosts, point_ids,
                                                     shared))
    receives = {}
    for g, piece in zip(blocks.local, pieces):
        identity = _matching.owners(g, piece.num_cells, partners[g], ghosts[g])
        receives[g] = _matching.from_owners(g, identity, np.flatnonzero(ghosts[g]))
    return _matching.plan(blocks, receives)


def _matched_cells(blocks, pieces, marks, point_ids, shared):
    """Each piece's owned cells and the plan bringing the others' ids, for outputs
    that lost their marks: a cell marked a copy is its owner's, and an unmarked
    cell of the same points in several pieces the lowest's."""
    ghosts = {g: (marks[g][0] & Ghosts.DUPLICATECELL) != 0 for g in blocks.local}
    partners = _matching.match(blocks, _cell_entries(blocks, pieces, ghosts, point_ids,
                                                     shared))
    owned, receives = {}, {}
    for g, piece in zip(blocks.local, pieces):
        identity = _matching.owners(g, piece.num_cells, partners[g], ghosts[g])
        owned[g] = (identity >> INDEX_BITS) == g
        receives[g] = _matching.from_owners(g, identity, np.flatnonzero(~owned[g]))
    return owned, _matching.plan(blocks, receives)


def _cell_entries(blocks, pieces, ghosts, point_ids, shared):
    """Each piece's cells that may be copies (marked ``ghosts``, or every point
    shared) keyed by their points' global ids, for ``_matching.match``."""
    entries = {}
    for g, piece in zip(blocks.local, pieces):
        offsets, connectivity = _cell_rows(piece.topology)
        ids = point_ids[g].numpy()
        ghost = ghosts[g]
        # A cell that is a copy elsewhere has every point there too.
        counts = np.diff(offsets)
        every_shared = np.add.reduceat(shared[g][connectivity].astype(np.int64),
                                       offsets[:-1]) == counts if len(counts) else \
            np.zeros(0, bool)
        chosen = np.flatnonzero(ghost | every_shared).astype(np.int64)
        columns = [np.zeros(len(chosen), np.uint64) for _ in range(4)]
        for size in np.unique(counts[chosen]).tolist():
            of_size = np.flatnonzero(counts[chosen] == size)
            cells = chosen[of_size]
            rows = connectivity[offsets[cells][:, None] + np.arange(size)]
            sorted_ids = np.sort(ids[rows], axis=1).astype(np.uint64)
            listed = list(sorted_ids.T)
            columns[0][of_size] = np.uint64(size)
            columns[1][of_size] = _matching.hashed(listed, seed=1)
            columns[2][of_size] = sorted_ids[:, 0]
            columns[3][of_size] = sorted_ids[:, -1]
        entries[g] = (chosen, columns, ghost[chosen].astype(np.uint8))
    return entries


def _cell_rows(topology):
    """Each cell's points on the host, CSR ``(offsets, members)``, in any order:
    a structured grid's corners by arithmetic, x fastest as it numbers them."""
    if isinstance(topology, StructuredTopology):
        dims = list(topology.point_dims)
        cells = [d - 1 for d in dims]
        index = [i.ravel(order="F") for i in np.meshgrid(*map(np.arange, cells), indexing="ij")]
        strides = np.cumprod([1, *dims[:-1]])
        base = sum(i * s for i, s in zip(index, strides))
        corners = [sum(c * s for c, s in zip(corner, strides))
                   for corner in np.ndindex(*([2] * len(dims)))]
        members = (base[:, None] + np.asarray(corners)[None, :]).astype(np.int64)
        return (np.arange(len(base) + 1, dtype=np.int64) * len(corners), members.reshape(-1))
    from tack.dataset.ghosts import _rows
    offsets, members = _rows(topology)
    return (np.asarray(arrays.to_host(offsets), np.int64).reshape(-1),
            np.asarray(arrays.to_host(members), np.int64).reshape(-1))


def _numbered(blocks, pieces, owned, plan, what):
    """Each piece's ids: its owned entities numbered after the pieces before it, the
    others brought from their owners by ``plan``."""
    offsets, _ = blocks.exclusive_scan({g: int(owned[g].sum()) for g in blocks.local})
    ids = {}
    for g, piece in zip(blocks.local, pieces):
        n = owned[g].size
        host = np.full(n, -1, np.int64)
        host[owned[g]] = offsets[g] + np.arange(int(owned[g].sum()), dtype=np.int64)
        ids[g] = tack.array(host, tack.i64, device=blocks.device(g))
    if plan is not None:
        plan.run(ids)
    for g in blocks.local:
        if n_missing := int((ids[g].numpy() < 0).sum()):
            raise ValueError(f"piece {g - blocks.local.start}: {n_missing} {what} have no "
                             f"owner to number them")
    return ids

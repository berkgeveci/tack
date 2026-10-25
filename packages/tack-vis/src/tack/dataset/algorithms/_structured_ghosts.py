"""Ghost layers for structured pieces, kept structured: ``ghost_cells``' path for
pieces that are all ``StructuredTopology``.

Each piece is a block of a larger grid: its cells start at ``topology.offset``
there. The pieces must tile a box of cells (VTK's image pieces and ``split``'s
do). A piece becomes the block grown by ``layers`` cells on every side, as far
as the box goes, with a new offset. Everything else is arithmetic on global
indices: a cell belongs to the piece whose block holds it, a point to the
lowest piece whose block holds it (``split``'s rule, and VTK's), and a
piece's ghosts take their values from those owners through two
``tack.communication.Plan``s -- the same ones ``update_ghosts`` runs later.
Uniform coordinates move their origin; rectilinear ones take the axes every
piece gives; explicit positions travel as a point field.
"""

import math

import numpy as np

import tack
from tack import arrays
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.communication._kernels import scatter
from tack.dataset import _matching
from tack.dataset.algorithms.split import _block_geometry
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.global_ids import GlobalIds
from tack.dataset.sets import Set
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology

__all__ = ["structured_ghost_cells"]


def structured_ghost_cells(blocks, pieces, layers, names, kind_of):
    """The pieces grown by ``layers`` cells, and the plans of their ghosts:
    ``(pieces, {"points": Plan, "cells": Plan})``. ``kind_of(field)`` says
    whether a field is on ``"points"`` or ``"cells"``."""
    dims = {len(p.topology.point_dims) for p in pieces}
    every = blocks.all_gather({g: (tuple(p.topology.offset),
                                   tuple(d - 1 for d in p.topology.point_dims))
                               for g, p in zip(blocks.local, pieces)})
    if len({len(o) for o, _ in every} | dims) > 1:
        raise ValueError("ghost_cells needs structured pieces of one dimension")
    lower = np.array([o for o, _ in every], np.int64)
    upper = lower + np.array([c for _, c in every], np.int64)
    whole_lower, whole_upper = lower.min(axis=0), upper.max(axis=0)
    if int(np.prod(upper - lower, axis=1).sum()) != int(np.prod(whole_upper - whole_lower)) \
            or _overlap(lower, upper):
        raise ValueError("ghost_cells needs structured pieces that tile a box of cells "
                         "(their offsets and sizes, as split and VTK's image pieces give)")
    grown_lower = np.maximum(lower - layers, whole_lower)
    grown_upper = np.minimum(upper + layers, whole_upper)

    def flat(local, extent):
        # x fastest
        index = np.zeros(len(local), np.int64)
        for axis in reversed(range(local.shape[1])):
            index = index * extent[axis] + local[:, axis]
        return index

    def grid(lo, hi):
        """The global indices from ``lo`` to ``hi`` (inclusive), x fastest."""
        axes = [np.arange(a, b + 1) for a, b in zip(lo, hi)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel(order="F") for m in mesh], axis=1).astype(np.int64)

    def point_owner(points):
        owner = np.full(len(points), -1, np.int64)
        for d in range(blocks.count):
            inside = np.all((points >= lower[d]) & (points <= upper[d]), axis=1)
            owner[inside & (owner < 0)] = d
        return owner

    def cell_owner(cells):
        owner = np.full(len(cells), -1, np.int64)
        for d in range(blocks.count):
            owner[np.all((cells >= lower[d]) & (cells < upper[d]), axis=1)] = d
        return owner

    receive = {"points": {}, "cells": {}}
    layout = {}
    for g in blocks.local:
        point_extent = grown_upper[g] - grown_lower[g] + 1
        cell_extent = grown_upper[g] - grown_lower[g]
        points = grid(grown_lower[g], grown_upper[g])
        cells = grid(grown_lower[g], grown_upper[g] - 1)
        p_owner, c_owner = point_owner(points), cell_owner(cells)
        own_points = np.flatnonzero(np.all((points >= lower[g]) & (points <= upper[g]), axis=1))
        own_cells = np.flatnonzero(c_owner == g)
        layout[g] = (point_extent, cell_extent, p_owner, c_owner, own_points, own_cells)
        for kind, entities, owner in (("points", points, p_owner), ("cells", cells, c_owner)):
            receive[kind][g] = {}
            for d in np.unique(owner[owner != g]).tolist():
                rows = np.flatnonzero(owner == d)
                extent = (grown_upper[d] - grown_lower[d] + (1 if kind == "points" else 0))
                theirs = flat(entities[rows] - grown_lower[d], extent)
                receive[kind][g][int(d)] = (theirs, rows)
    plans = {kind: _matching.plan(blocks, receive[kind]) for kind in ("points", "cells")}

    rectilinear = _axes(blocks, pieces, lower)
    grown = {}
    for g, piece in zip(blocks.local, pieces):
        point_extent, cell_extent, p_owner, c_owner, own_points, own_cells = layout[g]
        np_, nc = int(np.prod(point_extent)), int(np.prod(cell_extent))
        with blocks.device(g):
            topology = StructuredTopology(tuple(int(d) for d in point_extent),
                                          id_dtype=piece.topology.id_dtype,
                                          offset=tuple(int(v) for v in grown_lower[g]))
            spread = {}                       # the arrays the plans fill, per kind
            point_index = tack.array(own_points, tack.i64)
            cell_index = tack.array(own_cells, tack.i64)

            def placed(values, n, index):
                values = arrays.materialize(values)
                out = tack.empty((n,), values.dtype)
                scatter(values, index, out)
                return out

            for name in names:
                field = piece.fields[name]
                if kind_of(field) == "points":
                    spread.setdefault("points", {})[name] = placed(field.values, np_, point_index)
                else:
                    spread.setdefault("cells", {})[name] = placed(field.values, nc, cell_index)
            marks = piece.ghosts
            spread.setdefault("cells", {})["\0marks"] = placed(
                tack.zeros((piece.num_cells,), tack.u8) if marks is None or marks.cells is None
                else marks.cells, nc, cell_index)
            spread.setdefault("points", {})["\0marks"] = placed(
                tack.zeros((piece.num_points,), tack.u8) if marks is None or marks.points is None
                else marks.points, np_, point_index)
            ids = piece.global_ids
            for kind, index, n in (("points", point_index, np_), ("cells", cell_index, nc)):
                if ids is not None and getattr(ids, kind) is not None:
                    spread[kind]["\0ids"] = placed(getattr(ids, kind), n, index)
            geometry = piece.geometry.values
            implicit = isinstance(geometry, (UniformCoordinates, CartesianProduct))
            if not implicit:
                spread["points"]["\0positions"] = placed(geometry, np_, point_index)
        grown[g] = (piece, topology, spread, implicit, p_owner, c_owner, own_points, own_cells)

    for kind in ("points", "cells"):
        plans[kind].run({g: grown[g][2][kind] for g in blocks.local})

    out = []
    for g in blocks.local:
        piece, topology, spread, implicit, p_owner, c_owner, own_points, own_cells = grown[g]
        with blocks.device(g):
            if implicit:
                geometry = _grown_geometry(piece, rectilinear, grown_lower[g], topology,
                                           lower[g], blocks.device(g))
            else:
                geometry = spread["points"]["\0positions"]
            point_marks = spread["points"]["\0marks"].numpy() | np.where(
                p_owner != g, Ghosts.DUPLICATEPOINT, 0).astype(np.uint8)
            cell_marks = spread["cells"]["\0marks"].numpy() | np.where(
                c_owner != g, Ghosts.DUPLICATECELL, 0).astype(np.uint8)
            data = DataSet(topology, geometry, dtype=piece.dtype,
                           ghosts=Ghosts(cells=cell_marks, points=point_marks),
                           origin=piece.origin)
            for name in names:
                field = piece.fields[name]
                space = field.space
                if kind_of(field) == "points":
                    data.fields[name] = Field(H1(data) if isinstance(space, H1)
                                              else Values(data, "points"),
                                              spread["points"][name])
                else:
                    data.fields[name] = Field(Constant(data) if isinstance(space, Constant)
                                              else Values(data, "cells"), spread["cells"][name])
            if "\0ids" in spread["points"] or "\0ids" in spread["cells"]:
                data.global_ids = GlobalIds(cells=spread["cells"].get("\0ids"),
                                            points=spread["points"].get("\0ids"))
            for name, members in piece.sets.items():
                if members.on == "points":
                    data.sets[name] = Set(data, "points", own_points[members.ids.numpy()])
                elif members.on == "cells":
                    data.sets[name] = Set(data, "cells", own_cells[members.ids.numpy()])
        data._ghosts_lost = piece._ghosts_lost
        out.append(data)
    return out, plans


def _overlap(lower, upper):
    for a in range(len(lower)):
        for b in range(a + 1, len(lower)):
            if np.all(np.maximum(lower[a], lower[b]) < np.minimum(upper[a], upper[b])):
                return True
    return False


def _axes(blocks, pieces, lower):
    """For rectilinear pieces, each topology axis's coordinates over the whole box,
    ``{axis: {global index: coordinate}}``, from every piece's own axes; ``None``
    when the pieces are not rectilinear."""
    def mine(piece, offset):
        coordinates = piece.geometry.values
        if not isinstance(coordinates, CartesianProduct):
            return None
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        lines = (coordinates.x, coordinates.y, coordinates.z)
        return {i: (axis, int(offset[i]), lines[axis].numpy().astype(np.float64))
                for i, axis in enumerate(axes)}
    every = blocks.all_gather({g: mine(p, p.topology.offset)
                               for g, p in zip(blocks.local, pieces)})
    if any(e is None for e in every):
        return None
    out = {}
    for part in every:
        for i, (axis, start, values) in part.items():
            line = out.setdefault(i, (axis, {}))[1]
            for k, v in enumerate(values.tolist()):
                line.setdefault(start + k, v)
    return out


def _grown_geometry(piece, rectilinear, grown_lower, topology, lower, device):
    """The implicit coordinates of a piece grown to ``topology``'s block."""
    coordinates = piece.geometry.values
    point_dims = topology.point_dims
    if isinstance(coordinates, UniformCoordinates):
        axes = [a for a, d in enumerate(coordinates.dims) if d != 1]
        return _block_geometry(coordinates, axes, list(grown_lower - lower), point_dims,
                               device)
    lines = [np.zeros(1), np.zeros(1), np.zeros(1)]
    for i, d in enumerate(point_dims):
        axis, line = rectilinear[i]
        lines[axis] = np.array([line[int(grown_lower[i]) + k] for k in range(d)])
    for axis, values in enumerate((coordinates.x, coordinates.y, coordinates.z)):
        if math.prod(lines[axis].shape) == 1 and values.shape[0] == 1:
            lines[axis] = values.numpy().astype(np.float64)
    return CartesianProduct(*lines, dtype=arrays.dtype_of(coordinates), device=device)

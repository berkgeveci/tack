"""Ghost cells of an AMR hierarchy, and fine data averaged down onto coarse.

Every value a box holds but does not own is a weighted sum of values other
boxes own: a ghost cell or point a sibling box holds is that box's value
(weight 1, AMReX's ``FillBoundary``); one no sibling holds takes the coarser
level's -- the coarse cell's value for a cell (injection), the coarse cell's
corners by linear weights for a point (``FillPatchTwoLevels``) -- and a coarse
cell a finer level covers takes its fine cells' average (``average_down``).
So each is a ``_Fill``: a ``tack.communication.Plan`` that brings the sources'
values next to each destination box, over devices and MPI ranks, and one
kernel that sums them by weight. A grid's positions are filled the same way,
so rectilinear and curvilinear boxes get their ghost layers' geometry from
their siblings, or interpolated in the coarse cells.
"""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.arrays import CartesianProduct, UniformCoordinates
from tack.communication._kernels import take
from tack.dataset import _matching
from tack.dataset.amr import AMRDataSet
from tack.dataset.carry import _point_values
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import common_origin
from tack.dataset.spaces import H1, Constant, Values
from tack.dataset.topology import StructuredTopology

__all__ = ["average_down"]


@tack.kernel
def _apply(staged, offsets, entries, weights, targets, values):
    # Each target's value: its sources' staged values, summed by weight.
    for j in range(targets.shape[0]):
        first = offsets[j]
        total = staged[entries[first]] * weights[first]
        for e in range(first + 1, offsets[j + 1]):
            total += staged[entries[e]] * weights[e]
        values[targets[j]] = total


@tack.kernel
def _pick(staged, picks, targets, values):
    # An integer value: its weightiest source's, unchanged.
    for j in range(targets.shape[0]):
        values[targets[j]] = staged[picks[j]]


@tack.kernel
def _divide(sums, measures, targets, values):
    for j in range(targets.shape[0]):
        values[targets[j]] = sums[targets[j]] / measures[targets[j]]


class _Fill:
    """Values of some entries of each box (``targets``) as weighted sums of other
    boxes' entries: a ``Plan`` staging the sources after each box's own values,
    and per box CSR rows of ``(staging slot, weight)``."""

    def __init__(self, blocks, rows):
        """``rows[g]``: ``(n, targets, sources)``, the box's array length, its target
        indices, and per target a list of ``(source block, source index, weight)``."""
        self.blocks = blocks
        self.boxes = {}
        receives = {}
        for g in blocks.local:
            n, targets, sources = rows[g]
            slots, keys = {}, []
            offsets, entries, weights, picks, kept = [0], [], [], [], []
            for target, row in zip(targets, sources):
                if not row:
                    continue
                for source, index, weight in row:
                    key = (int(source), int(index))
                    if key not in slots:
                        slots[key] = len(keys)
                        keys.append(key)
                    entries.append(slots[key])
                    weights.append(float(weight))
                best = max(range(len(row)), key=lambda e: row[e][2])
                picks.append(slots[(int(row[best][0]), int(row[best][1]))])
                offsets.append(len(entries))
                kept.append(int(target))
            by_source = {}
            for slot, (source, index) in enumerate(keys):
                by_source.setdefault(source, ([], []))
                by_source[source][0].append(index)
                by_source[source][1].append(n + slot)
            receives[g] = {source: (np.asarray(a, np.int64), np.asarray(b, np.int64))
                           for source, (a, b) in by_source.items()}
            self.boxes[g] = {"n": n, "m": len(keys), "targets": np.asarray(kept, np.int64),
                             "offsets": np.asarray(offsets, np.int64),
                             "entries": np.asarray(entries, np.int64),
                             "weights": np.asarray(weights, np.float64),
                             "picks": np.asarray(picks, np.int64)}
        self.plan = _matching.plan(blocks, receives)

    def run(self, values, combine=True):
        """Fill each box's targets in ``values`` (``{g: array}``, the boxes' arrays of
        one kind), in place. With ``combine`` false, integer arrays included, each
        target takes its weightiest source's value."""
        blocks = self.blocks
        combined = {}
        for g in blocks.local:
            box, array = self.boxes[g], values[g]
            with blocks.device(g):
                staging = tack.zeros((box["m"],), array.dtype)
                combined[g] = tack.concat([array, staging]) if box["m"] else array
        self.plan.run(combined)
        for g in blocks.local:
            box, array = self.boxes[g], values[g]
            if not len(box["targets"]):
                continue
            with blocks.device(g):
                staged = take(combined[g], box["n"], box["m"])
                targets = tack.array(box["targets"], tack.i64)
                scalar = getattr(array.dtype, "scalar_dtype", array.dtype)
                if combine and scalar.numpy_dtype.kind == "f":
                    _apply(staged, tack.array(box["offsets"], tack.i64),
                           tack.array(box["entries"], tack.i64),
                           tack.array(box["weights"].astype(scalar.numpy_dtype), scalar),
                           targets, array)
                else:
                    _pick(staged, tack.array(box["picks"], tack.i64), targets, array)


# -- where boxes are ------------------------------------------------------------------

class _Layout:
    """Every box's block id and array extent, on every rank."""

    def __init__(self, amr):
        self.amr = amr
        local = [box for _, box in amr]
        self.blocks = tc.Blocks([box.device for box in local], amr.world)
        counts = amr.world._allgather([len(level) for level in amr.levels])
        self.counts = np.asarray(counts, np.int64).reshape(len(counts), -1)
        self.prefix = np.concatenate([np.zeros((1, amr.num_levels), np.int64),
                                      np.cumsum(self.counts, axis=0)])
        self.local = [(level, amr._first[level] + k) for level, pieces in enumerate(amr.levels)
                      for k in range(len(pieces))]

    def gid(self, level, k):
        """The block id of global box ``k`` of ``level``."""
        rank = int(np.searchsorted(self.prefix[:, level], k, side="right")) - 1
        position = int(self.counts[rank, :level].sum()) + int(k - self.prefix[rank, level])
        return int(self.blocks._starts[rank]) + position


def _cell_index(extent, q, d):
    """Flat indices of cells ``q`` (``(m, d)``) in a box's arrays of ``extent``."""
    lo, hi = extent[0, :d], extent[1, :d]
    n = hi - lo
    local = q - lo
    flat = np.zeros(len(q), np.int64)
    for a in range(d - 1, -1, -1):
        flat = flat * n[a] + local[:, a]
    return flat


def _point_index(extent, p, d):
    lo, hi = extent[0, :d], extent[1, :d]
    n = hi - lo + 1
    local = p - lo
    flat = np.zeros(len(p), np.int64)
    for a in range(d - 1, -1, -1):
        flat = flat * n[a] + local[:, a]
    return flat


def _grid(lo, hi, d):
    """Every index of the box ``[lo, hi)`` (``d`` axes), x fastest: ``(m, d)``."""
    axes = [np.arange(lo[a], hi[a]) for a in range(d)]
    mesh = np.meshgrid(*axes[::-1], indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh[::-1]], axis=1).astype(np.int64)


def _inside(index, lo, hi):
    """Which rows of ``index`` lie in ``[lo, hi)``."""
    return np.all((index >= lo) & (index < hi), axis=1)


# -- ghost cells ----------------------------------------------------------------------

def amr_ghost_cells(amr, layers, fields):
    """``ghost_cells`` of an ``AMRDataSet``: each box grown by ``layers`` cells within
    its level's domain, its ghosts filled from its siblings or the coarser level."""
    from tack.dataset.algorithms.ghost_cells import _carried, _kind

    if amr.layers:
        raise ValueError("the hierarchy already has ghost cells; give ghost_cells the "
                         "hierarchy without them")
    d = amr.dimension
    pieces = [box for _, box in amr]
    names = _carried(pieces, fields)
    common_origin(amr._as_pieces(), "ghost_cells of an AMRDataSet")
    extents = []
    for level in range(amr.num_levels):
        lo, hi = amr.level_domain(level)
        grown = amr._boxes[level].copy()
        grown[:, 0, :d] = np.maximum(grown[:, 0, :d] - layers, lo[:d])
        grown[:, 1, :d] = np.minimum(grown[:, 1, :d] + layers, hi[:d])
        extents.append(grown)
    layout = _Layout(amr)
    cell_rows, point_rows, hidden = {}, {}, {}
    for g, (level, k) in zip(layout.blocks.local, layout.local):
        cell_rows[g], point_rows[g], hidden[g] = _ghost_rows(amr, layout, extents, level, k)
    fills = {"cells": _Fill(layout.blocks, cell_rows),
             "points": _Fill(layout.blocks, point_rows)}

    # The grown boxes: own values placed, ghosts filled; positions too, for grids
    # whose positions are not an image's.
    grown_boxes, arrays_by_kind, positions = [], {}, {}
    for g, (level, k), piece in zip(layout.blocks.local, layout.local, pieces):
        box, ext = amr._boxes[level][k], extents[level][k]
        with piece.device:
            assembled = {name: _placed(piece.fields[name].values, _kind(piece.fields[name]),
                                       box, ext, d, piece) for name in names}
            positions[g] = tack.array(_placed_host(piece.positions(), "points", box, ext, d),
                                      tack.vec3(piece.dtype))
        arrays_by_kind[g] = assembled
        grown_boxes.append((g, level, k, piece, box, ext))
    for name in names:
        kind = _kind(pieces[0].fields[name])
        fills[kind].run({g: arrays_by_kind[g][name] for g in layout.blocks.local})
    fills["points"].run(positions)

    out = [[] for _ in range(amr.num_levels)]
    for g, level, k, piece, box, ext in grown_boxes:
        out[level].append(_grown_box(amr, level, piece, box, ext, d, arrays_by_kind[g], names,
                                     positions[g], hidden[g]))
    return AMRDataSet._grown(amr, out, extents, int(layers), fills)


def _ghost_rows(amr, layout, extents, level, k):
    """A box's ghost cells and points: ``(cell rows, point rows, (hidden cells,
    hidden points))``, rows as ``_Fill`` takes them."""
    d = amr.dimension
    box, ext = amr._boxes[level][k], extents[level][k]
    siblings = [(j, b) for j, b in enumerate(amr._boxes[level]) if j != k]
    coarse = list(enumerate(amr._boxes[level - 1])) if level else []
    ratio = np.asarray(amr.ratios[level - 1], np.int64)[:d] if level else None

    # Cells: a sibling's, else the coarse cell's.
    cells = _grid(ext[0, :d], ext[1, :d], d)
    ghost = ~_inside(cells, box[0, :d], box[1, :d])
    q = cells[ghost]
    cell_targets = _cell_index(ext, q, d)
    rows = [[] for _ in range(len(q))]
    open_ = np.ones(len(q), bool)
    for j, other in siblings:
        hit = open_ & _inside(q, other[0, :d], other[1, :d])
        if hit.any():
            source = layout.gid(level, j)
            index = _cell_index(extents[level][j], q[hit], d)
            for r, i in zip(np.flatnonzero(hit), index):
                rows[r] = [(source, i, 1.0)]
            open_ &= ~hit
    for j, other in coarse:
        if not open_.any():
            break
        c = q // ratio
        hit = open_ & _inside(c, other[0, :d], other[1, :d])
        if hit.any():
            source = layout.gid(level - 1, j)
            index = _cell_index(extents[level - 1][j], c[hit], d)
            for r, i in zip(np.flatnonzero(hit), index):
                rows[r] = [(source, i, 1.0)]
            open_ &= ~hit
    hidden_cells = cell_targets[open_]
    cell_rows = (int(np.prod(ext[1, :d] - ext[0, :d])), cell_targets, rows)

    # Points: a sibling's, else interpolated in a coarse cell.
    points = _grid(ext[0, :d], ext[1, :d] + 1, d)
    ghost = ~np.all((points >= box[0, :d]) & (points <= box[1, :d]), axis=1)
    p = points[ghost]
    point_targets = _point_index(ext, p, d)
    rows = [[] for _ in range(len(p))]
    open_ = np.ones(len(p), bool)
    for j, other in siblings:
        hit = open_ & np.all((p >= other[0, :d]) & (p <= other[1, :d]), axis=1)
        if hit.any():
            source = layout.gid(level, j)
            index = _point_index(extents[level][j], p[hit], d)
            for r, i in zip(np.flatnonzero(hit), index):
                rows[r] = [(source, i, 1.0)]
            open_ &= ~hit
    for j, other in coarse:
        if not open_.any():
            break
        f = p / ratio
        hit = open_ & np.all((f >= other[0, :d]) & (f <= other[1, :d]), axis=1)
        if not hit.any():
            continue
        base = np.minimum(np.floor(f[hit]).astype(np.int64), other[1, :d] - 1)
        base = np.maximum(base, other[0, :d])
        t = f[hit] - base
        corners = [np.array([(c >> a) & 1 for a in range(d)]) for c in range(2 ** d)]
        for r, b, tt in zip(np.flatnonzero(hit), base, t):
            row = []
            for corner in corners:
                w = float(np.prod(np.where(corner == 1, tt, 1.0 - tt)))
                if w > 0.0:
                    # From the corner's owner: the lowest coarse box holding it,
                    # whose value is never a copy refreshed in the same pass.
                    at = b + corner
                    owner = next(i for i, c in coarse
                                 if np.all((at >= c[0, :d]) & (at <= c[1, :d])))
                    index = _point_index(extents[level - 1][owner], at[None], d)[0]
                    row.append((layout.gid(level - 1, owner), index, w))
            rows[r] = row
        open_ &= ~hit
    hidden_points = point_targets[open_]
    # The box's own points on a face a lower sibling shares are that sibling's
    # (DUPLICATEPOINT): refreshed from it too, as split pieces' shared points are.
    own = points[~ghost]
    shared = np.zeros(len(own), bool)
    extra_targets, extra_rows = [], []
    for j, other in siblings:
        if j > k:
            break
        hit = ~shared & np.all((own >= other[0, :d]) & (own <= other[1, :d]), axis=1)
        if hit.any():
            source = layout.gid(level, j)
            extra_targets.append(_point_index(ext, own[hit], d))
            extra_rows += [[(source, i, 1.0)]
                           for i in _point_index(extents[level][j], own[hit], d)]
            shared |= hit
    if extra_targets:
        point_targets = np.concatenate([point_targets, *extra_targets])
        rows = rows + extra_rows
    point_rows = (int(np.prod(ext[1, :d] - ext[0, :d] + 1)), point_targets, rows)
    return cell_rows, point_rows, (hidden_cells, hidden_points)


def _placed_host(values, kind, box, ext, d):
    """A box's host values (``(n,)`` or ``(n, w)``) placed in arrays of the grown
    extent, zeros elsewhere."""
    values = np.asarray(values)
    extra = values.shape[1:]
    if kind == "cells":
        own, grown = box[1, :d] - box[0, :d], ext[1, :d] - ext[0, :d]
    else:
        own, grown = box[1, :d] - box[0, :d] + 1, ext[1, :d] - ext[0, :d] + 1
    out = np.zeros(tuple(int(v) for v in grown[::-1]) + extra, values.dtype)
    start = box[0, :d] - ext[0, :d]
    region = tuple(slice(int(s), int(s + n)) for s, n in zip(start[::-1], own[::-1]))
    out[region] = values.reshape(tuple(int(v) for v in own[::-1]) + extra)
    return out.reshape((-1, *extra))


def _placed(values, kind, box, ext, d, piece):
    count = piece.num_points if kind == "points" else piece.num_cells
    values = arrays.materialize(_point_values(values, count))
    host = arrays.to_host(values)
    width = getattr(values.dtype, "_vector_n", None) or getattr(values, "_vector_n", None)
    host = np.asarray(host).reshape(count, -1) if width else np.asarray(host).reshape(count)
    return tack.array(_placed_host(host, kind, box, ext, d), values.dtype)


def _grown_box(amr, level, piece, box, ext, d, assembled, names, positions, hidden):
    """The box grown to ``ext``: its topology, geometry, fields and marks."""
    from tack.dataset.algorithms.ghost_cells import _kind

    dims = tuple(int(v) + 1 for v in (ext[1, :d] - ext[0, :d]))
    with piece.device:
        topology = StructuredTopology(dims, id_dtype=piece.topology.id_dtype,
                                      offset=tuple(int(v) for v in ext[0, :d]))
    coordinates = piece.geometry.values
    hidden_cells, hidden_points = hidden
    if isinstance(coordinates, UniformCoordinates):
        spacing = np.asarray(coordinates._spacing, np.float64)
        shift = np.zeros(3)
        shift[:d] = (box[0, :d] - ext[0, :d]) * spacing[:d]
        geometry = UniformCoordinates(dims, np.asarray(coordinates._origin) - shift,
                                      coordinates._spacing, dtype=coordinates.dtype,
                                      device=piece.device)
    else:
        grid = positions.numpy().reshape(-1, 3).astype(np.float64)
        grid = _extrapolated(grid, hidden_points, dims)
        if isinstance(coordinates, CartesianProduct):
            shaped = grid.reshape((*reversed(dims), 3))
            axes = []
            for a in range(3):
                if a < d:
                    moved = np.moveaxis(shaped[..., a], d - 1 - a, -1)
                    axes.append(np.median(moved.reshape(-1, dims[a]), axis=0))
                else:
                    axes.append(np.asarray(
                        arrays.to_host((coordinates.x, coordinates.y, coordinates.z)[a]),
                        np.float64))
            geometry = CartesianProduct(*axes, dtype=coordinates.dtype, device=piece.device)
        else:
            with piece.device:
                geometry = tack.array(grid.astype(piece.dtype.numpy_dtype), tack.vec3(piece.dtype))
    cells, points = _grown_marks(amr, level, piece, box, ext, d, hidden_cells, hidden_points)
    with piece.device:
        out = DataSet(topology, geometry, dtype=piece.dtype, origin=piece.origin,
                      ghosts=Ghosts(cells=cells, points=points), unit=piece.unit)
        for name in names:
            field = piece.fields[name]
            if _kind(field) == "points":
                space = H1(out) if isinstance(field.space, H1) else Values(out, "points")
            else:
                space = Constant(out) if isinstance(field.space, Constant) else \
                    Values(out, "cells")
            out.fields[name] = Field(space, assembled[name], field.unit)
    return out


def _grown_marks(amr, level, piece, box, ext, d, hidden_cells, hidden_points):
    """The grown box's marks: its own, then ``DUPLICATECELL`` and ``DUPLICATEPOINT``
    on the ghosts, hidden where nothing filled them, refined where a finer box
    covers them."""
    cells = _placed_host(piece.ghosts.cells.numpy(), "cells", box, ext, d)
    points = _placed_host(piece.ghosts.points.numpy(), "points", box, ext, d)
    own_cells = _placed_host(np.ones(piece.num_cells, bool), "cells", box, ext, d)
    own_points = _placed_host(np.ones(piece.num_points, bool), "points", box, ext, d)
    refined_cells, refined_points = amr._refined(level, ext[0], ext[1])
    cells[~own_cells] |= Ghosts.DUPLICATECELL
    cells[~own_cells & refined_cells.reshape(-1)] |= Ghosts.REFINEDCELL
    cells[hidden_cells] |= Ghosts.HIDDENCELL
    points[~own_points] |= Ghosts.DUPLICATEPOINT
    points[~own_points & refined_points.reshape(-1)] |= Ghosts.REFINEDPOINT
    points[hidden_points] |= Ghosts.HIDDENPOINT
    return cells, points


def _extrapolated(grid, hidden, dims):
    """Positions of the points nothing filled, extrapolated linearly from their
    neighbours along each axis in turn, so a hidden ghost cell has a shape."""
    if not len(hidden):
        return grid
    shape = tuple(reversed(dims))
    known = np.ones(len(grid), bool)
    known[hidden] = False
    grid, known = grid.reshape((*shape, 3)).copy(), known.reshape(shape)
    for _ in range(2 * max(dims)):
        if known.all():
            break
        for axis in range(len(shape)):
            for step in (1, -1):
                near = np.roll(known, -step, axis)
                far = np.roll(known, -2 * step, axis)
                valid = ~known & near & far
                edge = [slice(None)] * len(shape)
                edge[axis] = slice(-2, None) if step == 1 else slice(None, 2)
                valid[tuple(edge)] = False
                if valid.any():
                    a = np.roll(grid, -step, axis)
                    b = np.roll(grid, -2 * step, axis)
                    grid[valid] = 2 * a[valid] - b[valid]
                    known |= valid
    return grid.reshape(-1, 3)


def amr_update_ghosts(amr, names):
    """``update_ghosts`` of a hierarchy ``ghost_cells`` made."""
    from tack.dataset.algorithms.ghost_cells import _kind

    if amr._fills is None:
        raise ValueError("update_ghosts needs the hierarchy ghost_cells made, which keeps "
                         "the exchanges")
    layout_boxes = [box for _, box in amr]
    blocks = amr._fills["cells"].blocks
    for name in names:
        values, kind = {}, None
        for g, box in zip(blocks.local, layout_boxes):
            field = box.fields[name]
            kind = _kind(field)
            if kind is None or not isinstance(field.values, tack.Array):
                raise TypeError(f"update_ghosts refreshes point and cell fields of explicit "
                                f"values; {name!r} is not one")
            values[g] = field.values
        amr._fills[kind].run(values)


# -- fine onto coarse -----------------------------------------------------------------

def average_down(amr, field):
    """Put the finer levels' data into the coarse cells and points they cover, in
    place, as AMReX's ``average_down``: each refined coarse cell (``REFINEDCELL``)
    takes the volume-weighted average of the fine cells over it, each refined
    coarse point (``REFINEDPOINT``) the value of the fine point at its location.
    ``field`` names a floating-point point or cell field every box has; finest
    level first, so a value reaches the coarsest level in one call."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry
    from tack.dataset.algorithms.ghost_cells import _kind

    if not isinstance(amr, AMRDataSet):
        raise TypeError(f"average_down takes an AMRDataSet, not {type(amr).__name__}")
    boxes = [box for _, box in amr]
    kinds = {_kind(box.fields[field]) for box in boxes}
    if len(kinds) != 1 or None in kinds:
        raise TypeError(f"average_down averages a point or cell field; {field!r} is not one")
    kind = kinds.pop()
    layout = _Layout(amr)
    d = amr.dimension
    for level in range(amr.num_levels - 2, -1, -1):
        rows = {}
        for g, (lv, k) in zip(layout.blocks.local, layout.local):
            rows[g] = _down_rows(amr, layout, lv, k, kind, d) if lv == level else \
                (_extent_size(amr._extents[lv][k], kind, d), np.zeros(0, np.int64), [])
        fill = _Fill(layout.blocks, rows)
        values = {g: box.fields[field].values for g, box in zip(layout.blocks.local, boxes)}
        for g, array in values.items():
            scalar = getattr(array.dtype, "scalar_dtype", array.dtype)
            if not isinstance(array, tack.Array) or scalar.numpy_dtype.kind != "f":
                raise TypeError(f"average_down averages floating-point values of explicit "
                                f"arrays; {field!r} is not one")
        if kind == "points":
            fill.run(values)
            continue
        # Cells: sums of measure times value over sums of measure.
        weighted, measures = {}, {}
        for g, box in zip(layout.blocks.local, boxes):
            with box.device:
                measures[g] = arrays.materialize(cell_geometry(box).measure.values)
                weighted[g] = _times(values[g], measures[g])
        fill.run(weighted)
        fill.run(measures)
        for g in layout.blocks.local:
            targets = fill.boxes[g]["targets"]
            if len(targets):
                with layout.blocks.device(g):
                    _divide(weighted[g], measures[g], tack.array(targets, tack.i64), values[g])


@tack.kernel
def _scaled(values, m, out):
    for i in range(out.shape[0]):
        out[i] = values[i] * m[i]


def _times(values, m):
    out = tack.empty(values.shape, values.dtype)
    if values.shape[0]:
        _scaled(values, m, out)
    return out


def _extent_size(ext, kind, d):
    extra = 1 if kind == "points" else 0
    return int(np.prod(ext[1, :d] - ext[0, :d] + extra))


def _down_rows(amr, layout, level, k, kind, d):
    """A coarse box's refined cells (or points) and the fine sources of each."""
    box, ext = amr._boxes[level][k], amr._extents[level][k]
    ratio = np.asarray(amr.ratios[level], np.int64)[:d]
    fine = list(enumerate(amr._boxes[level + 1]))
    n = _extent_size(ext, kind, d)
    if kind == "cells":
        own = _grid(box[0, :d], box[1, :d], d)
        targets, rows = [], []
        for c in own:
            lo, hi = c * ratio, (c + 1) * ratio
            row = []
            for j, other in fine:
                a, b = np.maximum(lo, other[0, :d]), np.minimum(hi, other[1, :d])
                if (a < b).all():
                    q = _grid(a, b, d)
                    index = _cell_index(amr._extents[level + 1][j], q, d)
                    source = layout.gid(level + 1, j)
                    row += [(source, i, 1.0) for i in index]
            if row:
                targets.append(_cell_index(ext, c[None], d)[0])
                rows.append(row)
        return n, np.asarray(targets, np.int64), rows
    own = _grid(box[0, :d], box[1, :d] + 1, d)
    targets, rows = [], []
    for p in own:
        f = p * ratio
        for j, other in fine:
            if np.all((f >= other[0, :d]) & (f <= other[1, :d])):
                targets.append(_point_index(ext, p[None], d)[0])
                rows.append([(layout.gid(level + 1, j),
                              _point_index(amr._extents[level + 1][j], f[None], d)[0], 1.0)])
                break
    return n, np.asarray(targets, np.int64), rows

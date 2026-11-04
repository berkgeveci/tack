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
    """Values of some entries of each box as weighted sums of other boxes' entries,
    one group at a time (``_Groups``: a rank's boxes on one device, their arrays
    end to end): a ``Plan`` between groups stages the sources other groups hold
    after the group's own values, and one kernel sums each target's sources by
    weight. Rows of one box per block, and a plan with a pair for every two boxes
    that touch, made AMR-Wind's 3,328 boxes 326,480 launches a fill."""

    def __init__(self, groups, kind, parts):
        """``parts[g]``, for each of this rank's groups: ``(targets, offsets, sources,
        weights)``, the targets' places in the group's buffer, CSR offsets into
        ``sources`` (rows of ``(group, place)``, each group's buffer's) and
        ``weights``; ``box_targets`` per local box, its targets' own indices."""
        self.groups, self.kind = groups, kind
        blocks = groups.blocks
        self.boxes, receives = {}, {}
        for g in blocks.local:
            targets, offsets, sources, weights, _ = parts[g]
            n = groups.size(g, kind)
            remote = sources[:, 0] != g
            keys, slot = np.unique(sources[remote], axis=0, return_inverse=True) \
                if remote.any() else (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))
            entries = sources[:, 1].copy()
            entries[remote] = n + slot.reshape(-1)
            receives[g] = {int(source): (keys[keys[:, 0] == source, 1],
                                         n + np.flatnonzero(keys[:, 0] == source))
                           for source in np.unique(keys[:, 0])}
            # Integers take their weightiest source's value: the first of the largest.
            picks = np.zeros(len(targets), np.int64)
            if len(targets):
                counts = np.diff(offsets)
                largest = np.maximum.reduceat(weights, offsets[:-1]) if len(weights) else weights
                hit = weights == np.repeat(largest, counts)
                first = np.minimum.reduceat(np.where(hit, np.arange(len(weights)), len(weights)),
                                            offsets[:-1])
                picks = entries[first]
            self.boxes[g] = {"n": n, "m": len(keys), "targets": targets, "offsets": offsets,
                             "entries": entries, "weights": weights, "picks": picks}
        self.box_targets = {i: t for g in blocks.local for i, t in parts[g][4].items()}
        self.plan = _matching.plan(blocks, receives)

    def run(self, values, combine=True):
        """Fill the targets of ``values`` (this rank's boxes' arrays of one kind, in
        the order the hierarchy holds them), in place. With ``combine`` false,
        integer arrays included, each target takes its weightiest source's
        value."""
        groups, blocks = self.groups, self.groups.blocks
        combined = {}
        for g in blocks.local:
            box = self.boxes[g]
            parts = [values[i] for i in groups.members[g]]
            with blocks.device(g):
                if box["m"] or not parts:
                    # A device none of whose boxes take part still joins the exchange.
                    dtype = values[0].dtype if values else tack.f32
                    parts = [*parts, tack.zeros((box["m"],), dtype)]
                combined[g] = tack.concat(parts) if len(parts) > 1 else parts[0].copy()
        self.plan.run(combined)
        for g in blocks.local:
            box, buffer = self.boxes[g], combined[g]
            if not len(box["targets"]):
                continue
            with blocks.device(g):
                targets = tack.array(box["targets"], tack.i64)
                scalar = getattr(buffer.dtype, "scalar_dtype", buffer.dtype)
                if combine and scalar.numpy_dtype.kind == "f":
                    _apply(buffer, tack.array(box["offsets"], tack.i64),
                           tack.array(box["entries"], tack.i64),
                           tack.array(box["weights"].astype(scalar.numpy_dtype), scalar),
                           targets, buffer)
                else:
                    _pick(buffer, tack.array(box["picks"], tack.i64), targets, buffer)
                for i, start in zip(groups.members[g], groups.starts(g, self.kind)):
                    if len(self.box_targets.get(i, ())):
                        _copy_into(buffer, int(start), values[i])


@tack.kernel
def _copy_into(source, start, values):
    for i in range(values.shape[0]):
        values[i] = source[start + i]


# -- where boxes are ------------------------------------------------------------------

class _Groups:
    """This rank's boxes by device: each device's boxes a group, a block of
    ``tack.communication``, its boxes' arrays end to end in hierarchy order; and,
    gathered over ranks, every box's group and the start of its arrays (of
    ``extents``, cells and points) in its group's."""

    def __init__(self, amr, extents, levels=None):
        """``levels``: the levels whose boxes are members (all by default); the groups
        are the devices of every box, so every rank makes the same blocks."""
        d = amr.dimension
        # Each box's device, looked up once: ``DataSet.device`` searches the dataset's
        # parts on every call (27 us; 3.6 s of average_down's per-box calls).
        every = [box.device for _, box in amr]
        devices = []
        for device in every:
            if device not in devices:
                devices.append(device)
        self.local, self.pieces, self.devices, of_box = [], [], [], []
        n = 0
        for level, pieces in enumerate(amr.levels):
            for k, box in enumerate(pieces):
                device, n = every[n], n + 1
                if levels is not None and level not in levels:
                    continue
                self.local.append((level, amr._first[level] + k))
                self.pieces.append(box)
                self.devices.append(device)
                of_box.append(devices.index(device))
        self.blocks = tc.Blocks(devices, amr.world)
        gids = list(self.blocks.local)
        self.members = {g: [i for i, k in enumerate(of_box) if k == n] for n, g in enumerate(gids)}
        sizes = {"cells": [], "points": []}
        for level, k in self.local:
            ext = extents[level][k]
            sizes["cells"].append(int(np.prod(ext[1, :d] - ext[0, :d])))
            sizes["points"].append(int(np.prod(ext[1, :d] - ext[0, :d] + 1)))
        self._sizes = {kind: np.asarray(v, np.int64) for kind, v in sizes.items()}
        base = {kind: np.zeros(len(self.local), np.int64) for kind in sizes}
        for g, members in self.members.items():
            for kind in sizes:
                lengths = self._sizes[kind][members]
                base[kind][members] = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        # Per level, over every rank's boxes in the hierarchy's order: group and starts.
        self.group, self.cell_base, self.point_base = [], [], []
        for level in range(amr.num_levels):
            mine = [i for i, (lv, _) in enumerate(self.local) if lv == level]
            row = np.stack([np.asarray([gids[of_box[i]] for i in mine], np.int64),
                            base["cells"][mine], base["points"][mine]], axis=1) \
                if mine else np.zeros((0, 3), np.int64)
            every = np.concatenate(amr.world._allgather(row)).reshape(-1, 3)
            self.group.append(every[:, 0])
            self.cell_base.append(every[:, 1])
            self.point_base.append(every[:, 2])

    def size(self, g, kind):
        return int(self._sizes[kind][self.members[g]].sum())

    def starts(self, g, kind):
        lengths = self._sizes[kind][self.members[g]]
        return np.concatenate([[0], np.cumsum(lengths)[:-1]])

    def boxes(self, g, level):
        """The group's boxes of ``level``: their places in the hierarchy's order of this
        rank's boxes, and their global indices in the level."""
        places = [i for i in self.members[g] if self.local[i][0] == level]
        return places, np.asarray([self.local[i][1] for i in places], np.int64)


@tack.device_class
class _LevelBoxes:
    """A level's boxes on a device, found by the bins of a uniform grid over the
    level (each bin the boxes that meet it, closed, in index order), with each
    box's arrays' extent and its place in its group's: what the source kernels
    look up."""

    def __init__(self, amr, groups, extents, level, device):
        d = amr.dimension
        boxes, ext = amr._boxes[level], extents[level]
        lo, hi = amr.level_domain(level)
        n = len(boxes)
        size = np.ones(3, np.int64)
        if n:
            size[:d] = np.maximum((boxes[:, 1, :d] - boxes[:, 0, :d]).max(axis=0), 1)
        dims = np.ones(3, np.int64)
        dims[:d] = (hi[:d] - lo[:d]) // size[:d] + 2
        first = (boxes[:, 0, :] - lo) // size
        last = (boxes[:, 1, :] - lo) // size
        first[:, d:], last[:, d:] = 0, 0
        last = np.minimum(last, dims - 1)
        span = (last - first + 1).max(axis=0) if n else np.ones(3, np.int64)
        bins, ids = [], []
        for dz in range(int(span[2])):
            for dy in range(int(span[1])):
                for dx in range(int(span[0])):
                    at = first + np.array([dx, dy, dz])
                    ok = (at <= last).all(axis=1)
                    bins.append(at[ok, 0] + dims[0] * (at[ok, 1] + dims[1] * at[ok, 2]))
                    ids.append(np.flatnonzero(ok))
        bins = np.concatenate(bins) if bins else np.zeros(0, np.int64)
        ids = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        order = np.lexsort((ids, bins))
        offsets = np.concatenate([[0], np.cumsum(np.bincount(bins, minlength=int(np.prod(dims))))])
        def put(a):
            return tack.array(np.ascontiguousarray(a, np.int64).reshape(-1), tack.i64)

        with device:
            self.lo, self.hi = put(boxes[:, 0]), put(boxes[:, 1])
            self.elo, self.ehi = put(ext[:, 0]), put(ext[:, 1])
            self.group = put(groups.group[level])
            self.cbase, self.pbase = put(groups.cell_base[level]), put(groups.point_base[level])
            self.origin, self.size, self.dims = put(lo), put(size), put(dims)
            self.offsets, self.ids = put(offsets), put(ids[order])
        self.d = int(d)

    @tack.func
    def holds(self, j, q0, q1, q2, closed):
        """Whether box ``j`` holds index ``q``: ``[lo, hi)``, or ``[lo, hi]`` closed."""
        inside = 1
        for a in range(self.d):
            q = q0 if a == 0 else (q1 if a == 1 else q2)
            low = self.lo[3 * j + a]
            high = self.hi[3 * j + a]
            if q < low or q > high or (closed == 0 and q == high):
                inside = 0
        return inside

    @tack.func
    def find(self, q0, q1, q2, closed, skip, below):
        """The lowest box below ``below``, other than ``skip``, that holds ``q``, or -1."""
        found = -1
        b0 = (q0 - self.origin[0]) // self.size[0]
        b1 = (q1 - self.origin[1]) // self.size[1] if self.d > 1 else 0
        b2 = (q2 - self.origin[2]) // self.size[2] if self.d > 2 else 0
        if (b0 >= 0 and b1 >= 0 and b2 >= 0 and b0 < self.dims[0] and b1 < self.dims[1]
                and b2 < self.dims[2]):
            b = b0 + self.dims[0] * (b1 + self.dims[1] * b2)
            m = self.offsets[b]
            while m < self.offsets[b + 1] and found < 0:
                j = self.ids[m]
                if j != skip and j < below and self.holds(j, q0, q1, q2, closed) == 1:
                    found = j
                m += 1
        return found

    @tack.func
    def find_scaled(self, q0, q1, q2, r0, r1, r2):
        """The lowest box that holds ``q / r`` (a finer level's point), closed, or -1."""
        found = -1
        b0 = (q0 // r0 - self.origin[0]) // self.size[0]
        b1 = (q1 // r1 - self.origin[1]) // self.size[1] if self.d > 1 else 0
        b2 = (q2 // r2 - self.origin[2]) // self.size[2] if self.d > 2 else 0
        if (b0 >= 0 and b1 >= 0 and b2 >= 0 and b0 < self.dims[0] and b1 < self.dims[1]
                and b2 < self.dims[2]):
            b = b0 + self.dims[0] * (b1 + self.dims[1] * b2)
            m = self.offsets[b]
            while m < self.offsets[b + 1] and found < 0:
                j = self.ids[m]
                inside = 1
                for a in range(self.d):
                    q = q0 if a == 0 else (q1 if a == 1 else q2)
                    r = r0 if a == 0 else (r1 if a == 1 else r2)
                    if q < self.lo[3 * j + a] * r or q > self.hi[3 * j + a] * r:
                        inside = 0
                if inside == 1:
                    found = j
                m += 1
        return found

    @tack.func
    def place(self, j, q0, q1, q2, points):
        """Where index ``q`` of box ``j``'s arrays is in its group's: its start plus the
        index's place in the box's extent, x fastest (points one more a side)."""
        extra = 1 if points == 1 else 0
        nx = self.ehi[3 * j] - self.elo[3 * j] + extra
        ny = self.ehi[3 * j + 1] - self.elo[3 * j + 1] + extra
        flat = q2 - self.elo[3 * j + 2] if self.d > 2 else q2 - q2
        if self.d > 1:
            flat = flat * ny + (q1 - self.elo[3 * j + 1])
        flat = flat * nx + (q0 - self.elo[3 * j])
        start = self.pbase[j] if points == 1 else self.cbase[j]
        return start + flat


@tack.func
def _entry(offsets, n, e):
    """The box (an index into ``offsets``) whose entries hold entry ``e``."""
    low = 0
    high = n
    while high - low > 1:
        mid = (low + high) // 2
        if offsets[mid] <= e:
            low = mid
        else:
            high = mid
    return low


@tack.func
def _index(same, j, r, points, use_box):
    """Entry ``r`` of box ``j``'s extent (or of the box itself), x fastest: ``(q0, q1,
    q2)``."""
    extra = 1 if points == 1 else 0
    lo0 = same.lo[3 * j] if use_box == 1 else same.elo[3 * j]
    lo1 = same.lo[3 * j + 1] if use_box == 1 else same.elo[3 * j + 1]
    lo2 = same.lo[3 * j + 2] if use_box == 1 else same.elo[3 * j + 2]
    hi0 = same.hi[3 * j] if use_box == 1 else same.ehi[3 * j]
    hi1 = same.hi[3 * j + 1] if use_box == 1 else same.ehi[3 * j + 1]
    nx = hi0 - lo0 + extra
    ny = (hi1 - lo1 + extra) if same.d > 1 else hi1 - hi1 + 1
    q0 = lo0 + r % nx
    q1 = lo1 + (r // nx) % ny if same.d > 1 else lo1
    q2 = lo2 + r // (nx * ny) if same.d > 2 else lo2
    return q0, q1, q2


@tack.func
def _hole(same, j, a, points):
    """Axis ``a`` of box ``j``'s grown extent, for ``_shell_index``: its size, and the
    start and size of the hole in it, the entries a sibling never holds (the box's
    own cells, or its own points off its boundary); an axis the boxes lack is one
    entry, inside the hole."""
    n = 1 + 0 * j
    start = 0 * j
    size = 1 + 0 * j
    if a < same.d:
        lo = same.lo[3 * j + a]
        hi = same.hi[3 * j + a]
        elo = same.elo[3 * j + a]
        n = same.ehi[3 * j + a] - elo + points
        start = lo - elo + points
        size = hi - lo - points
    return n, start, size


@tack.func
def _shell_index(same, j, r, points):
    """Entry ``r`` of box ``j``'s shell: its grown extent less the hole (``_hole``),
    x fastest, as ``(q0, q1, q2)``. Slabs of z before the hole, through it and
    after it, and in a slab through it rows of y alike, so the hole's entries are
    never enumerated."""
    nx, ax, hx = _hole(same, j, 0, points)
    ny, ay, hy = _hole(same, j, 1, points)
    nz, az, hz = _hole(same, j, 2, points)
    if hx <= 0 or hy <= 0 or hz <= 0:       # no hole
        hx = 0 * hx
        hy = 0 * hy
        hz = 0 * hz
        ax = 0 * ax
        ay = 0 * ay
        az = nz
    slab = nx * ny
    holed = slab - hx * hy
    z = 0 * r
    rest = r
    full = 1
    if r < az * slab:
        z = r // slab
        rest = r - z * slab
    elif r < az * slab + hz * holed:
        k = r - az * slab
        z = az + k // holed
        rest = k - (z - az) * holed
        full = 0
    else:
        k = r - az * slab - hz * holed
        z = az + hz + k // slab
        rest = k - (z - az - hz) * slab
    y = 0 * r
    x = 0 * r
    if full == 1:
        y = rest // nx
        x = rest - y * nx
    elif rest < ay * nx:
        y = rest // nx
        x = rest - y * nx
    elif rest < ay * nx + hy * (nx - hx):
        m = rest - ay * nx
        y = ay + m // (nx - hx)
        x = m - (y - ay) * (nx - hx)
        if x >= ax:
            x = x + hx
    else:
        m = rest - ay * nx - hy * (nx - hx)
        y = ay + hy + m // nx
        x = m - (y - ay - hy) * nx
    return same.elo[3 * j] + x, same.elo[3 * j + 1] + y, same.elo[3 * j + 2] + z


def _shell_sizes(boxes, extents, d, points):
    """Each box's shell entries (``_shell_index``): its grown extent's cells or points
    less the hole."""
    lo, hi = boxes[:, 0, :d], boxes[:, 1, :d]
    full = np.prod(extents[:, 1, :d] - extents[:, 0, :d] + points, axis=1)
    hole = np.prod(np.maximum(hi - lo - points, 0), axis=1)
    return full - hole


@tack.kernel
def _ghost_cells_of(same, coarse, has_coarse, r0, r1, r2, ids, offsets, n, targets, groups,
                    places):
    # Each entry of the boxes' shells (``_shell_index``): a ghost cell's target place and its one
    # source, a sibling's (the lowest holding it) else the coarse cell's; -1 targets
    # for the boxes' own cells, -1 groups for ghosts nothing holds (hidden).
    for e in range(targets.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _shell_index(same, j, e - offsets[b], 0)
        target = -1
        group = -1
        place = 0 * j
        if same.holds(j, q0, q1, q2, 0) == 0:
            target = same.place(j, q0, q1, q2, 0)
            s = same.find(q0, q1, q2, 0, j, same.ids.shape[0] + same.lo.shape[0])
            if s >= 0:
                group = same.group[s]
                place = same.place(s, q0, q1, q2, 0)
            elif has_coarse == 1:
                c0 = q0 // r0
                c1 = q1 // r1
                c2 = q2 // r2
                t = coarse.find(c0, c1, c2, 0, -1, coarse.ids.shape[0] + coarse.lo.shape[0])
                if t >= 0:
                    group = coarse.group[t]
                    place = coarse.place(t, c0, c1, c2, 0)
        targets[e] = target
        groups[e] = group
        places[e] = place


@tack.kernel
def _ghost_points_of(same, coarse, has_coarse, r0, r1, r2, den, ids, offsets, n, write, starts,
                     counts, targets, groups, places, weights):
    # Each point of the boxes' shells (``_shell_index``): its sources' count (-1 not a target, 0
    # hidden) and, with ``write``, the sources at ``starts``. A ghost point is a
    # sibling's (the lowest holding it), else interpolated in the coarse cell
    # holding it by its corners' owners (the lowest coarse box holding each); an
    # own point a lower sibling also holds is that sibling's. Weights are integer
    # numerators over ``den`` (the ratios' product), so no device needs f64.
    for e in range(counts.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _shell_index(same, j, e - offsets[b], 1)
        big = same.ids.shape[0] + same.lo.shape[0]
        count = -1
        at = starts[e] if write == 1 else 0 * e
        if same.holds(j, q0, q1, q2, 1) == 1:
            s = same.find(q0, q1, q2, 1, j, j)
            if s >= 0:
                count = 1
                if write == 1:
                    targets[e] = same.place(j, q0, q1, q2, 1)
                    groups[at] = same.group[s]
                    places[at] = same.place(s, q0, q1, q2, 1)
                    weights[at] = den
        else:
            count = 0
            if write == 1:
                targets[e] = same.place(j, q0, q1, q2, 1)
            s = same.find(q0, q1, q2, 1, j, big)
            if s >= 0:
                count = 1
                if write == 1:
                    groups[at] = same.group[s]
                    places[at] = same.place(s, q0, q1, q2, 1)
                    weights[at] = den
            elif has_coarse == 1:
                t = coarse.find_scaled(q0, q1, q2, r0, r1, r2)
                if t >= 0:
                    # The coarse cell: q // r, kept within the box; q's place in it.
                    c0 = min(max(q0 // r0, coarse.lo[3 * t]), coarse.hi[3 * t] - 1)
                    c1 = q1 - q1
                    c2 = q2 - q2
                    if coarse.d > 1:
                        c1 = min(max(q1 // r1, coarse.lo[3 * t + 1]), coarse.hi[3 * t + 1] - 1)
                    if coarse.d > 2:
                        c2 = min(max(q2 // r2, coarse.lo[3 * t + 2]), coarse.hi[3 * t + 2] - 1)
                    t0 = q0 - c0 * r0
                    t1 = q1 - c1 * r1
                    t2 = q2 - c2 * r2
                    corners = 1
                    for a in range(coarse.d):
                        corners = corners * 2
                    for c in range(corners):
                        a0 = c & 1
                        a1 = (c >> 1) & 1
                        a2 = (c >> 2) & 1
                        w = t0 if a0 == 1 else r0 - t0
                        if coarse.d > 1:
                            w = w * (t1 if a1 == 1 else r1 - t1)
                        if coarse.d > 2:
                            w = w * (t2 if a2 == 1 else r2 - t2)
                        if w > 0:
                            k0 = c0 + a0
                            k1 = c1 + a1 if coarse.d > 1 else c1
                            k2 = c2 + a2 if coarse.d > 2 else c2
                            owner = coarse.find(k0, k1, k2, 1, -1, big + coarse.lo.shape[0])
                            if write == 1:
                                groups[at + count] = coarse.group[owner]
                                places[at + count] = coarse.place(owner, k0, k1, k2, 1)
                                weights[at + count] = w
                            count += 1
        if write == 0:
            counts[e] = count


_MOST_FINE = tack.constant(64)


@tack.kernel
def _down_cells_of(coarse, fine, r0, r1, r2, ids, offsets, n, write, starts, counts, targets,
                   groups, places):
    # Each own cell of the coarse boxes: the fine cells over it, by fine box (the
    # lowest holding each), then x fastest within; -1 for a cell no fine box covers.
    for e in range(counts.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _index(coarse, j, e - offsets[b], 0, 1)
        big = fine.ids.shape[0] + fine.lo.shape[0]
        boxes = tack.local_array(tack.i64, _MOST_FINE)
        found = tack.local_array(tack.i64, _MOST_FINE)
        count = 0
        n1 = r1 if fine.d > 1 else 1
        n2 = r2 if fine.d > 2 else 1
        for k2 in range(n2):
            for k1 in range(n1):
                for k0 in range(r0):
                    f0 = q0 * r0 + k0
                    f1 = q1 * r1 + k1 if fine.d > 1 else q1
                    f2 = q2 * r2 + k2 if fine.d > 2 else q2
                    s = fine.find(f0, f1, f2, 0, -1, big)
                    if s >= 0:
                        # Insert after every entry of a box at most s: by box, stable.
                        at = count
                        while at > 0 and boxes[at - 1] > s:
                            boxes[at] = boxes[at - 1]
                            found[at] = found[at - 1]
                            at -= 1
                        boxes[at] = s
                        found[at] = fine.place(s, f0, f1, f2, 0)
                        count += 1
        if write == 1:
            if count > 0:
                targets[e] = coarse.place(j, q0, q1, q2, 0)
            for m in range(count):
                groups[starts[e] + m] = fine.group[boxes[m]]
                places[starts[e] + m] = found[m]
        else:
            counts[e] = count if count > 0 else -1


@tack.kernel
def _down_points_of(coarse, fine, r0, r1, r2, ids, offsets, n, targets, groups, places):
    # Each own point of the coarse boxes: the fine point at its location, the lowest
    # fine box's holding it; -1 targets where none does.
    for e in range(targets.shape[0]):
        b = _entry(offsets, n, e)
        j = ids[b]
        q0, q1, q2 = _index(coarse, j, e - offsets[b], 1, 1)
        f0 = q0 * r0
        f1 = q1 * r1 if fine.d > 1 else q1
        f2 = q2 * r2 if fine.d > 2 else q2
        s = fine.find(f0, f1, f2, 1, -1, fine.ids.shape[0] + fine.lo.shape[0])
        target = -1
        group = -1
        place = 0 * e
        if s >= 0:
            target = coarse.place(j, q0, q1, q2, 1)
            group = fine.group[s]
            place = fine.place(s, f0, f1, f2, 1)
        targets[e] = target
        groups[e] = group
        places[e] = place


def _ratio3(amr, level):
    r = np.ones(3, np.int64)
    r[:amr.dimension] = np.asarray(amr.ratios[level], np.int64)[:amr.dimension]
    return r


def _tables(amr, groups, extents, g):
    """Each level's ``_LevelBoxes`` on group ``g``'s device, made when first asked."""
    made = {}

    def table(level):
        if level not in made:
            made[level] = _LevelBoxes(amr, groups, extents, level, groups.blocks.device(g))
        return made[level]
    return table


def _entry_boxes(offsets):
    """Each entry's box, from the boxes' CSR ``offsets``: a ``searchsorted`` of every
    entry was 30 s of AMR-Wind's five levels."""
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def _entries(sizes, device):
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    with device:
        return offsets, tack.array(offsets, tack.i64)


def _part(targets, entry_box, base, places, sources, weights, offsets):
    """A ``_Fill`` part from the targets with sources: their places, CSR rows, and
    each box's targets as its own indices."""
    return targets, offsets, sources, weights, _by_box(targets - base, entry_box, places)


def _by_box(values, entry_box, places):
    """``values`` split by box: ``{place: its values}``; ``entry_box`` is in box order,
    so each box's values are one run (a mask per box was quadratic: 36 s for
    AMR-Wind's 6,436 boxes)."""
    if not len(values):
        return {}
    boxes, starts = np.unique(entry_box, return_index=True)
    ends = np.append(starts[1:], len(values))
    return {places[b]: values[a:e] for b, a, e in zip(boxes, starts, ends)}


def _ghost_parts(amr, groups, extents):
    """Every group's ghost-cell and ghost-point fills' parts, and each local box's
    hidden cells and points (its own indices)."""
    blocks = groups.blocks
    cell_parts, point_parts = {}, {}
    hidden = {i: (np.zeros(0, np.int64), np.zeros(0, np.int64)) for i in range(len(groups.local))}
    for g in blocks.local:
        device = blocks.device(g)
        table = _tables(amr, groups, extents, g)
        cells = {"targets": [], "box": [], "base": [], "places": [], "sources": []}
        points = {"targets": [], "box": [], "base": [], "places": [], "sources": [], "weights": [],
                  "counts": []}
        for level in range(amr.num_levels):
            places, ids = groups.boxes(g, level)
            if not places:
                continue
            same = table(level)
            coarse = table(level - 1) if level else same
            r = _ratio3(amr, level - 1) if level else np.ones(3, np.int64)
            den = int(np.prod(r[:amr.dimension]))
            ext = extents[level][ids]
            d = amr.dimension
            nb = len(ids)
            with device:
                ids_d = tack.array(ids, tack.i64)
            own = amr._boxes[level][ids]
            # Cells: one source each, for the shell's cells only.
            offsets, offsets_d = _entries(_shell_sizes(own, ext, d, 0), device)
            total = int(offsets[-1])
            with device:
                targets = tack.empty((total,), tack.i64)
                grp = tack.empty((total,), tack.i64)
                where = tack.empty((total,), tack.i64)
                if total:
                    _ghost_cells_of(same, coarse, int(level > 0), *map(int, r), ids_d, offsets_d, nb,
                                    targets, grp, where)
            t, gr, pl = targets.numpy(), grp.numpy(), where.numpy()
            box = _entry_boxes(offsets)
            cbase = groups.cell_base[level][ids]
            gone = np.flatnonzero((t >= 0) & (gr < 0))
            for place, lost in _by_box(t[gone] - cbase[box[gone]], box[gone], places).items():
                hidden[place] = (lost, hidden[place][1])
            kept = (t >= 0) & (gr >= 0)
            cells["targets"].append(t[kept])
            cells["box"].append(box[kept] + sum(len(x) for x in cells["places"]))
            cells["base"].append(cbase[box[kept]])
            cells["places"].append(places)
            cells["sources"].append(np.stack([gr[kept], pl[kept]], axis=1))
            # Points: a count, then the sources, for the shell's points only.
            offsets, offsets_d = _entries(_shell_sizes(own, ext, d, 1), device)
            total = int(offsets[-1])
            with device:
                counts = tack.empty((total,), tack.i32)
                one = tack.zeros((1,), tack.i64)
                if total:
                    _ghost_points_of(same, coarse, int(level > 0), *map(int, r), den, ids_d, offsets_d,
                                     nb, 0, one, counts, one, one, one, one)
            c = counts.numpy().astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(np.maximum(c, 0))])
            m = int(starts[-1])
            with device:
                targets = tack.full((max(total, 1),), -1, tack.i64)
                grp = tack.empty((max(m, 1),), tack.i64)
                where = tack.empty((max(m, 1),), tack.i64)
                nums = tack.empty((max(m, 1),), tack.i64)
                if total:
                    _ghost_points_of(same, coarse, int(level > 0), *map(int, r), den, ids_d, offsets_d,
                                     nb, 1, tack.array(starts[:-1], tack.i64), counts, targets, grp,
                                     where, nums)
            t = targets.numpy()[:total]
            box = _entry_boxes(offsets)
            pbase = groups.point_base[level][ids]
            gone = np.flatnonzero(c == 0)
            for place, lost in _by_box(t[gone] - pbase[box[gone]], box[gone], places).items():
                hidden[place] = (hidden[place][0], lost)
            kept = c > 0
            points["targets"].append(t[kept])
            points["box"].append(box[kept] + sum(len(x) for x in points["places"]))
            points["base"].append(pbase[box[kept]])
            points["places"].append(places)
            points["counts"].append(c[kept])
            points["sources"].append(np.stack([grp.numpy()[:m], where.numpy()[:m]], axis=1))
            points["weights"].append(nums.numpy()[:m].astype(np.float64) / den)
        cell_parts[g] = _assembled(cells, None)
        point_parts[g] = _assembled(points, points["counts"])
    return cell_parts, point_parts, hidden


def _assembled(chunks, counts):
    """One group's part from its levels' chunks."""
    if not chunks["targets"]:
        empty = np.zeros(0, np.int64)
        return empty, np.zeros(1, np.int64), np.zeros((0, 2), np.int64), np.zeros(0), {}
    targets = np.concatenate(chunks["targets"])
    places = [p for part in chunks["places"] for p in part]
    sources = np.concatenate(chunks["sources"])
    if counts is None:
        offsets = np.arange(len(targets) + 1, dtype=np.int64)
        weights = np.ones(len(targets))
    else:
        offsets = np.concatenate([[0], np.cumsum(np.concatenate(counts))]).astype(np.int64)
        weights = np.concatenate(chunks["weights"])
    return _part(targets, np.concatenate(chunks["box"]), np.concatenate(chunks["base"]), places,
                 sources, weights, offsets)


def _down_parts(amr, groups, level, kind):
    """Every group's ``average_down`` part for ``level``'s boxes: each covered coarse
    cell (point) and the fine cells (point) over it."""
    blocks, extents, d = groups.blocks, amr._extents, amr.dimension
    r = _ratio3(amr, level)
    if kind == "cells" and int(np.prod(r[:d])) > int(_MOST_FINE):
        raise ValueError(f"average_down takes refinement ratios of at most {int(_MOST_FINE)} "
                         f"fine cells a coarse cell, not {tuple(r[:d])}")
    parts = {}
    for g in blocks.local:
        device = blocks.device(g)
        places, ids = groups.boxes(g, level)
        if not places:
            parts[g] = (np.zeros(0, np.int64), np.zeros(1, np.int64), np.zeros((0, 2), np.int64),
                        np.zeros(0), {})
            continue
        table = _tables(amr, groups, extents, g)
        coarse, fine = table(level), table(level + 1)
        box = amr._boxes[level][ids]
        nb = len(ids)
        extra = 1 if kind == "points" else 0
        offsets, offsets_d = _entries(np.prod(box[:, 1, :d] - box[:, 0, :d] + extra, axis=1),
                                      device)
        total = int(offsets[-1])
        with device:
            ids_d = tack.array(ids, tack.i64)
            targets = tack.full((max(total, 1),), -1, tack.i64)
        entry_box = _entry_boxes(offsets)
        base = (groups.point_base if kind == "points" else groups.cell_base)[level][ids]
        if kind == "points":
            with device:
                grp = tack.empty((max(total, 1),), tack.i64)
                where = tack.empty((max(total, 1),), tack.i64)
                if total:
                    _down_points_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, targets, grp,
                                    where)
            t = targets.numpy()[:total]
            kept = t >= 0
            sources = np.stack([grp.numpy()[:total][kept], where.numpy()[:total][kept]], axis=1)
            offs = np.arange(int(kept.sum()) + 1, dtype=np.int64)
        else:
            with device:
                counts = tack.empty((max(total, 1),), tack.i32)
                one = tack.zeros((1,), tack.i64)
                if total:
                    _down_cells_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, 0, one, counts,
                                   one, one, one)
            c = counts.numpy()[:total].astype(np.int64)
            starts = np.concatenate([[0], np.cumsum(np.maximum(c, 0))])
            m = int(starts[-1])
            with device:
                grp = tack.empty((max(m, 1),), tack.i64)
                where = tack.empty((max(m, 1),), tack.i64)
                if total:
                    _down_cells_of(coarse, fine, *map(int, r), ids_d, offsets_d, nb, 1,
                                   tack.array(starts[:-1], tack.i64), counts, targets, grp, where)
            t = targets.numpy()[:total]
            kept = c > 0
            sources = np.stack([grp.numpy()[:m], where.numpy()[:m]], axis=1)
            offs = np.concatenate([[0], np.cumsum(c[kept])]).astype(np.int64)
        parts[g] = _part(t[kept], entry_box[kept], base[entry_box[kept]], places, sources,
                         np.ones(len(sources)), offs)
    return parts


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
    names = _carried(pieces, fields, "ghost_cells")
    common_origin(amr._as_pieces(), "ghost_cells of an AMRDataSet")
    extents = []
    for level in range(amr.num_levels):
        lo, hi = amr.level_domain(level)
        grown = amr._boxes[level].copy()
        grown[:, 0, :d] = np.maximum(grown[:, 0, :d] - layers, lo[:d])
        grown[:, 1, :d] = np.minimum(grown[:, 1, :d] + layers, hi[:d])
        extents.append(grown)
    groups = _Groups(amr, extents)
    cell_parts, point_parts, hidden = _ghost_parts(amr, groups, extents)
    fills = {"cells": _Fill(groups, "cells", cell_parts),
             "points": _Fill(groups, "points", point_parts)}

    # The grown boxes: own values placed, ghosts filled; positions too, for grids
    # whose positions are not an image's.
    # Only grids other than images read positions (an image keeps its lattice), but
    # an image box can be a source of their ghosts: all or none, over every rank.
    need = any(amr.world._allgather(
        any(not isinstance(piece.geometry.values, UniformCoordinates) for piece in pieces)))
    grown_boxes, arrays_by_kind, positions = [], [], []
    for i, ((level, k), piece) in enumerate(zip(groups.local, pieces)):
        box, ext = amr._boxes[level][k], extents[level][k]
        with groups.devices[i]:
            assembled = {name: _placed(piece.fields[name].values, _kind(piece.fields[name]),
                                       box, ext, d) for name in names}
            positions.append(tack.array(_placed_host(piece.positions(), "points", box, ext, d),
                                        tack.vec3(piece.dtype)) if need else None)
        arrays_by_kind.append(assembled)
        grown_boxes.append((i, level, k, piece, box, ext))
    for name in names:
        kind = _kind(pieces[0].fields[name])
        fills[kind].run([arrays[name] for arrays in arrays_by_kind])
    if need:
        fills["points"].run(positions)

    # What a finer level covers, each level's grown boxes at once.
    refined = {}
    for level in range(amr.num_levels):
        ks = [k for _, lvl, k, *_ in grown_boxes if lvl == level]
        refined.update(zip(((level, k) for k in ks),
                           amr._refined_boxes(level, extents[level][ks])))
    out = [[] for _ in range(amr.num_levels)]
    for i, level, k, piece, box, ext in grown_boxes:
        out[level].append(_grown_box(piece, groups.devices[i], box, ext, d, arrays_by_kind[i],
                                     names, positions[i], hidden[i], refined[level, k]))
    return AMRDataSet._grown(amr, out, extents, int(layers), fills)




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


def _placed(values, kind, box, ext, d):
    """A box's values placed in an array of the grown extent, zeros elsewhere, on the
    device (through the host it was 3.6 s of AMR-Wind's five levels)."""
    extra = 1 if kind == "points" else 0
    own = np.ones(3, np.int64)
    grown = np.ones(3, np.int64)
    start = np.zeros(3, np.int64)
    own[:d] = box[1, :d] - box[0, :d] + extra
    grown[:d] = ext[1, :d] - ext[0, :d] + extra
    start[:d] = box[0, :d] - ext[0, :d]
    values = arrays.materialize(_point_values(values, int(np.prod(own))))
    out = tack.zeros((int(np.prod(grown)),), values.dtype)
    if values.shape[0]:
        _place_into(values, out, *map(int, own[:2]), *map(int, grown[:2]), *map(int, start))
    return out


@tack.kernel
def _place_into(values, out, nx, ny, gx, gy, sx, sy, sz):
    for k in range(values.shape[0]):
        i = k % nx
        j = (k // nx) % ny
        m = k // (nx * ny)
        out[(i + sx) + gx * ((j + sy) + gy * (m + sz))] = values[k]


def _grown_box(piece, device, box, ext, d, assembled, names, positions, hidden, refined):
    """The box grown to ``ext``: its topology, geometry, fields and marks."""
    from tack.dataset.algorithms.ghost_cells import _kind

    dims = tuple(int(v) + 1 for v in (ext[1, :d] - ext[0, :d]))
    with device:
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
                                      device=device)
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
            geometry = CartesianProduct(*axes, dtype=coordinates.dtype, device=device)
        else:
            with device:
                geometry = tack.array(grid.astype(piece.dtype.numpy_dtype), tack.vec3(piece.dtype))
    cells, points = _grown_marks(piece, box, ext, d, hidden_cells, hidden_points, refined)
    with device:
        out = DataSet(topology, geometry, dtype=piece.dtype,
                      ghosts=Ghosts(cells=cells, points=points), **piece._frame())
        for name in names:
            field = piece.fields[name]
            if _kind(field) == "points":
                space = H1(out) if isinstance(field.space, H1) else Values(out, "points")
            else:
                space = Constant(out) if isinstance(field.space, Constant) else \
                    Values(out, "cells")
            out.fields[name] = Field(space, assembled[name], field.unit)
    return out


def _grown_marks(piece, box, ext, d, hidden_cells, hidden_points, refined):
    """The grown box's marks: its own, then ``DUPLICATECELL`` and ``DUPLICATEPOINT``
    on the ghosts, hidden where nothing filled them, refined where a finer box
    covers them (``refined``, ``AMRDataSet._refined_boxes``' for the grown box)."""
    cells = _placed_host(piece.ghosts.cells.numpy(), "cells", box, ext, d)
    points = _placed_host(piece.ghosts.points.numpy(), "points", box, ext, d)
    own_cells = _placed_host(np.ones(piece.num_cells, bool), "cells", box, ext, d)
    own_points = _placed_host(np.ones(piece.num_points, bool), "points", box, ext, d)
    refined_cells, refined_points = refined
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
    for name in names:
        values, kind = [], None
        for box in layout_boxes:
            field = box.fields[name]
            kind = _kind(field)
            if kind is None or not isinstance(field.values, tack.Array):
                raise TypeError(f"update_ghosts refreshes point and cell fields of explicit "
                                f"values; {name!r} is not one")
            values.append(field.values)
        amr._fills[kind].run(values)


# -- fine onto coarse -----------------------------------------------------------------

def average_down(amr, field):
    """Put the finer levels' data into the coarse cells and points they cover, in
    place, as AMReX's ``average_down``: each refined coarse cell (``REFINEDCELL``)
    takes the volume-weighted average of the fine cells over it, each refined
    coarse point (``REFINEDPOINT``) the value of the fine point at its location.
    ``field`` names a floating-point point or cell field every box has; finest
    level first, so a value reaches the coarsest level in one call."""
    from tack.dataset.algorithms.ghost_cells import _kind

    if not isinstance(amr, AMRDataSet):
        raise TypeError(f"average_down takes an AMRDataSet, not {type(amr).__name__}")
    boxes = [box for _, box in amr]
    kinds = {_kind(box.fields[field]) for box in boxes}
    if len(kinds) != 1 or None in kinds:
        raise TypeError(f"average_down averages a point or cell field; {field!r} is not one")
    kind = kinds.pop()
    measures = {}                       # (level, index): the box's cell measures
    for level in range(amr.num_levels - 2, -1, -1):
        if not len(amr._boxes[level + 1]) or not len(amr._boxes[level]):
            continue                         # nothing finer to average onto this level
        # Only this level's boxes and the next finer level's take part: a fill over
        # every box of the hierarchy copied all of them at every level.
        groups = _Groups(amr, amr._extents, levels=(level, level + 1))
        fill = _Fill(groups, kind, _down_parts(amr, groups, level, kind))
        values = [box.fields[field].values for box in groups.pieces]
        for array in values:
            scalar = getattr(array.dtype, "scalar_dtype", array.dtype)
            if not isinstance(array, tack.Array) or scalar.numpy_dtype.kind != "f":
                raise TypeError(f"average_down averages floating-point values of explicit "
                                f"arrays; {field!r} is not one")
        if kind == "points":
            fill.run(values)
            continue
        # Cells: sums of measure times value over sums of measure. The measures are
        # the boxes' own, each found once.
        weighted, sums = [], []
        for key, box, device, array in zip(groups.local, groups.pieces, groups.devices, values):
            if key not in measures:
                measures[key] = _measures(box, device, array.dtype)
            with device:
                weighted.append(_times(array, measures[key]))
                sums.append(measures[key].copy())       # the fill sums into them
        fill.run(weighted)
        fill.run(sums)
        for i, targets in fill.box_targets.items():
            if len(targets):
                with groups.devices[i]:
                    _divide(weighted[i], sums[i], tack.array(targets, tack.i64), values[i])


def _measures(box, device, dtype):
    """A box's cells' measures in ``dtype``: an image's, all alike, the product of its
    spacing (``cell_geometry`` of each of AMR-Wind's 15,598 boxes was 10 s)."""
    from tack.dataset.algorithms.cell_geometry import cell_geometry

    coordinates = box.geometry.values
    with device:
        if isinstance(coordinates, UniformCoordinates):
            d = len(box.topology.point_dims)
            volume = float(np.prod(np.asarray(coordinates._spacing, np.float64)[:d]))
            return tack.full((box.num_cells,), volume, dtype)
        return arrays.materialize(cell_geometry(box).measure.values)


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


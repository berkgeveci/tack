"""Block-structured adaptive mesh refinement: ``AMRDataSet``, levels of structured
boxes, each finer than the one before, as VTK's ``vtkOverlappingAMR`` and
AMReX's hierarchy of ``BoxArray``s.

The hierarchy is defined in index space alone, as AMReX defines it. A box of
level ``l`` is a cell-index box ``[lo, hi)`` of its level; level ``l + 1``'s
indices are level ``l``'s times the refinement ratio between them (one per
axis). A box is an ordinary dataset whose ``StructuredTopology.offset`` is
``lo``. Its geometry is whatever the box's is -- an image's
``UniformCoordinates``, a rectilinear grid's axes, a curvilinear grid's
positions -- since which cells and points a finer level covers is a matter of
integer boxes, not of positions. That is what VTK's images-only AMR lacks.

Making the hierarchy marks what each box holds that another holds better
(``Ghosts``):

- ``REFINEDCELL``: a cell a finer level's box covers. Its data are valid but
  superseded: filters that make new cells leave it out, and reductions skip
  it, so each region of space counts once.
- ``REFINEDPOINT`` (a Tack bit; VTK has none): a point at whose location a
  finer level has a point. A coarse point on the edge of a refined region is
  one, so each location counts once, at the finest level holding it.
- ``DUPLICATEPOINT``: a point a sibling box with a lower global index also
  holds. Sibling boxes abut, each with its own points on their shared face.

The boxes of every rank are gathered when the hierarchy is made
(``boxes(level)``), as VTK's metadata and AMReX's ``BoxArray`` are replicated,
so the marks need no other communication.
"""

import numpy as np

from tack.dataset.dataset import DataSet
from tack.dataset.ghosts import Ghosts
from tack.dataset.partitioned import PartitionedDataSet, _one_time
from tack.dataset.topology import StructuredTopology

__all__ = ["AMRDataSet"]


class AMRDataSet:
    """Levels of structured boxes, each level finer than the one before: a
    block-structured AMR hierarchy, as VTK's ``vtkOverlappingAMR`` and AMReX's.

    ``levels`` holds this rank's boxes of each level, a ``PartitionedDataSet`` or a
    list of ``DataSet``s each. Every box has a ``StructuredTopology`` (all of one
    dimension) whose ``offset`` is its first cell's index in its level, and any
    structured geometry: an image, a rectilinear grid or a curvilinear grid.
    ``ratios[l]`` refines level ``l`` into ``l + 1``: a whole number, or one per
    axis. ``domain`` is level 0's cell-index box ``(lo, hi)``, ``hi`` exclusive;
    by default its boxes' bounding box. ``world`` is where the other ranks' boxes
    are (``tack.communication.World``; the levels' own, else this process).

    Refused when made: boxes of a level that overlap; a box of level ``l + 1``
    that, coarsened, is not within level ``l``'s boxes (proper nesting); image
    boxes of a level not on one lattice (one spacing, one origin, axis-aligned:
    VTK's model); boxes with ghost cells (``DUPLICATECELL``).

    The boxes it holds are copies of those given with their marks computed
    (``REFINEDCELL``, ``REFINEDPOINT``, siblings' ``DUPLICATEPOINT``; other bits
    kept), sharing their topology, geometry and fields. Iterating over it gives
    ``(level, box)`` for this rank's boxes, coarsest level first."""

    def __init__(self, levels, ratios=(), domain=None, world=None):
        levels = list(levels)
        if not levels:
            raise ValueError("an AMRDataSet has at least one level")
        if world is None:
            world = next((lvl.world for lvl in levels if isinstance(lvl, PartitionedDataSet)),
                         None)
        if world is None:
            from tack.communication import World
            world = World()
        self.world = world
        given = [list(lvl) for lvl in levels]
        self.dimension = self._dimension(given)
        self.ratios = self._ratios(ratios, len(given))
        boxes, self._first = [], []
        for level, pieces in enumerate(given):
            mine = np.array([_box(piece, self.dimension) for piece in pieces],
                            np.int64).reshape(-1, 2, 3)
            every = world._allgather(mine)
            self._first.append(int(sum(len(part) for part in every[:world.rank])))
            boxes.append(np.concatenate(every).reshape(-1, 2, 3))
        self._boxes = boxes
        #: Each box's arrays' cell-index box: the box itself, or grown by ghost
        #: layers (``ghost_cells``).
        self._extents = [b.copy() for b in boxes]
        self.layers = 0
        self._fills = None
        self.domain = self._domain(domain)
        self._check(given)
        self.levels = tuple(PartitionedDataSet(self._marked(level, pieces), world)
                            for level, pieces in enumerate(given))

    @classmethod
    def _grown(cls, base, levels, extents, layers, fills):
        """A hierarchy of ``base``'s boxes grown by ghost layers: ``levels`` the grown
        boxes (marks made), ``extents`` their cell-index boxes, ``fills`` what
        refreshes their ghosts."""
        out = cls.__new__(cls)
        out.world, out.dimension, out.ratios = base.world, base.dimension, base.ratios
        out._boxes, out._first, out.domain = base._boxes, base._first, base.domain
        out._extents, out.layers, out._fills = extents, layers, fills
        out.levels = tuple(PartitionedDataSet(pieces, base.world) for pieces in levels)
        return out

    def level_domain(self, level):
        """``domain`` refined to ``level``: its cell-index box there."""
        scale = np.ones(3, np.int64)
        for r in self.ratios[:level]:
            scale *= np.asarray(r, np.int64)
        lo, hi = (np.asarray(c, np.int64) for c in self.domain)
        return lo * scale, hi * scale

    def _refined(self, level, lo, hi):
        """Which cells and points of the cell-index box ``[lo, hi)`` of ``level`` a
        finer level covers: two bool arrays shaped z, y, x (``dimension`` axes)."""
        box = np.stack([np.asarray(lo, np.int64), np.asarray(hi, np.int64)])[None]
        return self._refined_boxes(level, box)[0]

    def _refined_boxes(self, level, boxes):
        """``_refined`` of each cell-index box of ``boxes`` (``(n, 2, 3)``), every box
        at once: the boxes and the finer level's that meet them are found by
        ``_meeting``, so the work is in the pairs that meet, not in every pair (a
        loop over all the children for each box took 129 s for AMR-Wind's 6,436
        boxes)."""
        d = self.dimension
        lo, hi = boxes[:, 0, :d], boxes[:, 1, :d]
        out = [(np.zeros(tuple(int(v) for v in (b - a)[::-1]), bool),
                np.zeros(tuple(int(v) + 1 for v in (b - a)[::-1]), bool))
               for a, b in zip(lo, hi)]
        if level + 1 < self.num_levels:
            ratio = np.asarray(self.ratios[level], np.int64)[:d]
            children = self._boxes[level + 1][:, :, :d]
            # A cell refined: a child, coarsened, covers it.
            clo, chi = children[:, 0] // ratio, -(-children[:, 1] // ratio)
            for i, a, b in _cut(lo, hi, clo, chi, closed=False):
                out[i][0][_slices(a - lo[i], b - lo[i])] = True
            # A point refined: its location p * ratio is one of the child's points.
            plo, phi = -(-children[:, 0] // ratio), children[:, 1] // ratio
            for i, a, b in _cut(lo, hi, plo, phi, closed=True):
                out[i][1][_slices(a - lo[i], b - lo[i] + 1)] = True
        return out

    # -- what it holds ---------------------------------------------------

    @property
    def num_levels(self):
        return len(self._boxes)

    def boxes(self, level):
        """Every rank's boxes of ``level``, an ``(n, 2, 3)`` integer array of cell-index
        boxes ``[lo, hi)``, in order of global index: rank 0's first, each rank's
        in the order given. Axes a box does not have are ``[0, 1)``."""
        return self._boxes[level].copy()

    def __iter__(self):
        for level, pieces in enumerate(self.levels):
            for piece in pieces:
                yield level, piece

    def __repr__(self):
        counts = [len(self._boxes[level]) for level in range(self.num_levels)]
        return f"AMRDataSet({self.num_levels} levels, boxes {counts})"

    def _as_pieces(self):
        """This rank's boxes of every level, as one ``PartitionedDataSet``: what a
        reduction over the hierarchy reduces, each box's marks saying what it
        owns."""
        return PartitionedDataSet([box for _, box in self], self.world)

    # -- checks ----------------------------------------------------------

    def _dimension(self, given):
        mine = {len(piece.topology.point_dims) for pieces in given for piece in pieces
                if isinstance(piece.topology, StructuredTopology)}
        for level, pieces in enumerate(given):
            for k, piece in enumerate(pieces):
                if not isinstance(piece, DataSet) or \
                        not isinstance(piece.topology, StructuredTopology):
                    raise TypeError(f"box {k} of level {level} is not a dataset of a "
                                    f"StructuredTopology")
        every = set().union(*self.world._allgather(sorted(mine)))
        if len(every) > 1:
            raise ValueError(f"an AMRDataSet's boxes are of one dimension, not "
                             f"{sorted(every)}")
        if not every:
            raise ValueError("an AMRDataSet needs a box somewhere")
        return every.pop()

    def _ratios(self, ratios, count):
        ratios = list(ratios)
        if len(ratios) == count:            # VTK gives the finest level one too
            ratios = ratios[:-1]
        if len(ratios) != count - 1:
            raise ValueError(f"{count} levels take {count - 1} refinement ratios, not "
                             f"{len(ratios)}")
        out = []
        for r in ratios:
            values = np.atleast_1d(np.asarray(r))
            if values.size == 1:
                values = np.full(self.dimension, values[0])
            elif values.size == 3 and (values[self.dimension:] == 1).all():
                values = values[:self.dimension]      # as ``ratios`` gives them back
            if values.size != self.dimension or values.dtype.kind not in "iu" or \
                    (values < 1).any():
                raise ValueError(f"a refinement ratio is a whole number of at least 1, or "
                                 f"one per axis ({self.dimension}), not {r!r}")
            out.append(tuple(int(v) for v in values) + (1,) * (3 - self.dimension))
        return tuple(out)

    def _domain(self, domain):
        level0 = self._boxes[0]
        if domain is None:
            return (tuple(int(v) for v in level0[:, 0].min(axis=0)),
                    tuple(int(v) for v in level0[:, 1].max(axis=0)))
        lo, hi = (np.asarray(corner, np.int64) for corner in domain)
        lo = np.concatenate([lo, np.zeros(3 - lo.size, np.int64)])
        hi = np.concatenate([hi, np.ones(3 - hi.size, np.int64)])
        if (level0[:, 0] < lo).any() or (level0[:, 1] > hi).any():
            raise ValueError(f"level 0's boxes reach outside the domain {domain!r}")
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)

    def _check(self, given):
        from tack.dataset import units

        spatial = self.world._allgather(sorted({piece.unit or "" for pieces in given
                                                for piece in pieces}))
        units.common("an AMRDataSet's boxes' geometry",
                     [unit or None for part in spatial for unit in part])
        _one_time("an AMRDataSet's boxes", [t for part in self.world._allgather(
            sorted({(piece.time, piece.time_unit) for pieces in given for piece in pieces},
                   key=repr)) for t in part])
        for level, pieces in enumerate(given):
            for k, piece in enumerate(pieces):
                marks = piece.ghosts
                if marks is not None and marks.cells is not None and \
                        (marks.cells.numpy() & Ghosts.DUPLICATECELL).any():
                    raise ValueError(f"box {k} of level {level} has ghost cells "
                                     "(DUPLICATECELL): give the boxes without them")
        for level, boxes in enumerate(self._boxes):
            first = _overlap(boxes)
            if first is not None:
                a, b = first
                raise ValueError(f"boxes {a} and {b} of level {level} overlap: {_text(boxes[a])} "
                                 f"and {_text(boxes[b])}")
        for level in range(1, self.num_levels):
            ratio = np.asarray(self.ratios[level - 1], np.int64)
            coarse = self._boxes[level - 1]
            fine = self._boxes[level]
            lo, hi = fine[:, 0] // ratio, -(-fine[:, 1] // ratio)
            i, j = _meeting(lo, hi, coarse[:, 0], coarse[:, 1], closed=False)
            inside = (np.minimum(coarse[j, 1], hi[i]) - np.maximum(coarse[j, 0], lo[i])).prod(axis=1)
            covered = np.bincount(i, inside, minlength=len(fine))
            short = np.flatnonzero(covered != (hi - lo).prod(axis=1))
            if len(short):
                k = int(short[0])
                raise ValueError(f"box {k} of level {level} ({_text(fine[k])}) is not within "
                                 f"level {level - 1}'s boxes when coarsened by "
                                 f"{tuple(ratio[:self.dimension])}: the hierarchy is not "
                                 f"properly nested")
        for level, pieces in enumerate(given):
            lattices = self.world._allgather([_lattice(piece, self.dimension)
                                              for piece in pieces])
            lattices = [entry for part in lattices for entry in part if entry is not None]
            if len(lattices) < 2:
                continue
            spacings = np.array([np.asarray(entry[0], np.float64) for entry in lattices])
            origins = np.array([np.asarray(entry[1], np.float64) for entry in lattices])
            ref_spacing, ref_origin = spacings[0], origins[0]
            tolerance = 1e-6 * float(np.max(ref_spacing))
            # ``np.allclose`` of each against the first, every box at once.
            same = (np.abs(spacings - ref_spacing) <= 1e-9 * np.abs(ref_spacing)).all(axis=1) \
                & (np.abs(origins - ref_origin) <= tolerance).all(axis=1)
            if not same.all():
                k = int(np.flatnonzero(~same)[0])
                spacing, origin = spacings[k], origins[k]
                raise ValueError(
                    f"the image boxes of level {level} are not on one lattice: spacing "
                    f"{tuple(spacing)} and origin {tuple(origin)} against "
                    f"{tuple(ref_spacing)} and {tuple(ref_origin)} (VTK's model: one "
                    "spacing per level and one origin)")

    # -- the marks -------------------------------------------------------

    def _marked(self, level, pieces):
        """This rank's boxes of ``level``, ``pieces``, with their marks computed."""
        d = self.dimension
        first = self._first[level]
        every = self._boxes[level]
        mine = every[first:first + len(pieces)]
        refined = self._refined_boxes(level, mine)
        # Siblings' shared faces, edges and corners: the lower-numbered sibling owns them.
        lo, hi = mine[:, 0, :d], mine[:, 1, :d]
        lower = [[] for _ in pieces]
        for k, a, b, j in _cut(lo, hi, every[:, 0, :d], every[:, 1, :d], closed=True,
                               partners=True):
            if j < first + k:
                lower[k].append((a, b))
        cleared_cells = 0xFF & ~Ghosts.REFINEDCELL
        cleared_points = 0xFF & ~(Ghosts.REFINEDPOINT | Ghosts.DUPLICATEPOINT)
        out = []
        for k, piece in enumerate(pieces):
            refined_cells, refined_points = refined[k]
            cells = refined_cells.astype(np.uint8) * np.uint8(Ghosts.REFINEDCELL)
            points = refined_points.astype(np.uint8) * np.uint8(Ghosts.REFINEDPOINT)
            for a, b in lower[k]:
                points[_slices(a - lo[k], b - lo[k] + 1)] |= Ghosts.DUPLICATEPOINT
            marks = piece.ghosts
            if marks is not None and marks.cells is not None:
                cells |= (marks.cells.numpy() & cleared_cells).reshape(cells.shape)
            if marks is not None and marks.points is not None:
                points |= (marks.points.numpy() & cleared_points).reshape(points.shape)
            with piece.device:
                ghosts = Ghosts(cells=cells.reshape(-1), points=points.reshape(-1))
            out.append(DataSet(piece.topology, piece.geometry, fields=dict(piece.fields),
                               sets=dict(piece.sets), dtype=piece.dtype, ghosts=ghosts,
                               global_ids=piece.global_ids, **piece._frame()))
        return out


def _box(piece, dimension):
    """A box's cell-index box in its level, three axes, ``[lo, hi)``."""
    topology = piece.topology
    lo = np.zeros(3, np.int64)
    hi = np.ones(3, np.int64)
    n = len(topology.point_dims)
    lo[:n] = topology.offset
    hi[:n] = np.asarray(topology.offset) + np.asarray(topology.point_dims) - 1
    return lo, hi


def _coarsened(box, ratio):
    """The coarser level's cells a box covers: ``(lo, hi)``, three axes."""
    ratio = np.asarray(ratio, np.int64)
    return box[0] // ratio, -(-box[1] // ratio)


def _overlap(boxes):
    """The first two boxes that share a cell, or None."""
    i, j = _meeting(boxes[:, 0], boxes[:, 1], boxes[:, 0], boxes[:, 1], closed=False)
    keep = i < j
    if not keep.any():
        return None
    i, j = i[keep], j[keep]
    first = np.lexsort((j, i))[0]
    return int(i[first]), int(j[first])


def _meeting(alo, ahi, blo, bhi, closed):
    """Every pair ``(i, j)`` of box ``i`` of ``a`` and box ``j`` of ``b`` that meet, as
    two arrays ordered by ``i``: ``[lo, hi)`` boxes sharing a cell, or with
    ``closed``, ``[lo, hi]`` boxes sharing a point. Arrays ``(n, axes)``. ``b`` is
    sorted on its first axis, and each box of ``a`` tests only the boxes whose
    start is within the widest of ``b`` of its own (a sweep), not every box."""
    empty = np.zeros(0, np.int64)
    if not len(alo) or not len(blo):
        return empty, empty
    order = np.argsort(blo[:, 0], kind="stable")
    starts = blo[order, 0]
    width = max(int((bhi[:, 0] - blo[:, 0]).max()), 0)
    if closed:      # blo <= ahi and bhi >= alo, with bhi <= blo + width
        first = np.searchsorted(starts, alo[:, 0] - width, side="left")
        stop = np.searchsorted(starts, ahi[:, 0], side="right")
    else:           # blo < ahi and bhi > alo
        first = np.searchsorted(starts, alo[:, 0] - width, side="right")
        stop = np.searchsorted(starts, ahi[:, 0], side="left")
    counts = np.maximum(stop - first, 0)
    i = np.repeat(np.arange(len(alo)), counts)
    within = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    j = order[np.repeat(first, counts) + within]
    low = np.maximum(alo[i], blo[j])
    high = np.minimum(ahi[i], bhi[j])
    meet = (low <= high).all(axis=1) if closed else (low < high).all(axis=1)
    return i[meet], j[meet]


def _cut(alo, ahi, blo, bhi, closed, partners=False):
    """For each pair ``_meeting`` finds, ``(i, low, high)``: box ``i`` of ``a`` and
    where it meets the other box (``(i, low, high, j)`` with ``partners``)."""
    i, j = _meeting(alo, ahi, blo, bhi, closed)
    low = np.maximum(alo[i], blo[j])
    high = np.minimum(ahi[i], bhi[j])
    if partners:
        return zip(i.tolist(), low, high, j.tolist())
    return zip(i.tolist(), low, high)


def _slices(a, b):
    """Index slices of an array shaped z, y, x for ``[a, b)`` given x first."""
    return tuple(slice(int(s), int(e)) for s, e in zip(a[::-1], b[::-1]))


def _lattice(piece, dimension):
    """An image box's lattice, ``(spacing, origin)`` in world coordinates, where its
    level's index 0 is; None for other geometry. Oriented images are refused:
    an AMR image is axis-aligned, as VTK's are."""
    from tack.arrays import UniformCoordinates

    coordinates = piece.geometry.values
    if not isinstance(coordinates, UniformCoordinates):
        return None
    if coordinates.ORIENTED:
        raise ValueError("an AMR image box is axis-aligned: give it no direction matrix")
    spacing = np.asarray(coordinates._spacing, np.float64)[:dimension]
    offset = np.asarray(piece.topology.offset, np.float64)
    corner = np.asarray(coordinates._origin, np.float64)[:dimension] + \
        np.asarray(piece.origin, np.float64)[:dimension]
    return spacing, corner - offset * spacing


def _text(box):
    return f"[{tuple(int(v) for v in box[0])}, {tuple(int(v) for v in box[1])})"


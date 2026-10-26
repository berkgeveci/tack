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
        self.levels = tuple(PartitionedDataSet([self._marked(level, k, piece)
                                                for k, piece in enumerate(pieces)], world)
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
        d = self.dimension
        lo, hi = np.asarray(lo)[:d], np.asarray(hi)[:d]
        cells = np.zeros(tuple(int(v) for v in (hi - lo)[::-1]), bool)
        points = np.zeros(tuple(int(v) + 1 for v in (hi - lo)[::-1]), bool)
        if level + 1 < self.num_levels:
            ratio = np.asarray(self.ratios[level])[:d]
            for child in self._boxes[level + 1]:
                clo, chi = _coarsened(child, self.ratios[level])
                a, b = np.maximum(clo[:d], lo), np.minimum(chi[:d], hi)
                if (a < b).all():
                    cells[_slices(a - lo, b - lo)] = True
                # A point refined: its location p * ratio is one of the child's points.
                a = np.maximum(-(-child[0, :d] // ratio), lo)
                b = np.minimum(child[1, :d] // ratio, hi)
                if (a <= b).all():
                    points[_slices(a - lo, b - lo + 1)] = True
        return cells, points

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
            ratio = np.asarray(self.ratios[level - 1])
            coarse = self._boxes[level - 1]
            for k, box in enumerate(self._boxes[level]):
                lo, hi = _coarsened(box, ratio)
                need = int(np.prod(hi - lo))
                inside = np.maximum(np.minimum(coarse[:, 1], hi) - np.maximum(coarse[:, 0], lo), 0)
                if int(inside.prod(axis=1).sum()) != need:
                    raise ValueError(f"box {k} of level {level} ({_text(box)}) is not within "
                                     f"level {level - 1}'s boxes when coarsened by "
                                     f"{tuple(ratio[:self.dimension])}: the hierarchy is not "
                                     f"properly nested")
        for level, pieces in enumerate(given):
            lattices = self.world._allgather([_lattice(piece, self.dimension)
                                              for piece in pieces])
            lattices = [entry for part in lattices for entry in part if entry is not None]
            for spacing, origin in lattices[1:]:
                ref_spacing, ref_origin = lattices[0]
                tolerance = 1e-6 * float(np.max(ref_spacing))
                if not (np.allclose(spacing, ref_spacing, rtol=1e-9, atol=0.0)
                        and np.allclose(origin, ref_origin, rtol=0.0, atol=tolerance)):
                    raise ValueError(
                        f"the image boxes of level {level} are not on one lattice: spacing "
                        f"{tuple(spacing)} and origin {tuple(origin)} against "
                        f"{tuple(ref_spacing)} and {tuple(ref_origin)} (VTK's model: one "
                        "spacing per level and one origin)")

    # -- the marks -------------------------------------------------------

    def _marked(self, level, k, piece):
        """``piece``, box ``k`` of this rank's ``level``, with its marks computed."""
        d = self.dimension
        box = self._boxes[level][self._first[level] + k]
        lo, hi = box[0, :d], box[1, :d]
        cell_shape = tuple(int(v) for v in (hi - lo)[::-1])
        point_shape = tuple(int(v) + 1 for v in (hi - lo)[::-1])
        refined_cells, refined_points = self._refined(level, box[0], box[1])
        cells = np.where(refined_cells, Ghosts.REFINEDCELL, 0).astype(np.uint8)
        points = np.where(refined_points, Ghosts.REFINEDPOINT, 0).astype(np.uint8)
        index = self._first[level] + k
        for other in self._boxes[level][:index]:
            a, b = np.maximum(other[0, :d], lo), np.minimum(other[1, :d], hi)
            if (a <= b).all():                       # a shared face, edge or corner
                points[_slices(a - lo, b - lo + 1)] |= Ghosts.DUPLICATEPOINT
        cleared_cells = 0xFF & ~Ghosts.REFINEDCELL
        cleared_points = 0xFF & ~(Ghosts.REFINEDPOINT | Ghosts.DUPLICATEPOINT)
        marks = piece.ghosts
        if marks is not None and marks.cells is not None:
            cells |= (marks.cells.numpy() & cleared_cells).reshape(cell_shape)
        if marks is not None and marks.points is not None:
            points |= (marks.points.numpy() & cleared_points).reshape(point_shape)
        with piece.device:
            ghosts = Ghosts(cells=cells.reshape(-1), points=points.reshape(-1))
        return DataSet(piece.topology, piece.geometry, fields=dict(piece.fields),
                       sets=dict(piece.sets), dtype=piece.dtype, ghosts=ghosts,
                       global_ids=piece.global_ids, **piece._frame())


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
    for a in range(len(boxes) - 1):
        rest = boxes[a + 1:]
        inside = np.minimum(rest[:, 1], boxes[a, 1]) - np.maximum(rest[:, 0], boxes[a, 0])
        hit = np.flatnonzero((inside > 0).all(axis=1))
        if len(hit):
            return a, a + 1 + int(hit[0])
    return None


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


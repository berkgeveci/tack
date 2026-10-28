"""Averaging over each point's cells, through the point links: ``gradients(at="points")``,
``to_points`` and ``surface_normals``.

A point's links are a segment of the links' entries (``point_links``: CSR by
point), so a point's mean is a segmented mean, ``tack.algorithms.segment_reduce``
with ``op="mean"`` (and ``empty="keep"``: a point without cells keeps its
value). The values are read through the entries, as an
implicit array, never gathered: a point of up to 256 links sums them in the
links' order, one thread a point, as the loop this replaced did, so its mean
is the same bits; a point of more -- an apex or pole every cell of a fan or
graded mesh meets -- is summed by chunks over several threads, where one
thread walked them all."""

import tack
from tack.algorithms import segment_reduce
from tack.arrays import Permutation, _Implicit, _metadata, dtype_of, size_of


def _average_links(offsets, entries, contributions, out):
    """``out[p]``: the mean of ``contributions[entries[i]]`` over point ``p``'s links
    ``offsets[p]:offsets[p + 1]``; a point without links keeps its value."""
    if not size_of(out):
        return
    segment_reduce(Permutation(entries, contributions), offsets, op="mean", out=out,
                   empty="keep")


@tack.device_class
class _EntryCells(_Implicit):
    """The cell of each link entry, an entry being a cell's index in ``cells`` (a group
    view of one shape) times its corners plus the corner."""

    def __init__(self, entries, cells):
        self.entries = entries
        self.cells = cells
        _metadata(self, size_of(entries), None, dtype_of(entries))

    @tack.func
    def get(self, k):
        return self.cells.entity_id(self.entries[k] // self.cells.NUM_POINTS)


def _average_cell_values(offsets, entries, cells, values, out):
    """``out[p]``: the mean of the values of the cells around point ``p`` (``cells`` a
    group view of one shape, ``values`` one a cell), read through the links
    without a value a corner (8 a hexahedron: 640 MB at 20M)."""
    if not size_of(out):
        return
    segment_reduce(Permutation(_EntryCells(entries, cells), values), offsets, op="mean",
                   out=out, empty="keep")

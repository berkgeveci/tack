"""Partitioned datasets: a dataset in pieces, as VTK's ``vtkPartitionedDataSet``.

A ``PartitionedDataSet`` is an ordered list of ``DataSet``s. Each piece is a
whole dataset -- its own topology, numbering, geometry, fields, sets and
device -- so work that needs one piece at a time takes the piece: the caller
iterates over them. Only algorithms whose answer depends on more than one
piece take the partitioned dataset itself: ``statistics`` and ``entropy``
combine what each piece holds, and ``connected_components``, ``probe``,
``advect``, ``outliers``, ``voxel_downsample``, ``deposit`` and
``halo_catalog`` combine what pieces hold near each other. ``split`` makes one from a structured grid,
and ``tack.interop.vtk`` converts VTK's partitioned datasets and collections.

Every cell and point that several pieces hold has one owner; the other
copies are ghosts, marked in each piece's ``ghosts``
(``tack.dataset.ghosts``), and reductions count only what a piece owns.
"""

from tack.dataset.dataset import DataSet

__all__ = ["PartitionedDataSet"]


class PartitionedDataSet:
    """A dataset in pieces: an ordered list of ``DataSet``s (``partitions``), each
    with its own topology, fields, sets and device, as VTK's
    ``vtkPartitionedDataSet``. A piece may have no cells. Pieces may be on
    different devices; an algorithm given the partitioned dataset runs on each
    piece where the piece is.

    It is a container, not a dataset: iterate over it (``for piece in pds``),
    index it (``pds[i]``) or take its length.

    Across MPI ranks, it holds this rank's pieces, and ``world`` (a
    ``tack.communication.World``; this process alone by default) is where the
    others are, for the algorithms that combine pieces."""

    def __init__(self, partitions=(), world=None):
        partitions = tuple(partitions)
        for i, piece in enumerate(partitions):
            if not isinstance(piece, DataSet):
                raise TypeError(f"partition {i} is a {type(piece).__name__}, not a "
                                "tack.dataset.DataSet")
        self._partitions = partitions
        if world is None:
            from tack.communication import World
            world = World()
        self.world = world

    @property
    def partitions(self):
        """The pieces, a tuple of ``DataSet``s."""
        return self._partitions

    def __len__(self):
        return len(self._partitions)

    def __iter__(self):
        return iter(self._partitions)

    def __getitem__(self, index):
        return self._partitions[index]

    def __repr__(self):
        cells = sum(piece.num_cells for piece in self._partitions)
        return (f"PartitionedDataSet({len(self._partitions)} partitions, "
                f"{cells} cells)")


def fields_of(data, field, what):
    """The field named ``field`` in each piece of ``data``, for an algorithm that
    combines them (``what``): a name, since a ``Field`` belongs to one piece's
    topology; a piece without it is refused."""
    from tack.dataset.dataset import is_field_key

    if not is_field_key(field):
        raise TypeError(f"{what} of a PartitionedDataSet takes a field's name (or name "
                        "and space): a Field belongs to one partition's topology")
    for i, piece in enumerate(data):
        if field not in piece.fields:
            raise KeyError(f"partition {i} has no field {field!r}: it has "
                           f"{', '.join(map(repr, piece.fields)) or 'none'}")
    return field


def common_origin(data, what, world=None):
    """The origin every piece of ``data`` has, on every rank of ``world``
    (``data.world`` by default): ``what`` (an algorithm that compares or moves
    positions between pieces) refuses pieces on two origins, in two spatial
    units, or at two times. ``(0.0, 0.0, 0.0)``
    with no pieces anywhere."""
    from tack.dataset import units

    world = data.world if world is None else world
    mine = sorted({piece.origin for piece in data})
    spatial = sorted({piece.unit or "" for piece in data})
    times = sorted({(piece.time, piece.time_unit) for piece in data}, key=repr)
    gathered = world._allgather((mine, spatial, times))
    every = sorted({o for part, _, _ in gathered for o in part})
    units.common(f"{what}'s pieces' geometry",
                 [u or None for _, part, _ in gathered for u in part])
    _one_time(f"{what}'s pieces", [t for _, _, part in gathered for t in part])
    if len(every) > 1:
        raise ValueError(f"{what} needs its pieces on one origin, not {every}: rebase "
                         "them to one first (DataSet.rebased)")
    return every[0] if every else (0.0, 0.0, 0.0)


def _one_time(what, times):
    """Refuse ``times`` (``(time, time_unit)`` pairs) of more than one step: pieces of
    one dataset are at one time, compared in one unit where both are known. A
    piece with no time is not checked."""
    import math

    from tack.dataset import units

    known = sorted({(t, u) for t, u in times if t is not None}, key=repr)
    for t, u in known[1:]:
        t0, u0 = known[0]
        if u is not None and u0 is not None and u != u0:
            factor, offset = units.conversion(u, u0)
            t = t * factor + offset
        if not math.isclose(t, t0, rel_tol=1e-12):
            raise ValueError(f"{what} are at two times, {known[0]} and {(t, u)}: pieces of "
                             "one step have one time")

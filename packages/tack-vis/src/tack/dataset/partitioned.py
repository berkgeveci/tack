"""Partitioned datasets: a dataset in pieces, as VTK's ``vtkPartitionedDataSet``.

A ``PartitionedDataSet`` is an ordered list of ``DataSet``s. Each piece is a
whole dataset -- its own topology, numbering, geometry, fields, sets and
device -- so work that needs one piece at a time takes the piece: the caller
iterates over them. Only algorithms whose answer depends on more than one
piece take the partitioned dataset itself: ``statistics`` and ``entropy``
combine what each piece holds. ``split`` makes one from a structured grid,
and ``tack.interop.vtk`` converts VTK's partitioned datasets and collections.

A point on the boundary between two pieces belongs to both, so a reduction
over point data counts it once per piece; ghost marks will say which piece
owns it.
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
    index it (``pds[i]``) or take its length."""

    def __init__(self, partitions=()):
        partitions = tuple(partitions)
        for i, piece in enumerate(partitions):
            if not isinstance(piece, DataSet):
                raise TypeError(f"partition {i} is a {type(piece).__name__}, not a "
                                "tack.dataset.DataSet")
        self._partitions = partitions

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
    if not isinstance(field, str):
        raise TypeError(f"{what} of a PartitionedDataSet takes a field's name: a Field "
                        "belongs to one partition's topology")
    for i, piece in enumerate(data):
        if field not in piece.fields:
            raise KeyError(f"partition {i} has no field {field!r}: it has "
                           f"{', '.join(map(repr, piece.fields)) or 'none'}")
    return field

"""``boundary_faces``: the faces with one side, as a face ``Set``."""

from tack.data.sets import Set

__all__ = ['boundary_faces']


def boundary_faces(data):
    """The faces with one side -- the boundary -- as a face ``Set``. Name it in
    ``data.sets`` to iterate over it or carry it through filters."""
    return Set._from_sorted_ids(data.topology, "faces", data.topology.faces().boundary())

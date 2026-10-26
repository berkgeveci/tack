"""``boundary_faces``: the faces with one side, as a face ``Set``."""

from tack.dataset.sets import Set
from tack.runtime.dispatch import on_inputs_device

__all__ = ['boundary_faces']


@on_inputs_device
def boundary_faces(data):
    """The faces with one side -- the boundary -- as a face ``Set``: a 3D mesh's
    boundary faces, a 2D mesh's boundary edges, a polyline mesh's free ends,
    numbered as ``data.topology.faces()`` numbers them. Name it in ``data.sets``
    (``data.sets["boundary"] = boundary_faces(data)``) to iterate over it or
    carry it through filters; ``extract_surface`` makes a dataset of it."""
    return Set._from_sorted_ids(data.topology, "faces", data.topology.faces().boundary())

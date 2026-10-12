"""``external_faces``: the faces of the 3D cells that only one cell has, as a surface dataset: the boundary face set (kept in ``data.sets[name]``) through ``extract_surface``. Each face faces out of its cell and carries that cell's cell fields; point fields stay on the same points."""

from tack.data.algorithms.boundary_faces import boundary_faces
from tack.data.algorithms.extract_surface import extract_surface

__all__ = ['external_faces']


def external_faces(data, fields=None):
    """The faces of the 3D cells that only one cell has, as a surface dataset:
    ``extract_surface`` of ``boundary_faces``. Each face faces out of its cell and
    carries that cell's cell fields; point fields stay on the same points."""
    return extract_surface(data, boundary_faces(data), fields=fields)

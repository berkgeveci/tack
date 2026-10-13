"""``external_faces``: the boundary -- the faces that only one cell has -- as a dataset, through ``extract_surface``."""

from tack.data.algorithms.boundary_faces import boundary_faces
from tack.data.algorithms.extract_surface import extract_surface

__all__ = ['external_faces']


def external_faces(data, fields=None):
    """The faces that only one cell has, as a dataset: ``extract_surface`` of
    ``boundary_faces`` -- a 3D mesh's boundary surface, a 2D mesh's boundary
    lines, a polyline mesh's loose ends as vertices. Each face faces out of its
    cell and carries that cell's cell fields; point fields stay on the same
    points."""
    return extract_surface(data, boundary_faces(data), fields=fields)

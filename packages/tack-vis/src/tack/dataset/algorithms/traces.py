"""``traces``: ``field`` -- any field with a basis on the cells: ``H1``, ``L2`` (an order per cell too), ``Constant`` -- evaluated by each 3D cell at each of its faces' points: a ``SideTraces`` field. Each cell writes its own side, at the face's points in the face's own order (``face_position``), so the two sides of a face line up point by point. Boundary faces leave side 1 zero."""

import tack
from tack.dataset import arrays
from tack.dataset.algorithms._common import _field
from tack.dataset.dataset import Field
from tack.dataset.spaces import SideTraces, reference_cells

__all__ = ['traces']


@tack.kernel
def _traces(cells, u, out):
    for c in cells:
        for f in range(cells.NUM_FACES):
            first = (cells.face_id(c, f) * 2 + cells.face_side(c, f)) * 4
            for k in range(cells.face_num_points(f)):
                pc = cells.parametric_point(cells.face_corner(f, k))
                out[first + cells.face_position(c, f, k)] = u.value(c, pc)


def traces(data, field):
    """``field`` -- any field with a basis on the cells: ``H1``, ``L2`` (an order per
    cell too), ``Constant`` -- evaluated by each 3D cell at each of its faces'
    points: a ``SideTraces`` field. Each cell writes its own side, at the face's
    points in the face's own order (``face_position``), so the two sides of a
    face line up point by point. Boundary faces leave side 1 zero."""
    field = _field(data, field)
    space = field.space
    if not space.interpolated or space.on not in ("cells", "points"):
        raise TypeError(f"traces are taken from a field with a basis on the cells, "
                        f"not {space!r}")
    reference_cells(data.topology, "traces")
    data.topology.faces()
    out_space = SideTraces(data)
    width = arrays.width_of(field.values)
    out = (tack.Vector.field(width, arrays.dtype_of(field.values), shape=(out_space.size,))
           if width else tack.field(arrays.dtype_of(field.values), shape=(out_space.size,)))
    if out_space.size:
        out.fill(0)
    starts = data.topology.faces().group_starts
    for group in data.launch_groups("cells", [field]):
        if group.count and id(group.root) in starts:
            _traces(data.domain_view("cells", group), field.view(group), out)
    return Field(out_space, out)

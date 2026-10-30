"""``jump``: a field's difference across each face at the face's center, side 1 minus side 0 (zero on the boundary): a field on faces. Each side is the field as its own cell's basis has it at the face's center, so a DG field's jumps are its own, of any order, not its cells' averages; a continuous field's are zero. Cell data (``Constant``, values on cells) is read from each side's cell directly, on any topology, polyhedral included."""

import tack
from tack.arrays import materialize
from tack.dataset import units
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms._faces import _on_cells
from tack.dataset.algorithms.traces import traces
from tack.dataset.carry import _like
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values, reference_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['jump']


@tack.kernel
def _side_centers(cells, u, out):
    # Each cell's value at each of its faces' centres -- the mean of the face's
    # corners in the cell's reference element, which both cells map to the same
    # point -- in its side's slot: the field as the cell's own basis has it there.
    for c in cells:
        for k in range(cells.NUM_FACES):
            n = cells.face_num_points(k)
            pc = cells.parametric_point(cells.face_corner(k, 0))
            for i in range(1, n):
                pc += cells.parametric_point(cells.face_corner(k, i))
            out[2 * cells.face_id(c, k) + cells.face_side(c, k)] = u.value(c, pc / n)


@tack.kernel
def _trace_jump(faces, t, out):
    for f in faces:
        pc = faces.parametric_center()
        inside = t.value(f, 0, pc)
        out[faces.entity_id(f)] = (t.value(f, 1, pc) - inside if faces.num_sides(f) == 2
                                   else inside * 0.0)


@tack.kernel
def _jump(faces, sides, out):
    for f in faces:
        f_id = faces.entity_id(f)
        inside = sides[2 * f_id]
        out[f_id] = sides[2 * f_id + 1] - inside if faces.num_sides(f) == 2 else inside * 0.0


@tack.kernel
def _cell_jump(faces, values, out):
    for f in faces:
        inside = values[faces.side_cell(f, 0)]
        out[faces.entity_id(f)] = (values[faces.side_cell(f, 1)] - inside
                                   if faces.num_sides(f) == 2 else inside * 0.0)


@on_inputs_device
def jump(data, field):
    """A field's difference across each face at the face's center, side 1 minus side
    0 (zero on the boundary): a field on faces. Each side is the field as its own
    cell's basis has it at the face's center (the mean of the face's corners in
    the cell's reference element, the same point from both sides), so a DG
    field's jumps are its own, of any order, not its cells' averages; a
    continuous field's are zero. Up to order 1 the face's corner values
    (``traces``) give the center exactly; above it, the corners' linear mean was
    wrong, and each side's basis is evaluated there. Cell data (``Constant``, values on
    cells) is read from each side's cell directly, on any topology, polyhedral
    included."""
    field = _field(data, field)
    faces = data.topology.faces()
    out = _like(field.values, faces.num_faces)
    if _on_cells(data, field):
        for_each(_cell_jump, data, "faces", materialize(field.values), out)
    elif getattr(field.space, "order", 1) <= 1 or getattr(field.space, "varies", False):
        # Up to order 1 a side is linear over the face, so its corner values
        # (traces) give the center exactly -- on a pyramid's triangles too, whose
        # map from the reference is not affine.
        for_each(_trace_jump, data, "faces", traces(data, field), out)
    else:
        reference_cells(data.topology, "jump")
        sides = _like(field.values, 2 * faces.num_faces)
        if faces.num_faces:
            sides.fill(0)
        for group in data.launch_groups("cells", [field]):
            if group.count:
                _side_centers(data.domain_view("cells", group), field.view(group), sides)
        for_each(_jump, data, "faces", sides, out)
    return Field(Values(data, "faces"), out, units.delta(field.unit))

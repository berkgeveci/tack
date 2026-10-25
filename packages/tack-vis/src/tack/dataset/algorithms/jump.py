"""``jump``: a field's difference across each face at the face's center, side 1 minus side 0 (zero on the boundary): a field on faces. Each side is the field as its own cell has it -- ``traces`` -- so a DG field's jumps are its own, not its cells' averages; a continuous field's are zero. Cell data (``Constant``, values on cells) is read from each side's cell directly, on any topology, polyhedral included."""

import tack
from tack.arrays import materialize
from tack.dataset import units
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms._faces import _on_cells
from tack.dataset.algorithms.traces import traces
from tack.dataset.carry import _like
from tack.dataset.dataset import Field, for_each
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['jump']


@tack.kernel
def _jump(faces, t, out):
    for f in faces:
        pc = faces.parametric_center()
        inside = t.value(f, 0, pc)
        out[faces.entity_id(f)] = (t.value(f, 1, pc) - inside if faces.num_sides(f) == 2
                                   else inside * 0.0)


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
    cell has it -- ``traces`` -- so a DG field's jumps are its own, not its cells'
    averages; a continuous field's are zero. Cell data (``Constant``, values on
    cells) is read from each side's cell directly, on any topology, polyhedral
    included."""
    field = _field(data, field)
    out = _like(field.values, data.topology.faces().num_faces)
    if _on_cells(data, field):
        for_each(_cell_jump, data, "faces", materialize(field.values), out)
    else:
        for_each(_jump, data, "faces", traces(data, field), out)
    return Field(Values(data, "faces"), out, units.delta(field.unit))

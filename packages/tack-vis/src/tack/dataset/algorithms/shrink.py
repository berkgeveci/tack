"""``shrink``: every cell shrunk toward its centroid (the mean of its corners) by ``factor``, as Viskores' Shrink and VTK's vtkShrinkFilter. Viskores gives each cell its own copies of its points; here that is an ``L2`` geometry on the same topology, and point data becomes ``L2`` the same way (each cell's own copy of its corners' values). Everything else is kept, the topology's faces and edges with it."""

import tack
from tack.dataset.algorithms.discontinuous import discontinuous
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import H1, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['shrink']


@tack.kernel
def _shrink_corners(positions, offsets, factor, n_cells):
    for c in range(n_cells):
        first = offsets[c]
        end = offsets[c + 1]
        center = positions[first]
        for k in range(first + 1, end):
            center += positions[k]
        center = center / (end - first)
        for k in range(first, end):
            positions[k] = center + factor * (positions[k] - center)


@on_inputs_device
def shrink(data, factor=0.5, fields=None):
    """Every cell shrunk toward its centroid (the mean of its corners) by ``factor``,
    as Viskores' Shrink and VTK's vtkShrinkFilter. Viskores gives each cell its
    own copies of its points; here that is an ``L2`` geometry on the same
    topology, and point data becomes ``L2`` the same way (each cell's own copy
    of its corners' values). Everything else is kept, the topology's faces and
    edges with it."""

    if not 0 <= factor <= 1:
        raise ValueError(f"factor must lie in [0, 1], not {factor!r}")
    corners = discontinuous(data, data.geometry)
    if data.num_cells:
        _shrink_corners(corners.values, corners.space.offsets, float(factor), data.num_cells)
    from tack.dataset.carry import selected

    out = {}
    for name in selected(data, fields):
        f = data.fields[name]
        space = f.space
        if isinstance(space, Values) and space.on == "points":
            f, space = Field(H1(data), f.values), H1(data)
        out[name] = discontinuous(data, f) if isinstance(space, H1) else f
    return DataSet(data.topology, corners, fields=out, sets=dict(data.sets))

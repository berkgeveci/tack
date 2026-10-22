"""Ghost marks: which of a dataset's cells and points another piece owns.

A piece of a partitioned dataset holds copies of what its neighbours own, so
that work near its edge sees whole neighbourhoods. Every cell and point that
several pieces hold has one owner; every other copy is a ghost, marked with
VTK's ``vtkGhostType`` bits in ``data.ghosts`` (a ``Ghosts``, kept apart from
the fields as the geometry is). ``vtk_to_dataset`` and ``dataset_to_vtk`` map
the marks to and from VTK's arrays, ``split`` marks the points its pieces
share, and reductions (``statistics``, ``entropy``) count only what a piece
owns, so each value counts once across all pieces.
"""

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import materialize, size_of
from tack.runtime.dispatch import as_device, backend_of, get_backend, same_device

__all__ = ["Ghosts"]


def _marks(values, what):
    """``values`` as a one-dimensional u8 array, or ``None``."""
    if values is None:
        return None
    if isinstance(values, tack.Array):
        if values.dtype is not tack.u8 or len(values.shape) != 1:
            raise TypeError(f"ghost marks on {what} are a one-dimensional u8 array")
        return values
    host = np.asarray(values)
    if host.ndim != 1:
        raise TypeError(f"ghost marks on {what} are one-dimensional")
    return tack.array(host.astype(np.uint8), tack.u8)


class Ghosts:
    """Ghost marks of a dataset's cells and points: one u8 per cell (``cells``) and
    per point (``points``), either of which may be ``None``, holding VTK's
    ``vtkGhostType`` bits.

    A cell or point marked ``DUPLICATECELL`` or ``DUPLICATEPOINT`` is a copy of one
    another piece owns. ``HIDDENCELL`` and ``HIDDENPOINT`` mark data that is not
    valid there (an AMR cell a finer level covers, a blanked cell). The other
    bits are carried, not read."""

    DUPLICATECELL = 1
    HIGHCONNECTIVITYCELL = 2
    LOWCONNECTIVITYCELL = 4
    REFINEDCELL = 8
    EXTERIORCELL = 16
    HIDDENCELL = 32
    DUPLICATEPOINT = 1
    HIDDENPOINT = 2

    #: What a reduction leaves out: copies of what another piece owns, and
    #: what is not valid there.
    SKIPPED_CELLS = DUPLICATECELL | HIDDENCELL
    SKIPPED_POINTS = DUPLICATEPOINT | HIDDENPOINT

    def __init__(self, cells=None, points=None):
        self.cells = _marks(cells, "cells")
        self.points = _marks(points, "points")
        _ = self._device                    # refuses marks on two devices

    @property
    def _device(self):
        return same_device("Ghosts", (self.__dict__.get("cells"), self.__dict__.get("points")))

    @property
    def device(self):
        """The `tack.Device` the marks are on."""
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """These marks on ``device``."""
        target = as_device(backend_of(device))
        return Ghosts(None if self.cells is None else self.cells.to_device(target),
                      None if self.points is None else self.points.to_device(target))

    def _check(self, topology):
        for what, marks, n in (("cells", self.cells, topology.num_cells),
                               ("points", self.points, topology.num_points)):
            if marks is not None and marks.shape[0] != n:
                raise ValueError(f"{marks.shape[0]} ghost marks for {n} {what}")

    def __repr__(self):
        parts = [f"{what}={'marked' if marks is not None else None}"
                 for what, marks in (("cells", self.cells), ("points", self.points))]
        return f"Ghosts({', '.join(parts)})"


@tack.kernel
def _keep(marks, skipped, flags):
    for i in range(flags.shape[0]):
        flags[i] = 1 if (marks[i] & skipped) == 0 else 0


@tack.kernel
def _keep_blocks(marks, skipped, offsets, flags):
    # One flag per value of an L2 field, the values of cell c from offsets[c].
    for c in range(marks.shape[0]):
        keep = 1 if (marks[c] & skipped) == 0 else 0
        for k in range(offsets[c], offsets[c + 1]):
            flags[k] = keep


@tack.kernel
def _kept(flags, slots, ids):
    for i in range(flags.shape[0]):
        if flags[i] == 1:
            ids[slots[i]] = i


def owned_values(data, field):
    """The values of ``field`` (a ``Field`` of ``data``) that ``data`` owns: all of
    them when it has no marks for where they live, else those of cells and
    points not marked a duplicate or hidden, materialized. Fields on faces or
    edges, and order-2 fields, have no marks of their own yet; on a dataset
    with ghost marks they are refused rather than counted twice."""
    from tack.dataset.carry import _take
    from tack.dataset.spaces import H1, L2, Constant, Values

    values = materialize(field.values)
    ghosts = data.ghosts
    if ghosts is None or (ghosts.cells is None and ghosts.points is None):
        return values
    space = field.space
    n = size_of(values)
    on_points = (isinstance(space, H1) and space.order == 1) or (
        isinstance(space, Values) and space.on == "points")
    on_cells = isinstance(space, (Constant, L2)) or (
        isinstance(space, Values) and space.on == "cells")
    if on_points:
        marks, skipped = ghosts.points, Ghosts.SKIPPED_POINTS
    elif on_cells:
        marks, skipped = ghosts.cells, Ghosts.SKIPPED_CELLS
    else:
        raise NotImplementedError(f"which values of a field on {space!r} a piece owns is "
                                  "not decided yet; this dataset has ghost marks")
    if marks is None:
        return values
    flags = tack.empty((n,), tack.i32)
    if isinstance(space, L2):
        if marks.shape[0]:
            _keep_blocks(marks, skipped, space.offsets, flags)
    elif n:
        _keep(marks, skipped, flags)
    idt = data.id_dtype
    slots = tack.empty((n,), idt)
    count = exclusive_scan(flags, slots, n) if n else 0
    if count == n:
        return values
    ids = tack.empty((count,), idt)
    if count:
        _kept(flags, slots, ids)
    return _take(values, ids)

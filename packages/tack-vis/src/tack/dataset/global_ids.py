"""Global ids: each point's and cell's number across every piece of a partitioned dataset.

A piece numbers its points and cells from 0, so the same point has a number per
piece holding it. Global ids give each one number everywhere: ``data.global_ids``
(a ``GlobalIds``, kept apart from the fields as the ghost marks are), ``i64``
per point and per cell, either of which may be ``None``. ``vtk_to_dataset`` and
``dataset_to_vtk`` map them to and from VTK's ``GlobalIds`` attributes, and
``generate_global_ids`` makes them for pieces that have none.

Filters carry them where an output entity is an input entity (``carry``'s
``Same`` and ``Take``): kept points and kept or extracted cells keep their ids.
What a filter makes -- a contour's points, a clip's pieces of cells, a
surface's faces -- has no global id, as in VTK.
"""

import numpy as np

import tack
from tack.runtime.dispatch import as_device, backend_of, get_backend, same_device

__all__ = ["GlobalIds"]


class GlobalIds:
    """Global ids of a dataset's points (``points``) and cells (``cells``): one
    ``i64`` each, either of which may be ``None``. A point or cell several pieces
    hold has the same id in each; ids are unique across the pieces otherwise."""

    def __init__(self, cells=None, points=None):
        self.cells = _ids(cells, "cells")
        self.points = _ids(points, "points")
        _ = self._device                    # refuses ids on two devices

    @property
    def _device(self):
        return same_device("GlobalIds", (self.__dict__.get("cells"),
                                         self.__dict__.get("points")))

    @property
    def device(self):
        """The `tack.Device` the ids are on."""
        return as_device(self._device or get_backend())

    def to_device(self, device):
        """These ids on ``device``."""
        target = as_device(backend_of(device))
        return GlobalIds(None if self.cells is None else self.cells.to_device(target),
                         None if self.points is None else self.points.to_device(target))

    def _check(self, topology):
        for what, ids, n in (("cells", self.cells, topology.num_cells),
                             ("points", self.points, topology.num_points)):
            if ids is not None and ids.shape[0] != n:
                raise ValueError(f"{ids.shape[0]} global ids for {n} {what}")

    def __repr__(self):
        parts = [f"{what}={'numbered' if ids is not None else None}"
                 for what, ids in (("cells", self.cells), ("points", self.points))]
        return f"GlobalIds({', '.join(parts)})"


def _ids(values, what):
    """``values`` as a one-dimensional i64 array, or ``None``."""
    if values is None:
        return None
    if isinstance(values, tack.Array):
        if values.dtype is not tack.i64 or len(values.shape) != 1:
            raise TypeError(f"global ids of {what} are a one-dimensional i64 array")
        return values
    host = np.asarray(values)
    if host.ndim != 1 or (host.size and host.dtype.kind not in "iu"):
        raise TypeError(f"global ids of {what} are one-dimensional integers")
    return tack.array(host.astype(np.int64), tack.i64)


def carry_ids(source, out, points=None, cells=None):
    """``out``, made from ``source`` through ``carry``'s maps, with the global ids of
    the entities that are input entities: points and cells by ``Same`` or
    ``Take`` (not ``Pieces``, which share their source cell's)."""
    from tack.dataset.carry import Pieces, Same, Take, _take

    ids = source.global_ids
    if ids is None:
        return out

    def moved(values, how):
        if values is None or how is None or isinstance(how, Pieces):
            return None
        if isinstance(how, Same):
            return values
        if isinstance(how, Take):
            return _take(values, how.ids)
        return None

    point_ids, cell_ids = moved(ids.points, points), moved(ids.cells, cells)
    if point_ids is not None or cell_ids is not None:
        out.global_ids = GlobalIds(cells=cell_ids, points=point_ids)
    return out

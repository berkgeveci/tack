"""``probe``: ``data``'s fields at the points of ``where`` -- a dataset, or an ``(n, 3)`` array of points, which becomes vertex cells -- as Viskores' Probe and VTK's vtkProbeFilter: the result is ``where`` with each field as point data, a field with a basis (``H1``, ``L2``, ``Constant``, values on points) evaluated through it in the cell holding the point, values on cells taken from that cell, and ``valid`` (1, or 0 where no cell holds the point and the values are 0). ``locator`` reuses a ``CellLocator`` of ``data``."""

import numpy as np

import tack
from tack import arrays
from tack.dataset import ids
from tack.dataset.algorithms._common import _field
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.buckets import bucket_groups
from tack.dataset.by_id import field_by_id
from tack.dataset.dataset import DataSet, Field
from tack.dataset.locator import CellLocator
from tack.dataset.spaces import H1, Constant, Values
from tack.runtime.dispatch import on_inputs_device

__all__ = ['probe']


@tack.kernel
def _evaluate(cells, u, links, link_points, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        for n in range(links[e], links[e + 1]):
            q = link_points[n]
            out[q] = u.value(c, pcs[q])


@tack.kernel
def _evaluate_in_tets(cells, u, links, link_points, tets, pcs, out):
    for c in cells:
        e = cells.entity_id(c)
        for n in range(links[e], links[e + 1]):
            q = link_points[n]
            out[q] = u.tet_value(c, tets[q], pcs[q])


@tack.kernel
def _values_from_points(mesh, field, found, pcs, tets, out):
    # One thread per point, the field read by cell id (by_id).
    for q in range(found.shape[0]):
        e = found[q]
        if e >= 0:
            out[q] = field.value(mesh, e, mesh.cells.cell_type(e), pcs[q], tets[q])


@tack.kernel
def _found_keys(found, ncells, keys):
    for q in range(found.shape[0]):
        keys[q] = found[q] if found[q] >= 0 else ncells


@tack.kernel
def _valid(found, out):
    for q in range(found.shape[0]):
        out[q] = 1 if found[q] >= 0 else 0


@on_inputs_device
def probe(data, where, fields=None, locator=None):
    """``data``'s fields at the points of ``where`` -- a dataset, or an ``(n, 3)``
    array of points, which becomes vertex cells -- as Viskores' Probe and VTK's
    vtkProbeFilter: the result is ``where`` with each field as point data, a
    field with a basis (``H1``, ``L2``, ``Constant``, values on points)
    evaluated through it in the cell holding the point, values on cells taken
    from that cell, and ``valid`` (1, or 0 where no cell holds the point and the
    values are 0). ``locator`` reuses a ``CellLocator`` of ``data``."""
    from tack.dataset.carry import _point_values, selected

    if not isinstance(where, DataSet):
        where = _vertices(np.asarray(where, float).reshape(-1, 3), data)
    if not isinstance(where.geometry.space, H1):
        raise TypeError("probe needs the probed points: an H1 geometry")
    locator = locator or CellLocator(data)
    if locator.data is not data:
        raise ValueError("the locator was built for another dataset")
    points = arrays.materialize(_point_values(where.geometry.values, where.num_points))
    found, pcs, tets = locator.find(points)
    m = found.shape[0]
    # Fields read by cell id where find searched from the points; built only
    # then, since it needs the cell bins.
    search = locator.__dict__.get("_mesh_search")
    grouped = []

    def by_cell():
        # The points grouped by the cell found, for the fields evaluated from
        # the cells (through each group's view), made once if any is.
        if not grouped:
            keys = tack.empty((m,), found.dtype)
            if m:
                _found_keys(found, data.num_cells, keys)
            # The extra cell: not found, never evaluated, so not filled.
            grouped.append(bucket_groups(keys, data.num_cells + 1,
                                         ids.at_least(data.id_dtype, m), skip=data.num_cells))
        return grouped[0]

    out = {}
    for name in selected(data, fields):
        f = _field(data, name)
        space = f.space
        if isinstance(space, Values) and space.on in ("points", "cells"):
            f = Field((H1 if space.on == "points" else Constant)(data), f.values)
            space = f.space
        if not getattr(space, "interpolated", False) or space.on not in ("points", "cells"):
            continue
        width = arrays.width_of(f.values)
        dtype = arrays.dtype_of(f.values)
        values = (tack.empty((m,), tack.vec(width, dtype)) if width
                  else tack.empty((m,), dtype))
        by_id = field_by_id(data, f) if search is not None else None
        if m and by_id is not None:
            values.fill(0)
            # tets is read only in polyhedra; found stands in for it elsewhere.
            _values_from_points(search, by_id, found, pcs,
                                tets if tets is not None else found, values)
        elif m:
            values.fill(0)
            order, links = by_cell()
            for group in data.launch_groups("cells", [f]):
                if group.count and locator.polyhedral:
                    _evaluate_in_tets(data.domain_view("cells", group), f.view(group), links,
                                      order, tets, pcs, values)
                elif group.count:
                    _evaluate(data.domain_view("cells", group), f.view(group), links, order,
                              pcs, values)
        out[name] = Field(H1(where), values)
    valid = tack.empty((m,), tack.i32)
    if m:
        _valid(found, valid)
    out["valid"] = Field(Values(where, "points"), valid)
    kept = dict(where.fields)
    kept.update(out)
    from tack.dataset.ghosts import keep_marks

    return keep_marks(where, DataSet(where.topology, where.geometry, fields=kept,
                                     sets=dict(where.sets)))

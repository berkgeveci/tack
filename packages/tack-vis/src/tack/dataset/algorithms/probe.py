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
from tack.dataset.locator import cached_locator
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
    values are 0). ``locator`` reuses a ``CellLocator`` of ``data``.

    ``data`` may be a ``PartitionedDataSet``: each point goes to the piece one
    of whose own cells (not a ghost) holds it, the lowest such piece, which
    evaluates it there, and the values come back. ``where`` is then this
    rank's points -- a dataset, an array, or a ``PartitionedDataSet``, which
    gives one back, each piece probed -- and every rank's points are probed
    over every rank's pieces of ``data.world``. A point on a face two pieces
    share takes the lower piece's values, as one dataset takes its lower
    cell's."""
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        if locator is not None:
            raise ValueError("probe of a PartitionedDataSet uses each piece's own locator")
        return _probe_pieces(data, where, fields)
    from tack.dataset.ghosts import refuse_hidden, visible

    if locator is None:
        data = visible(data)
    else:
        refuse_hidden(data, "probe with a locator")
    from tack.dataset.carry import _point_values, selected

    if not isinstance(where, DataSet):
        where = _vertices(np.asarray(where, float).reshape(-1, 3), data)
    if not isinstance(where.geometry.space, H1):
        raise TypeError("probe needs the probed points: an H1 geometry")
    locator = locator or cached_locator(data)
    if not locator.serves(data):
        raise ValueError("the locator was built for another topology or geometry, or "
                         "Tack has written the positions since: make a new one")
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


def _held_box(piece):
    """A piece's box, grown by what its locator accepts outside a cell: a share
    (``locator._TOLERANCE``) of a cell's diagonal, so of the piece's at most."""
    from tack.dataset.locator import _TOLERANCE

    x = piece.positions().astype(np.float64)
    lower, upper = x.min(axis=0), x.max(axis=0)
    grow = _TOLERANCE * float(np.linalg.norm(upper - lower))
    return lower - grow, upper + grow


def _probe_pieces(data, where, fields):
    """``probe`` with a partitioned source (``probe``'s docstring)."""
    from tack import communication as tc
    from tack.communication._kernels import gather
    from tack.dataset.ghosts import Ghosts, keep_marks, visible
    from tack.dataset.partitioned import PartitionedDataSet

    pieces = []
    for piece in data:                       # each piece's work on its own device
        with piece.device:
            pieces.append(visible(piece))
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    if not len(blocks.local):
        raise ValueError("probe of a PartitionedDataSet needs pieces on this rank to "
                         "start from")

    # This rank's targets, their points end to end, from its first block.
    many = isinstance(where, PartitionedDataSet)
    targets = list(where) if many else [where]
    for i, target in enumerate(targets):
        if not isinstance(target, DataSet):
            targets[i] = _vertices(np.asarray(target, float).reshape(-1, 3), pieces[0])
        elif not isinstance(target.geometry.space, H1):
            raise TypeError("probe needs the probed points: an H1 geometry")
    points = [np.asarray(t.positions(), np.float64).reshape(-1, 3) for t in targets]
    counts = [len(p) for p in points]
    first = blocks.local.start
    positions = {g: (np.concatenate(points) if g == first and points else np.zeros((0, 3)))
                 for g in blocks.local}
    boxes = {}
    locators = {}
    for g, piece in zip(blocks.local, pieces):
        if piece.num_points:
            boxes[g] = _held_box(piece)
            with piece.device:
                locators[g] = cached_locator(piece)
        else:
            boxes[g] = None

    def holds(g, x):
        piece = pieces[g - blocks.local.start]
        with piece.device:
            cells, held = locators[g]._find_holding(x.astype(piece.dtype.numpy_dtype))
        cells = arrays.to_host(cells).astype(np.int64)
        owned = cells >= 0
        marks = None if piece.ghosts is None else piece.ghosts.cells
        if marks is not None:
            owned &= (marks.numpy()[np.maximum(cells, 0)] & Ghosts.SKIPPED_CELLS) == 0
        # Holding the point ranks ahead of coming within the tolerance, as one
        # dataset's locator ranks its cells.
        return np.where(owned, 1 + held.numpy(), 0)

    routing = tc.Routing(blocks, positions, boxes, holds)
    arrived = routing.send()

    # What each piece gives back, and its element types, from a probe of no
    # points (so a rank whose points went nowhere still knows them).
    with pieces[0].device:
        empty = probe(pieces[0], np.zeros((0, 3)), fields)
    names = [n for n in empty.fields if n not in ("valid",)] + ["valid"]
    like = {n: empty.fields[n].values.dtype for n in names}
    answers = {}
    for g, piece in zip(blocks.local, pieces):
        x, _ = arrived[g]
        if len(x):
            with piece.device:
                probed = probe(piece, x, fields)
            answers[g] = {n: probed.fields[n].values for n in names}
    replied = routing.reply(answers, like)[first]

    out = []
    start = 0
    for target, n in zip(targets, counts):
        kept = dict(target.fields)
        rows = tack.arange(start, start + n, dtype=tack.i64, device=blocks.device(first))
        for name in names:
            values = gather(replied[name], rows).to_device(target.device)
            space = Values(target, "points") if name == "valid" else H1(target)
            kept[name] = Field(space, values)
        start += n
        out.append(keep_marks(target, DataSet(target.topology, target.geometry, fields=kept,
                                              sets=dict(target.sets))))
    return PartitionedDataSet(out, world=data.world) if many else out[0]

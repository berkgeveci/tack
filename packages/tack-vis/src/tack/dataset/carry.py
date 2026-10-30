"""How a filter's output takes its input's fields.

A filter that makes a new dataset says how each kind of output entity comes
from the input -- the same entity (``Same``), entity ``ids[i]`` of the input
(``Take``), a piece of input cell ``ids[i]`` (``Pieces``: a contour's
triangles, a surface's faces), or a point between two input points
(``Interpolate``) -- and ``carry`` applies one set of rules to every field:

- **point data** (``H1``, values on points) goes through the point map. A
  field is interpolated only if its values are floating point; an integer
  field meets an ``Interpolate`` map and is left behind.
- **an H1 field above order 1** (or a curved geometry) keeps its order where
  the cell map keeps or takes whole cells: each output cell's nodes are its
  source cell's (``_carry_h1``). Elsewhere (a contour, a surface's pieces) it
  brings its corner values, as the filters read it.
- **cell data** (``Constant``, values on cells) goes through the cell map
  and keeps its kind.
- **L2 (DG) data** is copied cell by cell, keeping each cell's order, when
  the cell map keeps or takes whole cells (not pieces of them) and the
  output has reference cells to lay it out on.
- **values on faces** go through the face map -- negated where the map
  turns a face around, if they are oriented -- or become values on the
  output's cells when those cells are input faces (``faces_to_cells``).
- **values on edges** go through the edge map, which only a filter whose
  edges are the input's gives (``as_polyhedra``), or become values on the
  output's cells when those cells are input edges (``edges_to_cells``).
- traces, and fields on any entity without a map, are left behind: the
  output's own edges and faces are derived afresh, so the old numbering
  means nothing there.

The geometry follows the same rules unless the filter computed positions
itself. ``fields`` selects which fields come along: ``None`` (all), a list
of names, or an empty list (none); naming a field the input does not have
raises ``KeyError``.
"""

import numpy as np

import tack
from tack import arrays
from tack.arrays import materialize, size_of, width_of
from tack.dataset import ids
from tack.dataset.buckets import bucket_order
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import H1, L2, Constant, Values
from tack.lang.array import Array as _DeviceArray
from tack.runtime.dispatch import on_inputs_device

__all__ = ["Interpolate", "Pieces", "Same", "Take", "carry"]


def _id_field(values):
    """Ids as a field: a field as it is, a list or an array copied to one."""
    if values is None or isinstance(values, _DeviceArray):
        return values
    host = np.asarray(values)
    return ids.as_ids(host, ids.at_least(tack.i32, int(host.max()) + 1 if host.size else 0))


class Same:
    """Output entity ``i`` is input entity ``i``."""


class Take:
    """Output entity ``i`` is input entity ``ids[i]``. ``turned``, for faces: an
    i32 array, 1 where the output face is the input face turned around, so
    its oriented values change sign."""

    def __init__(self, ids, turned=None):
        self.ids = _id_field(ids)
        self.turned = turned if turned is None or isinstance(turned, _DeviceArray) \
            else tack.array(np.asarray(turned, np.int32))


class Pieces(Take):
    """Output cell ``i`` lies inside input cell ``ids[i]``: it takes that cell's
    values, but not its L2 data, which belongs to the whole cell's shape."""

    def __init__(self, ids):
        super().__init__(ids)


class Interpolate:
    """Output point ``i`` lies between input points ``ends[i]`` (a 2-vector of ids)
    at ``weights[i]`` from the first. ``averages``, ``(offsets, members)``: further
    output points after those, each the average of earlier output points
    ``members[offsets[k]:offsets[k + 1]]`` (clip's centroid points), and inside
    input cell ``average_cells[k]``, whose ghost mark it takes. ``exact``, when
    given, puts a field's values at the first points where they truly are rather
    than between their ends (``_crossings``: ``exact(field, values)``)."""

    def __init__(self, ends, weights, averages=None, average_cells=None, exact=None):
        self.ends = ends
        self.weights = weights
        self.averages = averages
        self.average_cells = average_cells
        self.exact = exact


# ── Values ──────────────────────────────────────────────────────────

def _like(values, n):
    """``n`` values of ``values``' element type, on its device."""
    width = width_of(values)
    dtype = tack.vec(width, arrays.dtype_of(values)) if width else values.dtype
    return tack.empty((n,), dtype, device=values.device)


@tack.kernel
def _take_rows(values, ids, out):
    for i in range(ids.shape[0]):
        out[i] = values[ids[i]]


def _take(values, ids):
    out = _like(values, ids.shape[0])
    if ids.shape[0]:
        _take_rows(materialize(values), ids, out)
    return out


def _point_values(values, n):
    """An H1 field's values at the points: all of an order-1 field's, the first ``n``
    of an order-2 field's (which go on to its edges, faces and cells)."""
    if size_of(values) == n:
        return values
    return _take(values, tack.arange(n, dtype=ids.at_least(tack.i32, n)))


@tack.kernel
def _interpolate_rows(values, ends, weights, out):
    for i in range(weights.shape[0]):
        ab = ends[i]
        va = values[ab[0]]
        vb = values[ab[1]]
        w = weights[i]
        # At an end, that end's value: a blend there is NaN beside infinity.
        out[i] = va if w == 0.0 else (vb if w == 1.0 else va + w * (vb - va))


@tack.kernel
def _average_rows(out, offsets, members, first):
    for k in range(offsets.shape[0] - 1):
        begin = offsets[k]
        total = out[members[begin]]
        for i in range(begin + 1, offsets[k + 1]):
            total += out[members[i]]
        out[first + k] = total / (offsets[k + 1] - begin)


@tack.kernel
def _negate_turned(values, turned):
    for f in range(turned.shape[0]):
        if turned[f] == 1:
            values[f] = -values[f]


@tack.kernel
def _copy_blocks(values, src_offsets, sources, dst_offsets, out, count):
    for i in range(count):
        a = src_offsets[sources[i]]
        b = dst_offsets[i]
        for k in range(dst_offsets[i + 1] - b):
            out[b + k] = values[a + k]


@tack.kernel
def _copy_laid_out(values, sources, src_offsets, src_cells, src_count, dst_offsets,
                   dst_cells, per_cell, out, count):
    # Each kept cell's values, each side in its layout: cell-fastest (value k of
    # cell c at k * cells + c) or blocks (from offsets[c]).
    for i in range(count):
        s = sources[i]
        for k in range(per_cell):
            a = k * src_count + s if src_cells == 1 else src_offsets[s] + k
            b = k * count + i if dst_cells == 1 else dst_offsets[i] + k
            out[b] = values[a]


def _apply(values, how, oriented=False, field=None):
    """The values through one map, or ``None`` if they cannot go through it. An
    ``Interpolate`` with ``exact`` places ``field``'s values exactly."""
    if isinstance(how, Same):
        return values
    if isinstance(how, Take):
        out = _take(values, how.ids)
        if oriented and how.turned is not None and how.ids.shape[0]:
            _negate_turned(out, how.turned)
        return out
    if isinstance(how, Interpolate):
        if arrays.dtype_of(values) not in (tack.f32, tack.f64):
            return None
        n = how.weights.shape[0]
        extra = how.averages[0].shape[0] - 1 if how.averages is not None else 0
        out = _like(values, n + extra)
        if n:
            _interpolate_rows(materialize(values), how.ends, how.weights, out)
            if how.exact is not None and field is not None:
                how.exact(field, out)
        if extra:
            _average_rows(out, how.averages[0], how.averages[1], n)
        return out
    raise TypeError(f"not a map: {how!r}")


# ── Fields ──────────────────────────────────────────────────────────

def _on(space, *kinds):
    return isinstance(space, Values) and space.on in kinds


def _carry_l2(field, out, cells):
    """An L2 field cell by cell onto the kept or taken cells, each keeping its order."""
    space = field.space
    if not getattr(out, "reference_cells", True):
        return None
    if isinstance(cells, Same):
        topology = field.space.topology
        sources = tack.arange(topology.num_cells, dtype=topology.id_dtype)
    elif isinstance(cells, Take) and not isinstance(cells, Pieces):
        sources = cells.ids
    else:
        return None
    count = sources.shape[0]
    if space.varies:
        orders = np.asarray(arrays.to_host(space.cell_keys()))
        out_space = L2(out, order=orders[sources.numpy()] if count else np.zeros(0, int))
    else:
        # The same basis: order, nodes and the layout asked for ("auto" resolves on
        # the output's device).
        out_space = L2(out, order=space.order, basis=space.basis or "gauss_lobatto",
                       layout=space.params.get("layout", "auto"))
    values = _like(field.values, out_space.size)
    if count and space.layout == out_space.layout == "blocks":
        _copy_blocks(materialize(field.values), space.offsets, sources, out_space.offsets,
                     values, count)
    elif count:
        per_cell = space.size // max(space.topology.num_cells, 1)
        _copy_laid_out(materialize(field.values), sources, space.offsets,
                       int(space.layout == "cells"), space.topology.num_cells,
                       out_space.offsets, int(out_space.layout == "cells"), per_cell, values,
                       count)
    return Field(out_space, values, field.unit)


@tack.kernel
def _node_indices(cells, u, per_cell, start, entry, out):
    # Each cell's nodes' places in its space, entries a cell's nodes together.
    for c in cells:
        first = start + cells.index(c) * per_cell
        entry[cells.entity_id(c)] = first
        for n in range(per_cell):
            out[first + n] = u.dof_index(c, n)


@tack.kernel
def _first_of_each(order, starts, writers, n):
    # The first (cell, node) entry of each node, in entry order, writes it.
    for d in range(n):
        if starts[d + 1] > starts[d]:
            writers[order[starts[d]]] = 1


@tack.kernel
def _copy_nodes(cells, sources, per_cell, out_entry, out_index, writers, in_entry, in_index,
                values, out):
    # An output cell's node n is its source cell's node n: the same shape, the
    # same corners, each space placing it through its own edges and faces.
    for c in cells:
        o = cells.entity_id(c)
        a = out_entry[o]
        b = in_entry[sources[o]]
        for n in range(per_cell):
            if writers[a + n] == 1:
                out[out_index[a + n]] = values[in_index[b + n]]


def _nodes_per_cell(space, shape):
    """The nodes an ``H1`` space of order 2 or more gives a cell of ``shape``."""
    from math import comb

    from tack.dataset import bases

    if space.order == 2:
        return shape.NUM_QUADRATIC
    if bases.simplex_shape(shape):
        return comb(space.order + shape.DIMENSION, shape.DIMENSION)
    if bases.wedge_shape(shape):
        return comb(space.order + 2, 2) * (space.order + 1)
    return (space.order + 1) ** shape.DIMENSION


def _node_entries(space, values):
    """Every cell's nodes' places in ``space`` (a field of it on ``values``' type):
    ``(index, entry, groups)``, ``entry[cell]`` where its nodes start in
    ``index``, ``groups`` each group with its start and nodes a cell."""
    topology = space.topology
    groups, total = [], 0
    for group in topology.groups():
        if group.count:
            per_cell = _nodes_per_cell(space, group.shape)
            groups.append((group, total, per_cell))
            total += group.count * per_cell
    index = tack.empty((max(total, 1),), ids.at_least(topology.id_dtype, space.size))
    entry = tack.empty((max(topology.num_cells, 1),), ids.at_least(topology.id_dtype, total))
    view_of = Field(space, values)
    for group, start, per_cell in groups:
        _node_indices(group.view(), view_of.view(group), per_cell, start, entry, index)
    return index, entry, groups, total


def _carry_h1(field, out, points, cells):
    """An ``H1`` field of order 2 or more onto the kept or taken whole cells, keeping
    its order and nodes; ``None`` where the cell map is not one of whole cells."""
    space = field.space
    if not getattr(out, "reference_cells", True) or isinstance(points, Interpolate):
        return None
    if isinstance(cells, Same):
        topology = space.topology
        sources = tack.arange(topology.num_cells, dtype=topology.id_dtype)
    elif isinstance(cells, Take) and not isinstance(cells, Pieces):
        sources = cells.ids
    else:
        return None
    out_space = H1(out, **space.params)
    values = _like(field.values, out_space.size)
    if points is not None and out.num_points:
        # Points no cell uses still take their values.
        corners = _apply(_point_values(field.values, space.topology.num_points), points)
        if corners is not None:
            _copy_prefix(corners, values)
    if sources.shape[0]:
        held = materialize(field.values)
        in_index, in_entry, _, _ = _node_entries(space, held)
        out_index, out_entry, groups, total = _node_entries(out_space, values)
        writers = tack.zeros((max(total, 1),), tack.i32)
        if total:
            order, starts = bucket_order(out_index, out_space.size,
                                         ids.at_least(out.id_dtype, total))
            _first_of_each(order, starts, writers, out_space.size)
        for group, _, per_cell in groups:
            _copy_nodes(group.view(), sources, per_cell, out_entry, out_index, writers,
                        in_entry, in_index, held, values)
    return Field(out_space, values, field.unit)


@tack.kernel
def _copy_rows(values, out):
    for i in range(values.shape[0]):
        out[i] = values[i]


def _copy_prefix(values, out):
    """``values`` into the first of ``out``'s rows."""
    if values.shape[0]:
        _copy_rows(materialize(values), out)


def _carry_field(field, data, out, points, cells, faces, faces_to_cells, edges=None,
                 edges_to_cells=None):
    """``field`` on ``out``, or ``None`` when no map applies to it."""
    space = field.space
    if isinstance(space, H1) and space.order > 1:
        kept = _carry_h1(field, out, points, cells)
        if kept is not None:
            return kept
    if isinstance(space, H1) or _on(space, "points"):
        if points is None:
            return None
        values = field.values
        if isinstance(space, H1):
            values = _point_values(values, data.num_points)
        values = _apply(values, points, field=field)
        if values is None:
            return None
        return Field(H1(out) if isinstance(space, H1) else Values(out, "points"), values,
                     field.unit)
    if isinstance(space, Constant) or _on(space, "cells"):
        if cells is None:
            return None
        return Field(Constant(out) if isinstance(space, Constant) else Values(out, "cells"),
                     _apply(field.values, cells), field.unit)
    if isinstance(space, L2):
        return None if cells is None else _carry_l2(field, out, cells)
    if _on(space, "faces"):
        if faces_to_cells is not None:
            return Field(Values(out, "cells"), _apply(field.values, faces_to_cells),
                         field.unit)
        if faces is None:
            return None
        return Field(Values(out, "faces", oriented=space.oriented),
                     _apply(field.values, faces, oriented=space.oriented), field.unit)
    if _on(space, "edges"):
        if edges_to_cells is not None:
            return Field(Values(out, "cells"), _apply(field.values, edges_to_cells),
                         field.unit)
        if edges is not None:
            return Field(Values(out, "edges"), _apply(field.values, edges), field.unit)
    return None


@tack.kernel
def _member_of_source(flags, sources, out):
    for i in range(sources.shape[0]):
        out[i] = flags[sources[i]]


def _carry_set(member, out, points, cells, faces, faces_to_cells, edges, edges_to_cells):
    """``member`` on ``out``: the output entities that come from its members, or
    ``None`` when no map takes its kind there (or a map makes new entities)."""
    from tack.dataset.sets import Set

    kind = member.on
    target = kind
    mapping = {"points": points, "cells": cells, "faces": faces, "edges": edges}[kind]
    if kind == "faces" and faces_to_cells is not None:
        mapping, target = faces_to_cells, "cells"
    if kind == "edges" and edges_to_cells is not None:
        mapping, target = edges_to_cells, "cells"
    if isinstance(mapping, Same):
        return Set._from_flags(out, target, member.flags())
    if not isinstance(mapping, Take):
        return None
    flags = tack.empty((mapping.ids.shape[0],), tack.i32)
    if mapping.ids.shape[0]:
        _member_of_source(member.flags(), mapping.ids, flags)
    return Set._from_flags(out, target, flags)


def selected(data, fields):
    """The names of ``data``'s fields that ``fields`` selects."""
    names = list(data.fields)
    if fields is None:
        return names
    fields = [fields] if isinstance(fields, str) else list(fields)
    missing = [name for name in fields if name not in names]
    if missing:
        raise KeyError(f"no field {', '.join(map(repr, missing))}: the dataset has "
                       f"{', '.join(map(repr, names)) or 'none'}")
    return [name for name in names if name in fields]


@on_inputs_device
def carry(data, out, *, points=None, cells=None, faces=None, faces_to_cells=None,
          edges=None, edges_to_cells=None, geometry=None, fields=None, sets=None):
    """A dataset on topology ``out`` with ``data``'s fields, by the maps given.

    ``points``, ``cells``, ``faces`` and ``edges`` say how ``out``'s come from
    ``data``'s (``Same``, ``Take``, ``Pieces`` for cells, ``Interpolate`` for
    points); ``faces_to_cells`` (``edges_to_cells``) that ``out``'s cells are
    ``data``'s faces (edges). A
    kind of entity with no map takes no fields. ``geometry`` is the output's
    geometry field when the filter computed it, otherwise the input's is
    carried like any point (or L2) field.

    Ghost marks go by the same maps (``ghosts.carry_marks``): what the output makes
    takes the owner of what it came from. Global ids go with the entities that are
    input entities (``global_ids.carry_ids``).

    Sets go by the same maps: an output entity is in a set when the input entity
    it comes from is, and a face or edge set becomes a cell set where faces or
    edges become cells (``faces_to_cells``, ``edges_to_cells``). A set whose kind has no map, or one that makes new
    entities (``Interpolate``), is left behind; ``sets``, if given, replaces
    them.
    """
    crossings = getattr(data, "_crossings", None)
    if crossings is not None and isinstance(points, Interpolate) and points.exact is None:
        # A refined dataset's cut: its crossings at the field's roots, the positions
        # the filter computed between lattice points too (and the averages of them).
        points = crossings.located(points)
        if geometry is not None and points.weights.shape[0]:
            values = geometry.values
            points.exact(data.geometry, values)
            if points.averages is not None and points.averages[0].shape[0] > 1:
                _average_rows(values, points.averages[0], points.averages[1],
                              points.weights.shape[0])
    if geometry is None:
        geometry = _carry_field(data.geometry, data, out, points, cells, faces,
                                faces_to_cells)
        if geometry is None:
            raise ValueError("the geometry cannot be carried by these maps; pass it")
    carried = {}
    for name in selected(data, fields):
        field = _carry_field(data.fields[name], data, out, points, cells, faces,
                             faces_to_cells, edges, edges_to_cells)
        if field is not None:
            carried[name] = field
    if sets is None:
        sets = {}
        for name, member in data.sets.items():
            moved = _carry_set(member, out, points, cells, faces, faces_to_cells, edges,
                               edges_to_cells)
            if moved is not None:
                sets[name] = moved
    from tack.dataset.ghosts import carry_marks
    from tack.dataset.global_ids import carry_ids

    # A geometry the filter computed is in the input's spatial unit.
    made = DataSet(out, geometry, fields=carried, sets=sets, **data._frame())
    carry_ids(data, made, points=points, cells=None if faces_to_cells is not None or
              edges_to_cells is not None else cells)
    return carry_marks(data, made, points=points, cells=cells, faces_to_cells=faces_to_cells,
                       edges_to_cells=edges_to_cells)

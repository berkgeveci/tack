"""How a filter's output takes its input's fields.

A filter that makes a new dataset says how each kind of output entity comes
from the input -- the same entity (``Same``), entity ``ids[i]`` of the input
(``Take``), a piece of input cell ``ids[i]`` (``Pieces``: a contour's
triangles, a surface's faces), or a point between two input points
(``Interpolate``) -- and ``carry`` applies one set of rules to every field:

- **point data** (``H1``, values on points) goes through the point map. An
  order-2 field brings its corner values, as the filters read it. A field
  is interpolated only if its values are floating point; an integer field
  meets an ``Interpolate`` map and is left behind.
- **cell data** (``Constant``, values on cells) goes through the cell map
  and keeps its kind.
- **L2 (DG) data** is copied cell by cell, keeping each cell's order, when
  the cell map keeps or takes whole cells (not pieces of them) and the
  output has reference cells to lay it out on.
- **values on faces** go through the face map -- negated where the map
  turns a face around, if they are oriented -- or become values on the
  output's cells when those cells are input faces (``faces_to_cells``).
- **values on edges** go through the edge map, which only a filter whose
  edges are the input's gives (``as_polyhedra``).
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
from tack.data import arrays
from tack.data.arrays import materialize, size_of, width_of
from tack.data.dataset import DataSet, Field
from tack.data.spaces import H1, L2, Constant, Values

__all__ = ["Interpolate", "Pieces", "Same", "Take", "carry"]


class Same:
    """Output entity ``i`` is input entity ``i``."""


class Take:
    """Output entity ``i`` is input entity ``ids[i]``. ``turned``, for faces: an
    i32 field, 1 where the output face is the input face turned around, so
    its oriented values change sign."""

    def __init__(self, ids, turned=None):
        self.ids = ids
        self.turned = turned


class Pieces(Take):
    """Output cell ``i`` lies inside input cell ``ids[i]``: it takes that cell's
    values, but not its L2 data, which belongs to the whole cell's shape."""

    def __init__(self, ids):
        super().__init__(ids)


class Interpolate:
    """Output point ``i`` lies between input points ``ends[i]`` (a 2-vector of ids)
    at ``weights[i]`` from the first."""

    def __init__(self, ends, weights):
        self.ends = ends
        self.weights = weights


# ── Values ──────────────────────────────────────────────────────────

def _like(values, n):
    width = width_of(values)
    if width:
        return tack.Vector.field(width, values.dtype, shape=(n,))
    return tack.field(values.dtype, shape=(n,))


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
    return _take(values, tack.arange(n, tack.i32))


@tack.kernel
def _interpolate_rows(values, ends, weights, out):
    for i in range(out.shape[0]):
        ab = ends[i]
        va = values[ab[0]]
        out[i] = va + weights[i] * (values[ab[1]] - va)


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


def _apply(values, how, oriented=False):
    """The values through one map, or ``None`` if they cannot go through it."""
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
        out = _like(values, n)
        if n:
            _interpolate_rows(materialize(values), how.ends, how.weights, out)
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
        sources = tack.arange(field.space.topology.num_cells, tack.i32)
    elif isinstance(cells, Take) and not isinstance(cells, Pieces):
        sources = cells.ids
    else:
        return None
    count = sources.shape[0]
    if space.varies:
        orders = np.asarray(arrays.to_host(space.cell_keys()))
        out_space = L2(out, order=orders[sources.to_numpy()] if count else np.zeros(0, int))
    else:
        out_space = L2(out, order=space.order)
    values = _like(field.values, out_space.size)
    if count:
        _copy_blocks(materialize(field.values), space.offsets, sources, out_space.offsets,
                     values, count)
    return Field(out_space, values)


def _carry_field(field, data, out, points, cells, faces, faces_to_cells, edges=None):
    """``field`` on ``out``, or ``None`` when no map applies to it."""
    space = field.space
    if isinstance(space, H1) or _on(space, "points"):
        if points is None:
            return None
        values = field.values
        if isinstance(space, H1):
            values = _point_values(values, data.num_points)
        values = _apply(values, points)
        if values is None:
            return None
        return Field(H1(out) if isinstance(space, H1) else Values(out, "points"), values)
    if isinstance(space, Constant) or _on(space, "cells"):
        if cells is None:
            return None
        return Field(Constant(out) if isinstance(space, Constant) else Values(out, "cells"),
                     _apply(field.values, cells))
    if isinstance(space, L2):
        return None if cells is None else _carry_l2(field, out, cells)
    if _on(space, "faces"):
        if faces_to_cells is not None:
            return Field(Values(out, "cells"), _apply(field.values, faces_to_cells))
        if faces is None:
            return None
        return Field(Values(out, "faces", oriented=space.oriented),
                     _apply(field.values, faces, oriented=space.oriented))
    if _on(space, "edges") and edges is not None:
        return Field(Values(out, "edges"), _apply(field.values, edges))
    return None


def selected(data, fields):
    """The names of ``data``'s fields (the geometry aside) that ``fields`` selects."""
    names = [name for name in data.fields if name != "shape"]
    if fields is None:
        return names
    fields = [fields] if isinstance(fields, str) else list(fields)
    missing = [name for name in fields if name not in names]
    if missing:
        raise KeyError(f"no field {', '.join(map(repr, missing))}: the dataset has "
                       f"{', '.join(map(repr, names)) or 'none'}")
    return [name for name in names if name in fields]


def carry(data, out, *, points=None, cells=None, faces=None, faces_to_cells=None,
          edges=None, geometry=None, fields=None, sets=None):
    """A dataset on topology ``out`` with ``data``'s fields, by the maps given.

    ``points``, ``cells``, ``faces`` and ``edges`` say how ``out``'s come from
    ``data``'s (``Same``, ``Take``, ``Pieces`` for cells, ``Interpolate`` for
    points); ``faces_to_cells`` that ``out``'s cells are ``data``'s faces. A
    kind of entity with no map takes no fields. ``geometry`` is the output's
    geometry field when the filter computed it, otherwise the input's is
    carried like any point (or L2) field. ``sets`` are the output's sets.
    """
    if geometry is None:
        geometry = _carry_field(data.geometry, data, out, points, cells, faces,
                                faces_to_cells)
        if geometry is None:
            raise ValueError("the geometry cannot be carried by these maps; pass it")
    carried = {}
    for name in selected(data, fields):
        field = _carry_field(data.fields[name], data, out, points, cells, faces,
                             faces_to_cells, edges)
        if field is not None:
            carried[name] = field
    return DataSet(out, geometry, fields=carried, sets=sets)

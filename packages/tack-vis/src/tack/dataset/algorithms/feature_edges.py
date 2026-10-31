"""``feature_edges``: a surface's boundary, feature, non-manifold and manifold edges as
line cells on the same points, as VTK's ``vtkFeatureEdges`` finds them."""

import enum
import math

import numpy as np

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids, shapes
from tack.dataset.algorithms._common import _table, _vectors
from tack.dataset.buckets import key_order, run_offsets
from tack.dataset.carry import Same, _point_values, carry
from tack.dataset.dataset import Field
from tack.dataset.spaces import H1, Values
from tack.dataset.topology import UnstructuredTopology, uniform_cells
from tack.runtime.dispatch import on_inputs_device

__all__ = ['EdgeKind', 'feature_edges']


class EdgeKind(enum.IntEnum):
    """The values of ``feature_edges``' ``kind`` cell field."""

    BOUNDARY = 1        # on one cell
    FEATURE = 2         # between two cells meeting at the feature angle or more
    NON_MANIFOLD = 3    # on three or more cells
    MANIFOLD = 4        # between two cells, not a feature edge


# The corner order going round each surface shape (a pixel's corners are x
# fastest, not round it).
_ROUND = {shapes.Triangle: (0, 1, 2), shapes.Quad: (0, 1, 2, 3), shapes.Pixel: (0, 1, 3, 2)}


@tack.kernel
def _loop_sizes(cells, size, sizes):
    for c in cells:
        sizes[cells.entity_id(c)] = size


@tack.kernel
def _shape_loops(cells, order, offsets, loops):
    for c in cells:
        e = cells.entity_id(c)
        for j in range(order.shape[0]):
            loops[offsets[e] + j] = cells.point_id(c, order[j])


@tack.kernel
def _sides(loops, offsets, positions, n_cells, keys, owners, normals):
    # Each cell's sides as (low, high) point pairs, and its Newell normal.
    for c in range(n_cells):
        start = offsets[c]
        size = offsets[c + 1] - start
        origin = positions[loops[start]] if size > 0 else tack.vec3(0.0, 0.0, 0.0)
        n = tack.vec3(0.0, 0.0, 0.0)
        for j in range(size):
            a = loops[start + j]
            b = loops[start + (j + 1 if j + 1 < size else 0)]
            keys[start + j] = [min(a, b), max(a, b)]
            owners[start + j] = c
            n += (positions[a] - origin).cross(positions[b] - origin)
        length = n.norm()
        normals[c] = n / length if length > 0.0 else n


@tack.kernel
def _classify(order, offsets, keys, owners, normals, cosine, wanted, kinds, flags, rows, count):
    # An edge's cells: one a boundary, three or more non-manifold, two a feature
    # where their normals are further apart than the feature angle.
    for r in range(count):
        first = offsets[r]
        n = offsets[r + 1] - first
        key = keys[order[first]]
        kind = 0
        if key[0] != key[1]:                       # a repeated point is no edge
            if n == 1:
                kind = 1
            elif n >= 3:
                kind = 3
            else:
                # A feature edge only where feature edges are asked for, as VTK's;
                # otherwise two cells' edge is manifold, whatever its angle.
                d = normals[owners[order[first]]].dot(normals[owners[order[first + 1]]])
                kind = 2 if wanted[2] == 1 and d <= cosine else 4
        kinds[r] = kind
        rows[r] = key
        flags[r] = 1 if kind > 0 and wanted[kind] == 1 else 0


@tack.kernel
def _chosen(flags, slots, kinds, rows, connectivity, kind_out, count):
    for r in range(count):
        if flags[r] == 1:
            s = slots[r]
            connectivity[2 * s] = rows[r][0]
            connectivity[2 * s + 1] = rows[r][1]
            kind_out[s] = kinds[r]


@on_inputs_device
def feature_edges(data, angle=30.0, boundary=True, feature=True, non_manifold=True,
                  manifold=False, fields=None):
    """A surface's edges of the kinds asked for, as line cells on the same points,
    each from its lower point id to its higher, in order of those ids, with
    ``kind`` (``EdgeKind``) on the cells -- as VTK's ``vtkFeatureEdges``:

    - ``boundary``: edges of one cell;
    - ``feature``: edges of two cells whose normals are at least ``angle``
      degrees apart (two cells wound opposite ways meet at almost 180);
    - ``non_manifold``: edges of three or more cells;
    - ``manifold``: the other edges of two cells -- all of them when feature
      edges are not asked for, as in VTK (off by default).

    The surface is the 2D cells -- triangles, quads, pixels, polygons; a
    polygon's normal is Newell's. Other cells are left out; a volume mesh's
    surface comes from ``extract_surface`` first. Point fields stay on the same
    points (``fields`` chooses which, as ``extract_edges``); cell fields are
    left behind, an edge belonging to every cell around it. Hidden cells are
    left out. The geometry must be ``H1``."""
    from tack.dataset.ghosts import visible
    from tack.dataset.polyhedra import PolygonalTopology

    data = visible(data)
    if not isinstance(data.geometry.space, H1):
        raise TypeError("feature_edges keeps the points, so needs an H1 geometry")
    topology = data.topology
    if topology.dimension == 3:
        raise ValueError("feature_edges takes a surface; take a volume mesh's with "
                         "extract_surface first")
    if not 0.0 <= angle <= 180.0:
        raise ValueError(f"the feature angle is in degrees, 0 to 180, not {angle!r}")
    idt = data.id_dtype
    n_cells = data.num_cells
    # Each 2D cell's loop of point ids.
    if isinstance(topology, PolygonalTopology):
        offsets, loops = (ids.as_ids(a, idt) for a in topology.loops())
    else:
        sizes = tack.zeros((n_cells,), tack.i64)
        groups = [g for g in data.launch_groups("cells", []) if g.count]
        for group in groups:
            if group.shape in _ROUND:
                _loop_sizes(data.domain_view("cells", group), len(_ROUND[group.shape]), sizes)
        offsets = tack.empty((n_cells + 1,), tack.i64)
        total = int(exclusive_scan(sizes, offsets, n_cells, close=True))
        loops = tack.empty((total,), idt)
        for group in groups:
            if group.shape in _ROUND:
                order = _table(("round", group.shape), _ROUND[group.shape])
                _shape_loops(data.domain_view("cells", group), order, offsets, loops)
    total = int(loops.shape[0])
    positions = _point_values(data.geometry.values, data.num_points)
    keys = tack.empty((total,), tack.vec(2, idt))
    owners = tack.empty((total,), tack.i64)
    normals = _vectors(n_cells, data.dtype)
    if n_cells:
        _sides(loops, offsets, positions, n_cells, keys, owners, normals)
    count = 0
    if total:
        order = key_order(keys, data.num_points, idt)
        runs, count = run_offsets(keys, order)
    wanted = tack.array(np.array([0, boundary, feature, non_manifold, manifold], np.int32))
    kinds = tack.empty((count,), tack.i32)
    flags = tack.empty((count,), tack.i32)
    rows = tack.empty((count,), tack.vec(2, idt))
    chosen = 0
    slots = tack.empty((count,), tack.i64)
    if count:
        _classify(order, runs, keys, owners, normals, math.cos(math.radians(angle)), wanted,
                  kinds, flags, rows, count)
        chosen = int(exclusive_scan(flags, slots, count))
    connectivity = tack.empty((2 * chosen,), idt)
    kind_out = tack.empty((chosen,), tack.i32)
    if chosen:
        _chosen(flags, slots, kinds, rows, connectivity, kind_out, count)
    types, line_offsets = uniform_cells(shapes.LINE, chosen, idt)
    lines = UnstructuredTopology(types, line_offsets, connectivity, num_points=data.num_points)
    out = carry(data, lines, points=Same(), fields=fields)
    out.fields["kind"] = Field(Values(out, "cells"), kind_out)
    return out

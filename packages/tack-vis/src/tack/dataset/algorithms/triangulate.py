# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""``triangulate``: every 2D cell as triangles: a quad or pixel in 2, a polygon by ear cutting; other cells are left out. The points are the same; cell fields go to each cell's triangles."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.arrays import materialize, size_of
from tack.dataset import ids, shapes
from tack.dataset.algorithms._refine import _ears, _fan_counts, _pieces_topology, _split
from tack.runtime.dispatch import on_inputs_device

__all__ = ['triangulate']


@on_inputs_device
def triangulate(data, fields=None):
    """Every 2D cell as triangles: a quad or pixel in 2, a polygon of ``n`` corners in
    ``n - 2`` by ear cutting in its plane (as VTK's vtkTriangleFilter), so a concave
    one is covered once, each triangle wound as the polygon; other cells are left
    out. The points are the same; cell fields go to each cell's triangles."""
    from tack.dataset.ghosts import visible

    data = visible(data)
    t = data.topology
    if getattr(t, "reference_cells", True):
        return _split(data, "triangles", fields)
    if t.dimension != 2:
        raise NotImplementedError(f"triangulate takes 2D cells, not {t.dimension}D ones")
    loop_offsets, loop_points = t.loops()
    n = data.num_cells
    odt = ids.for_output(data, 3 * size_of(loop_points))
    counts = tack.empty((n,), tack.i32)
    starts = tack.empty((n,), odt)
    if n:
        _fan_counts(loop_offsets, counts)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((3 * count,), odt)
    sources = tack.empty((count,), data.id_dtype)
    if count:
        # The corners left of each polygon as it is cut, linked within its range.
        after = tack.empty((size_of(loop_points),), tack.i32)
        before = tack.empty((size_of(loop_points),), tack.i32)
        _ears(loop_offsets, loop_points, materialize(data.geometry.values), starts, after,
              before, connectivity, sources)
    return _pieces_topology(data, connectivity, sources, count, shapes.TRIANGLE, 3, fields)

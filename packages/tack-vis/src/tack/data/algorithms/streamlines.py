"""``streamlines``: the paths of ``advect``'s particles, as Viskores' Streamline: each seed's positions after every step, the seed first, as consecutive line cells (one per step; a polyline per seed in Viskores, until Tack has a line topology), with ``seed`` (which seed's path) on the cells and ``time`` on the points. A seed that takes no step has a point and no line."""

import numpy as np

import tack
from tack.data import shapes
from tack.data.algorithms._advection import _run
from tack.data.dataset import DataSet, Field
from tack.data.spaces import Values

__all__ = ['streamlines']


@tack.kernel
def _segments(trace, trace_times, lengths, starts, capacity, points, times, types, offsets,
              connectivity, seed_of, segment_starts):
    # Each path's points in order, and a line from each to the next.
    for q in range(lengths.shape[0]):
        first = starts[q]
        count = lengths[q]
        lines = segment_starts[q]
        for k in range(count):
            points[first + k] = trace[q * capacity + k]
            times[first + k] = trace_times[q * capacity + k]
        for k in range(count - 1):
            types[lines + k] = tack.u8(shapes.LINE)
            offsets[lines + k] = 2 * (lines + k)
            connectivity[2 * (lines + k)] = first + k
            connectivity[2 * (lines + k) + 1] = first + k + 1
            seed_of[lines + k] = q
        if q == lengths.shape[0] - 1:
            offsets[lines + max(count - 1, 0)] = 2 * (lines + max(count - 1, 0))


def streamlines(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """The paths of ``advect``'s particles, as Viskores' Streamline: each seed's
    positions after every step, the seed first, as consecutive line cells (one
    per step; a polyline per seed in Viskores, until Tack has a line topology),
    with ``seed`` (which seed's path) on the cells and ``time`` on the points.
    A seed that takes no step has a point and no line."""
    from tack.algorithms.scan import exclusive_scan
    from tack.data.topology import UnstructuredTopology

    _, _, _, _, (trace, trace_times), lengths, capacity = _run(
        data, field, seeds, step_size, steps, integrator, locator, record=True)
    m = int(lengths.shape[0])
    host_lengths = lengths.to_numpy() if m else np.zeros(0, np.int32)
    total = int(host_lengths.sum())
    nlines = int(np.maximum(host_lengths - 1, 0).sum())
    idt = tack.i64 if max(total, 2 * nlines) > 2**31 - 1 else tack.i32
    starts = tack.field(idt, shape=(m,))
    segment_starts = tack.field(idt, shape=(m,))
    if m:
        exclusive_scan(lengths, starts, m)
        segment_starts.from_numpy(np.concatenate(
            [[0], np.cumsum(np.maximum(host_lengths - 1, 0))[:-1]]).astype(idt.numpy_dtype))
    points = tack.Vector.field(3, data.dtype, shape=(total,))
    point_times = tack.field(data.dtype, shape=(total,))
    types = tack.field(tack.u8, shape=(nlines,))
    offsets = tack.field(idt, shape=(nlines + 1,))
    connectivity = tack.field(idt, shape=(2 * nlines,))
    seed_of = tack.field(tack.i32, shape=(nlines,))
    offsets.fill(0)
    if m:
        _segments(trace, trace_times, lengths, starts, capacity, points, point_times, types,
                  offsets, connectivity, seed_of, segment_starts)
    topology = UnstructuredTopology(types, offsets, connectivity, num_points=total)
    out = DataSet(topology, points)
    out.fields["seed"] = Field(Values(out, "cells"), seed_of)
    out.fields["time"] = Field(Values(out, "points"), point_times)
    return out

"""``streamlines``: the paths of ``advect``'s particles, as Viskores' Streamline: one polyline per seed."""

import numpy as np

import tack
from tack.data.algorithms._advection import _run
from tack.data.dataset import DataSet, Field
from tack.data.spaces import Values

__all__ = ['streamlines']


@tack.kernel
def _paths(trace, trace_times, lengths, starts, capacity, points, times):
    # Each path's points in order.
    for q in range(lengths.shape[0]):
        first = starts[q]
        for k in range(lengths[q]):
            points[first + k] = trace[q * capacity + k]
            times[first + k] = trace_times[q * capacity + k]


@tack.kernel
def _chains(kept, starts, lengths, line_offsets, line_points):
    # A polyline per path that moved, through its points in order.
    for i in range(kept.shape[0]):
        q = kept[i]
        for k in range(lengths[q]):
            line_points[line_offsets[i] + k] = starts[q] + k


def streamlines(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """The paths of ``advect``'s particles, as Viskores' Streamline: each seed's
    positions after every step, the seed first, as one polyline per seed
    (a ``PolylineTopology``), with ``seed`` (which seed's path) on the cells and
    ``time`` on the points. A seed that takes no step has a point and no line."""
    from tack.algorithms.scan import exclusive_scan
    from tack.data.polyhedra import PolylineTopology

    _, _, _, _, (trace, trace_times), lengths, capacity = _run(
        data, field, seeds, step_size, steps, integrator, locator, record=True)
    m = int(lengths.shape[0])
    host_lengths = lengths.to_numpy() if m else np.zeros(0, np.int32)
    total = int(host_lengths.sum())
    moved = np.flatnonzero(host_lengths > 1)
    idt = tack.i64 if total > 2**31 - 1 else tack.i32
    starts = tack.field(idt, shape=(m,))
    if m:
        exclusive_scan(lengths, starts, m, total=False)
    points = tack.Vector.field(3, data.dtype, shape=(total,))
    point_times = tack.field(data.dtype, shape=(total,))
    if m:
        _paths(trace, trace_times, lengths, starts, capacity, points, point_times)
    # The paths that moved, their points numbered as they are, make the chains.
    kept = tack.field(idt, shape=(len(moved),))
    line_offsets = tack.field(idt, shape=(len(moved) + 1,))
    line_points = tack.field(idt, shape=(int(host_lengths[moved].sum()),))
    seed_of = tack.field(tack.i32, shape=(len(moved),))
    line_offsets.from_numpy(np.concatenate([[0], np.cumsum(host_lengths[moved])])
                            .astype(idt.numpy_dtype))
    if len(moved):
        kept.from_numpy(moved.astype(idt.numpy_dtype))
        seed_of.from_numpy(moved.astype(np.int32))
        _chains(kept, starts, lengths, line_offsets, line_points)
    topology = PolylineTopology(line_offsets, line_points, num_points=total, id_dtype=idt)
    out = DataSet(topology, points)
    out.fields["seed"] = Field(Values(out, "cells"), seed_of)
    out.fields["time"] = Field(Values(out, "points"), point_times)
    return out

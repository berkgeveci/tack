"""``streamlines``: the paths of ``advect``'s particles, as Viskores' Streamline: one polyline per seed."""

import numpy as np

import tack
from tack.dataset import ids
from tack.dataset.algorithms._advection import _run
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import Values

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
    from tack.dataset.polyhedra import PolylineTopology

    _, _, _, _, (trace, trace_times), lengths, capacity = _run(
        data, field, seeds, step_size, steps, integrator, locator, record=True)
    m = int(lengths.shape[0])
    host_lengths = lengths.numpy() if m else np.zeros(0, np.int32)
    total = int(host_lengths.sum())
    moved = np.flatnonzero(host_lengths > 1)
    idt = ids.for_output(data, total)
    starts = tack.empty((m,), idt)
    if m:
        exclusive_scan(lengths, starts, m, total=False)
    points = tack.empty((total,), tack.vec(3, data.dtype))
    point_times = tack.empty((total,), data.dtype)
    if m:
        _paths(trace, trace_times, lengths, starts, capacity, points, point_times)
    # The paths that moved, their points numbered as they are, make the chains.
    kept = tack.empty((len(moved),), idt)
    line_offsets = tack.empty((len(moved) + 1,), idt)
    line_points = tack.empty((int(host_lengths[moved].sum()),), idt)
    seed_of = tack.empty((len(moved),), tack.i32)
    tack.copyto(line_offsets, np.concatenate([[0], np.cumsum(host_lengths[moved])])
                            .astype(idt.numpy_dtype))
    if len(moved):
        tack.copyto(kept, moved.astype(idt.numpy_dtype))
        tack.copyto(seed_of, moved.astype(np.int32))
        _chains(kept, starts, lengths, line_offsets, line_points)
    topology = PolylineTopology(line_offsets, line_points, num_points=total, id_dtype=idt)
    out = DataSet(topology, points)
    out.fields["seed"] = Field(Values(out, "cells"), seed_of)
    out.fields["time"] = Field(Values(out, "points"), point_times)
    return out

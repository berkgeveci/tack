"""``streamlines``: the paths of ``advect``'s particles, as Viskores' Streamline: one polyline per seed."""

import tack
from tack.algorithms.scan import exclusive_scan
from tack.dataset import ids
from tack.dataset.algorithms._advection import _run
from tack.dataset.algorithms._common import _close
from tack.dataset.dataset import DataSet, Field
from tack.dataset.spaces import Values
from tack.runtime.dispatch import on_inputs_device

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
def _moved(lengths, flags):
    # A path of two points or more is a line.
    for q in range(lengths.shape[0]):
        flags[q] = 1 if lengths[q] > 1 else 0


@tack.kernel
def _moved_paths(flags, slots, lengths, kept, seed_of, line_lengths):
    for q in range(flags.shape[0]):
        if flags[q] == 1:
            kept[slots[q]] = q
            seed_of[slots[q]] = q
            line_lengths[slots[q]] = lengths[q]


@tack.kernel
def _chains(kept, starts, lengths, line_offsets, line_points):
    # A polyline per path that moved, through its points in order.
    for i in range(kept.shape[0]):
        q = kept[i]
        for k in range(lengths[q]):
            line_points[line_offsets[i] + k] = starts[q] + k


@on_inputs_device
def streamlines(data, field, seeds, step_size, steps, integrator="rk4", locator=None):
    """The paths of ``advect``'s particles, as Viskores' Streamline: each seed's
    positions after every step, the seed first, as one polyline per seed
    (a ``PolylineTopology``), with ``seed`` (which seed's path) on the cells and
    ``time`` on the points. A seed that takes no step has a point and no line.

    Over a ``PartitionedDataSet`` (``field`` by name), paths cross from piece to
    piece, each piece holding the stretches it traced
    (``_particles.streamlines_pieces``)."""
    from tack.dataset.partitioned import PartitionedDataSet

    if isinstance(data, PartitionedDataSet):
        from tack.dataset.algorithms._particles import streamlines_pieces
        return streamlines_pieces(data, field, seeds, step_size, steps, integrator)
    from tack.dataset.ghosts import refuse_hidden, visible

    if locator is None:
        data, field = visible(data, field)
    else:
        refuse_hidden(data, "streamlines with a locator")
    from tack.dataset.polyhedra import PolylineTopology

    _, _, _, _, (trace, trace_times), lengths, capacity = _run(
        data, field, seeds, step_size, steps, integrator, locator, record=True)
    # Everything is chosen and numbered on the device; only counts come back.
    m = int(lengths.shape[0])
    idt = ids.for_output(data, m * capacity)
    starts = tack.empty((m,), idt)
    total = exclusive_scan(lengths, starts, m) if m else 0
    points = tack.empty((total,), tack.vec(3, data.dtype))
    point_times = tack.empty((total,), data.dtype)
    if m:
        _paths(trace, trace_times, lengths, starts, capacity, points, point_times)
    # The paths that moved, their points numbered as they are, make the chains.
    flags = tack.empty((m,), tack.i32)
    slots = tack.empty((m,), idt)
    if m:
        _moved(lengths, flags)
    lines = exclusive_scan(flags, slots, m) if m else 0
    kept = tack.empty((lines,), idt)
    seed_of = tack.empty((lines,), tack.i32)
    line_lengths = tack.empty((lines,), idt)
    if lines:
        _moved_paths(flags, slots, lengths, kept, seed_of, line_lengths)
    line_offsets = tack.empty((lines + 1,), idt)
    length = exclusive_scan(line_lengths, line_offsets, lines) if lines else 0
    _close(line_offsets, lines, length)
    line_points = tack.empty((length,), idt)
    if lines:
        _chains(kept, starts, lengths, line_offsets, line_points)
    topology = PolylineTopology(line_offsets, line_points, num_points=total, id_dtype=idt)
    out = DataSet(topology, points, origin=data.origin)
    out.fields["seed"] = Field(Values(out, "cells"), seed_of)
    out.fields["time"] = Field(Values(out, "points"), point_times)
    return out

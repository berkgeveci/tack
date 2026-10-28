"""Particles that cross the pieces of a partitioned dataset: ``advect`` and
``streamlines`` over a ``PartitionedDataSet``, written on ``tack.communication``.

Rounds until no particle is in flight:

1. **Route.** Each particle in flight goes to the pieces whose boxes hold it;
   each asks its locator whether one of its own cells (not a ghost) holds the
   particle, and the lowest piece that says yes takes it. A particle no piece
   takes has left the domain (``SPATIAL_BOUNDS``; a seed outside also loses
   ``SUCCESS``, as on one dataset).
2. **Advance.** Each piece advects what it took, going on from each particle's
   time, steps and status. On a piece with ghost cells a particle stops after a
   whole step that ends in a ghost cell (``HANDOFF``), and the cell's owner goes
   on from there with whole steps: the path is the whole mesh's wherever a step
   fits in the ghost layer (``step_size`` times the speed under its width).
   Without ghost cells it stops where it leaves the piece, by the short step to
   the boundary and the push out, as on one dataset, and goes on in the piece
   it was pushed into.
3. Particles still in flight go back to 1.

Every round each particle in flight takes a step or stops, so the rounds end.
The particles' states are on the host between rounds, and messages carry
coordinates and times as f64 bits in u64 arrays, which every device stores.
"""

import numpy as np

import tack
from tack import arrays
from tack import communication as tc
from tack.dataset.algorithms._advection import (
    HANDOFF,
    SPATIAL_BOUNDS,
    SUCCESS,
    TERMINATE,
    ZERO_VELOCITY,
    _run,
)
from tack.dataset.algorithms._common import _time_unit
from tack.dataset.algorithms._extract import _vertices
from tack.dataset.dataset import DataSet, Field
from tack.dataset.ghosts import Ghosts, visible
from tack.dataset.locator import cached_locator
from tack.dataset.partitioned import PartitionedDataSet, common_origin, fields_of
from tack.dataset.spaces import Values

__all__ = ["advect_pieces", "streamlines_pieces"]

_FIELDS = ("time", "steps", "status", "seed", "segments")
_DTYPES = {"time": np.float64, "steps": np.int32, "status": np.int32, "seed": np.int64,
           "segments": np.int32}


class _Particles:
    """Particles on the host: positions (f64), and per particle ``time``,
    ``steps``, ``status``, ``seed`` (its global number) and ``segments`` (the
    pieces' runs it has had, for ``streamlines``)."""

    def __init__(self, x=None, **values):
        self.x = np.zeros((0, 3)) if x is None else np.asarray(x, np.float64).reshape(-1, 3)
        for name in _FIELDS:
            setattr(self, name, np.asarray(values.get(name, np.zeros(len(self.x))),
                                           _DTYPES[name]))

    def __len__(self):
        return len(self.x)

    def take(self, rows):
        return _Particles(self.x[rows], **{n: getattr(self, n)[rows] for n in _FIELDS})

    @staticmethod
    def joined(parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return _Particles()
        return _Particles(np.concatenate([p.x for p in parts]),
                          **{n: np.concatenate([getattr(p, n) for p in parts])
                             for n in _FIELDS})

    def message(self, device, rows=None):
        """These particles (or ``rows`` of them) as arrays to send."""
        part = self if rows is None else self.take(rows)
        out = _bits(part.x, device)
        out["time"] = tack.array(np.ascontiguousarray(part.time).view(np.uint64), tack.u64,
                                 device=device)
        for name in ("steps", "status", "segments"):
            out[name] = tack.array(getattr(part, name), tack.i32, device=device)
        out["seed"] = tack.array(part.seed, tack.i64, device=device)
        return out

    @staticmethod
    def received(message):
        return _Particles(_unbits(message),
                          time=message["time"].numpy().view(np.float64),
                          **{n: message[n].numpy() for n in ("steps", "status", "seed",
                                                             "segments")})


def _bits(x, device):
    bits = np.ascontiguousarray(np.asarray(x, np.float64).reshape(-1, 3)).view(np.uint64)
    return {f"x{axis}": tack.array(bits[:, axis].copy(), tack.u64, device=device)
            for axis in range(3)}


def _unbits(message):
    return np.stack([message[f"x{axis}"].numpy().view(np.float64) for axis in range(3)],
                    axis=1)


def advect_pieces(data, field, seeds, step_size, steps, integrator="rk4", neighbors=False):
    """``advect`` over the pieces of ``data``: the particles where they stopped, a
    ``PartitionedDataSet`` of one vertex dataset per piece -- each particle on
    the piece it stopped in, or left from -- with ``steps``, ``time``,
    ``status`` and ``seed`` (its number among every rank's seeds, in order).

    ``seeds`` are this rank's, an ``(n, 3)`` array; across MPI ranks each rank
    passes its own, numbered after the ranks before it."""
    blocks, pieces, finished, _ = _rounds(data, field, seeds, step_size, steps, integrator,
                                          record=False, neighbors=neighbors)
    out = []
    for g, piece in zip(blocks.local, pieces):
        done = finished[g]
        order = np.argsort(done.seed, kind="stable")
        done = done.take(order)
        with blocks.device(g):
            vertices = _vertices(np.asarray(done.x, piece.dtype.numpy_dtype).reshape(-1, 3),
                                 piece)
            points = Values(vertices, "points")
            vertices.fields["steps"] = Field(points, tack.array(done.steps, tack.i32))
            vertices.fields["time"] = Field(points, tack.array(
                done.time.astype(piece.dtype.numpy_dtype), piece.dtype),
                _time_unit(piece, field))
            vertices.fields["status"] = Field(points, tack.array(done.status, tack.i32))
            vertices.fields["seed"] = Field(points, tack.array(done.seed, tack.i64))
        out.append(vertices)
    return PartitionedDataSet(out, world=blocks.world)


def streamlines_pieces(data, field, seeds, step_size, steps, integrator="rk4", unit="time",
                       neighbors=False):
    """``streamlines`` over the pieces of ``data``: each piece's share of the
    paths, a ``PartitionedDataSet`` of polyline datasets. A particle's path is
    cut where it crosses from piece to piece: each piece holds the stretches it
    traced, as polylines with ``seed`` (whose path) and ``segment`` (which
    stretch, from 0) on the cells and ``time`` on the points. A stretch after
    the first starts at the point the one before ended at, which that point's
    copy marks ``DUPLICATEPOINT``, so each path's points are owned once."""
    from tack.dataset.polyhedra import PolylineTopology

    blocks, pieces, _, traced = _rounds(data, field, seeds, step_size, steps, integrator,
                                        record=True, unit=unit, neighbors=neighbors)
    out = []
    for g, piece in zip(blocks.local, pieces):
        stretches = sorted(traced[g], key=lambda s: (s[0], s[1]))
        positions, times, marks, offsets, members, seed, segment = [], [], [], [0], [], [], []
        count = 0
        for seed_id, index, x, t in stretches:
            n = len(x)
            positions.append(x)
            times.append(t)
            mark = np.zeros(n, np.uint8)
            if index > 0:
                mark[0] = Ghosts.DUPLICATEPOINT
            marks.append(mark)
            if n >= 2:
                members.extend(range(count, count + n))
                offsets.append(offsets[-1] + n)
                seed.append(seed_id)
                segment.append(index)
            count += n
        dtype = piece.dtype.numpy_dtype
        points = np.concatenate(positions).astype(dtype) if positions else np.zeros((0, 3), dtype)
        with blocks.device(g):
            topology = PolylineTopology(np.asarray(offsets, np.int64),
                                        np.asarray(members, np.int64), num_points=count)
            lines = DataSet(topology, points, dtype=piece.dtype,
                            ghosts=Ghosts(points=np.concatenate(marks) if marks
                                          else np.zeros(0, np.uint8)), **piece._frame())
            lines.fields["seed"] = Field(Values(lines, "cells"),
                                         tack.array(np.asarray(seed, np.int64), tack.i64))
            lines.fields["segment"] = Field(Values(lines, "cells"),
                                            tack.array(np.asarray(segment, np.int32),
                                                       tack.i32))
            lines.fields["time"] = Field(Values(lines, "points"), tack.array(
                (np.concatenate(times) if times else np.zeros(0)).astype(dtype),
                piece.dtype), _time_unit(piece, field))
        out.append(lines)
    return PartitionedDataSet(out, world=blocks.world)


def _rounds(data, field, seeds, step_size, steps, integrator, record, unit="time",
            neighbors=False):
    """The rounds: ``(blocks, pieces, finished, traced)``, the particles each piece
    finished with, and with ``record`` each piece's stretches ``(seed, segment,
    positions, times)``."""
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    fields_of(data, field, "advection")
    pieces = []
    for piece in data:                       # each piece's work on its own device
        with piece.device:
            pieces.append(visible(piece, field)[0])
    blocks = tc.Blocks([piece.device for piece in pieces], data.world)
    locators = {}
    for g, piece in zip(blocks.local, pieces):
        with piece.device:
            locators[g] = cached_locator(piece)
    boxes = _boxes(blocks, pieces)
    pending = _seeded(blocks, seeds, common_origin(data, "advection over pieces"))
    finished = {g: [] for g in blocks.local}
    traced = {g: [] for g in blocks.local}
    for _ in range(int(steps) + 2):
        if not blocks.all_reduce({g: len(pending[g]) for g in blocks.local}):
            break
        arrived, outside = _route(blocks, pieces, locators, boxes, pending)
        for g in blocks.local:
            finished[g].append(outside[g])
            pending[g], done = _advance(pieces[g - blocks.local.start], field,
                                        arrived[g], step_size, steps, integrator, record,
                                        traced[g], unit, neighbors)
            finished[g].append(done)
    else:
        raise RuntimeError("particles were still in flight after a round per step")
    return blocks, pieces, {g: _Particles.joined(finished[g]) for g in blocks.local}, traced


def _boxes(blocks, pieces):
    """This rank's pieces' boxes as the locator sees them (``probe._held_box``),
    ``None`` for a piece with no points."""
    from tack.dataset.algorithms.probe import _held_box

    return {g: _held_box(piece) if piece.num_points else None
            for g, piece in zip(blocks.local, pieces)}


def _seeded(blocks, seeds, origin):
    """This rank's seeds, numbered after the ranks before it, on its first block:
    host seeds are world positions, taken relative to the pieces' ``origin``; a
    Tack array's are relative to it already."""
    if hasattr(seeds, "numpy"):
        host = np.asarray(seeds.numpy(), np.float64).reshape(-1, 3)
    else:
        host = np.asarray(seeds, np.float64).reshape(-1, 3) - np.asarray(origin)
    if len(host) and not len(blocks.local):
        raise ValueError("a rank with no pieces has no piece to start its seeds from")
    first = blocks.local.start if len(blocks.local) else None
    offsets, _ = blocks.exclusive_scan({g: len(host) if g == first else 0
                                        for g in blocks.local})
    pending = {g: _Particles() for g in blocks.local}
    if first is not None:
        n = len(host)
        pending[first] = _Particles(host, status=np.full(n, int(SUCCESS)),
                                    seed=offsets[first] + np.arange(n))
    return pending


def _route(blocks, pieces, locators, boxes, pending):
    """Each particle in flight to the lowest piece one of whose own cells holds it
    (``tack.communication.Routing``): ``(arrived, outside)``, the particles each
    piece takes, and those no piece takes, left with the piece they were on."""
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

    routing = tc.Routing(blocks, {g: pending[g].x for g in blocks.local}, boxes, holds)
    outside, state = {}, {}
    for g in blocks.local:
        particles = pending[g]
        left = particles.take(np.flatnonzero(routing.destination[g] < 0))
        left.status = left.status | int(SPATIAL_BOUNDS)
        never = left.steps == 0
        left.status[never] &= ~int(SUCCESS)          # a seed outside the mesh
        outside[g] = left
        message = particles.message(blocks.device(g))
        state[g] = {n: v for n, v in message.items() if not n.startswith("x")}
    arrived = {}
    for g, (x, values) in routing.send(state).items():
        if not len(x):
            arrived[g] = _Particles()
            continue
        values = {**values, **_bits(x, blocks.device(g))}
        arrived[g] = _Particles.received(values)
    return arrived, outside


def _advance(piece, field, particles, step_size, steps, integrator, record, traced,
             unit="time", neighbors=False):
    """Advect a piece's particles: ``(in flight, finished)``. With ``record``, each
    particle's stretch here goes into ``traced``."""
    if not len(particles):
        return _Particles(), _Particles()
    dtype = piece.dtype.numpy_dtype
    resume = (particles.time.astype(dtype), particles.steps, particles.status)
    with piece.device:
        xs, times, taken, status, (trace, trace_times), lengths, capacity = _run(
            piece, field, np.asarray(particles.x, dtype), step_size, steps, integrator,
            None, record, resume=resume, handoff=True, unit=unit, neighbors=neighbors)
    out = _Particles(xs.numpy().reshape(-1, 3), time=times.numpy(), steps=taken.numpy(),
                     status=status.numpy(), seed=particles.seed,
                     segments=particles.segments + 1)
    if record:
        lengths = lengths.numpy()
        trace = trace.numpy().reshape(-1, capacity, 3)
        trace_times = trace_times.numpy().reshape(-1, capacity)
        for q in range(len(out)):
            n = int(lengths[q])
            traced.append((int(particles.seed[q]), int(particles.segments[q]),
                           trace[q, :n].astype(np.float64), trace_times[q, :n]))
    moving = (out.status & (int(HANDOFF) | int(SPATIAL_BOUNDS))) != 0
    stopped = (out.status & (int(TERMINATE) | int(ZERO_VELOCITY))) != 0
    flying = out.take(np.flatnonzero(moving & ~stopped))
    flying.status = flying.status & ~(int(HANDOFF) | int(SPATIAL_BOUNDS))
    return flying, out.take(np.flatnonzero(~(moving & ~stopped)))

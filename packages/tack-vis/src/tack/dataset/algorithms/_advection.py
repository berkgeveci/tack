"""The advection kernel ``advect`` and ``streamlines`` share, by Viskores' rules:
RK4 or Euler steps; a step leaving the mesh bisected to the boundary and the
particle pushed out; a stop after its steps, outside, or at zero velocity.
Each particle's path is one thread, its stages located from the particle's
side by ``by_id``'s search (over ``CellLocator.cell_bins``, ranked as the
locator ranks cells), its velocity read by cell id there, as
``CellLocator.find`` and ``probe`` do. On the CPU the whole path is one
launch; on a GPU, launches of a bounded number of steps, sized to a fraction
of a second, each going on where the last stopped: a GPU driver resets a
device whose launch runs past its limit (7.5 s on headroom's Level Zero, 2 s
by default on Windows), and a lost device takes every array with it."""

import enum

import numpy as np

import tack
from tack import arrays
from tack.dataset import ids
from tack.dataset.algorithms._common import _field
from tack.dataset.by_id import _as_vectors, cells_by_id, field_by_id, mesh_search
from tack.dataset.dataset import Field
from tack.dataset.locator import cached_locator
from tack.dataset.spaces import H1
from tack.lang.array import Array

# A particle's status bits, Viskores' ParticleStatus: the kernel's constants,
# and the flag users compare against.
SUCCESS = tack.constant(1, tack.i32)
TERMINATE = tack.constant(2, tack.i32)
SPATIAL_BOUNDS = tack.constant(4, tack.i32)
TOOK_ANY_STEPS = tack.constant(16, tack.i32)
ZERO_VELOCITY = tack.constant(64, tack.i32)
# Not Viskores': the particle ended a whole step in a ghost cell, so it goes to
# the piece that owns that cell (advection over a partitioned dataset). The
# driver clears it; users never see it.
HANDOFF = tack.constant(128, tack.i32)


class ParticleStatus(enum.IntFlag):
    """The bits of ``advect``'s ``status`` field, Viskores' ParticleStatus:
    ``SUCCESS`` (cleared when the seed was outside the mesh), ``TERMINATE`` (it
    stopped after its steps or at zero velocity), ``SPATIAL_BOUNDS`` (it left
    the mesh), ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``."""

    SUCCESS = 1
    TERMINATE = 2
    SPATIAL_BOUNDS = 4
    TOOK_ANY_STEPS = 16
    ZERO_VELOCITY = 64


# What the next point location is for.
_STEP = tack.constant(0, tack.i32)       # a stage of a whole step


_START = tack.constant(1, tack.i32)      # the particle itself, starting the bisection


_CHECK = tack.constant(2, tack.i32)      # a stage of a bisection's trial step


_TRIAL = tack.constant(3, tack.i32)      # where the trial step ends


_LAST = tack.constant(4, tack.i32)       # the last point inside, for the push out


_OUT = tack.constant(5, tack.i32)        # where the push out ends


@tack.kernel
def _advect(mesh, velocity, h, max_steps, rk4, by_length, eps, xs, times, steps, status,
            trace, trace_times, lengths, capacity, record, ghosts, handoff, budget, first,
            moved, hints):
    # by_length: h is an arc length, and the particle follows the velocity's
    # direction, v / |v| -- vtkStreamTracer's length unit -- with the time a
    # unit of length takes, 1 / |v|, integrated beside it by the same weights.
    # One launch takes at most budget steps of each particle and pauses it after
    # a whole step, where its state is its position, time, steps and status (the
    # cell it was last found in only speeds the search); the next launch
    # (first 0) goes on from there, its trace after lengths[q] points, with
    # moved[q], whether it took a step in this call, which a hand-off needs,
    # and hints[q], the cell it was last found in: the hint and the search
    # find the same cell, but a GPU compiler may round the two call sites'
    # arithmetic apart, so a launch goes on with the hint one launch would
    # have had.
    for q in range(xs.shape[0]):
        x = xs[q]
        t = times[q]
        n = steps[q]
        st = status[q]
        took = moved[q]
        here = 0
        length = lengths[q]
        if record == 1 and first == 1:
            trace[q * capacity] = x
            trace_times[q * capacity] = t
            length = 1
        k1 = x - x
        k2 = k1
        k3 = k1
        vel = k1
        w1 = h - h
        w2 = w1
        w3 = w1
        curr = x
        out = x
        mode = _STEP
        stage = 0
        sub = h
        r0 = h - h
        r1 = h
        div = 1.0
        last = mesh.ncells        # the cell the particle was last found in
        if first == 0:
            last = hints[q]
        # A resumed particle that stopped -- outside, at zero velocity, handed on,
        # or with its steps done -- stays where it is; one that stopped at an
        # earlier call's step limit goes on.
        running = 1
        if (st & (SPATIAL_BOUNDS | ZERO_VELOCITY | HANDOFF)) != 0 or n >= max_steps:
            running = 0
        elif first == 1:
            st = st & ~TERMINATE
        while running == 1:
            # Where to locate the particle next.
            p = x
            if mode == _STEP or mode == _CHECK:
                if stage == 1:
                    p = x + (0.5 * sub) * k1
                elif stage == 2:
                    p = x + (0.5 * sub) * k2
                elif stage == 3:
                    p = x + sub * k3
            elif mode == _TRIAL:
                p = x + sub * vel
            elif mode == _START or mode == _LAST:
                p = curr
            elif mode == _OUT:
                p = out

            # Its cell, ranked as the locator ranks them -- holding the point
            # ahead of coming within the tolerance, then the smaller id -- tried
            # first in the cell it was last found in, and the velocity there
            # (by_id).
            best, best_type, best_pc, best_tet = mesh.find_from(p, last)
            found = 1 if best < mesh.ncells else 0
            if found == 1:
                last = best
            v = p - p
            if found == 1:
                v = velocity.value(mesh, best, best_type, best_pc, best_tet)
            # Along the direction, the time a unit of length takes; a speed
            # at zero velocity's threshold stops the particle as before.
            w = h - h + 1.0
            if by_length == 1:
                if v.dot(v) > eps:
                    w = 1.0 / v.norm()
                    v = v * w
                else:
                    v = v - v
                    w = h - h

            # A whole step that ended in a ghost cell hands the particle to the
            # cell's owner, which goes on from there with whole steps; only
            # after a step here, so a particle never goes back without moving.
            handed = 0
            if handoff == 1 and mode == _STEP and stage == 0 and took == 1 and found == 1:
                if (ghosts[best] & 1) != 0:
                    handed = 1

            # What it means for the particle.
            bisect = 0
            if handed == 1:
                st = st | HANDOFF
                running = 0
            elif mode == _STEP or mode == _CHECK:
                if found == 0:
                    if mode == _STEP:
                        # Viskores' SmallStep: from the particle, bisect the step.
                        mode = _START
                        curr = x
                        r0 = h - h
                        r1 = h
                        div = 1.0
                    else:
                        r1 = sub
                        bisect = 1
                elif rk4 == 1 and stage < 3:
                    if stage == 0:
                        k1 = v
                        w1 = w
                    elif stage == 1:
                        k2 = v
                        w2 = w
                    else:
                        k3 = v
                        w3 = w
                    stage += 1
                else:
                    vel = v
                    pace = w
                    if rk4 == 1:
                        vel = (k1 + 2.0 * k2 + 2.0 * k3 + v) / 6.0
                        pace = (w1 + 2.0 * w2 + 2.0 * w3 + w) / 6.0
                    else:
                        k1 = v
                    stage = 0
                    if mode == _STEP:
                        x = x + h * vel
                        if by_length == 1:
                            t += h * pace
                        else:
                            t += h
                        n += 1
                        here += 1
                        took = 1
                        if record == 1:
                            trace[q * capacity + length] = x
                            trace_times[q * capacity + length] = t
                            length += 1
                        if vel.dot(vel) <= eps:
                            st = st | ZERO_VELOCITY | TERMINATE
                            running = 0
                        elif n == max_steps:
                            st = st | TERMINATE
                            running = 0
                        elif here == budget:
                            running = 0
                    else:
                        mode = _TRIAL
            elif mode == _START:
                if found == 0:
                    # The seed itself is outside: it cannot move.
                    st = (st & ~SUCCESS) | SPATIAL_BOUNDS
                    running = 0
                else:
                    bisect = 1
            elif mode == _TRIAL:
                if found == 1:
                    curr = x + sub * vel
                    r0 = sub
                else:
                    r1 = sub
                bisect = 1
            elif mode == _LAST:
                vel = v
                out = curr + r1 * vel
                if by_length == 1:
                    t += r1 * w
                else:
                    t += r1
                mode = _OUT
            else:
                # Where the push out ends: the step is taken, in or out.
                x = out
                n += 1
                here += 1
                took = 1
                if record == 1:
                    trace[q * capacity + length] = x
                    trace_times[q * capacity + length] = t
                    length += 1
                if found == 0:
                    st = st | SPATIAL_BOUNDS
                if vel.dot(vel) <= eps:
                    st = st | ZERO_VELOCITY | TERMINATE
                if n == max_steps:
                    st = st | TERMINATE
                if (st & (SPATIAL_BOUNDS | ZERO_VELOCITY | TERMINATE)) != 0 or here == budget:
                    running = 0
                else:
                    mode = _STEP
                # Whole steps again, from their first stage and at h: they kept
                # the bisection's last trial size and, when a trial's stage left
                # the mesh, its stage, so a step after a push out that stayed
                # inside sampled its stages short of where it went, or began
                # mid-step on the trial's stage velocities.
                sub = h
                div = 1.0
                stage = 0
            if bisect == 1:
                # The next trial step, or the push out once the bracket is small.
                if r1 - r0 > 10.0 * eps:
                    div = div * 2.0
                    sub = r0 + h / div
                    mode = _CHECK
                    stage = 0
                else:
                    mode = _LAST
        xs[q] = x
        times[q] = t
        steps[q] = n
        status[q] = st | (TOOK_ANY_STEPS if took == 1 else 0)
        lengths[q] = length
        moved[q] = took
        hints[q] = last


@tack.kernel
def _running(steps, status, max_steps, count):
    # Particles a later launch would move: not stopped, steps left.
    for q in range(steps.shape[0]):
        if (status[q] & (SPATIAL_BOUNDS | ZERO_VELOCITY | HANDOFF | TERMINATE)) == 0 \
                and steps[q] < max_steps:
            tack.atomic_add(count, 0, 1)


# Launches of advection on a GPU: the first takes this many steps of each
# particle, and each later one as many as should take _LAUNCH_SECONDS at the
# last launch's pace, at most _GROWTH times as many. The CPU has no such
# limit and runs the whole path in one launch (_CHUNKED_ON_CPU: the tests).
_FIRST_STEPS = 16
_LAUNCH_SECONDS = 0.2
_GROWTH = 4
_CHUNKED_ON_CPU = False


def _launches(mesh, velocity, step_size, steps, args, device):
    """Run ``_advect`` (``args``, its arguments from ``rk4`` to ``handoff``) until no
    particle has steps left or is not stopped: on the CPU in one launch, else
    in launches of a bounded number of steps."""
    import time

    xs, taken, status = args[3], args[5], args[6]
    m = int(xs.shape[0])
    moved = tack.zeros((m,), tack.i32)
    hints = tack.empty((m,), ids.at_least(tack.i32, int(mesh.ncells)))
    if device.type == "cpu" and not _CHUNKED_ON_CPU:
        _advect(mesh, velocity, step_size, steps, *args, steps, 1, moved, hints)
        return
    count = tack.empty((1,), tack.i32)
    budget, first = min(_FIRST_STEPS, steps), 1
    while True:
        start = time.perf_counter()
        _advect(mesh, velocity, step_size, steps, *args, budget, first, moved, hints)
        count.fill(0)
        _running(taken, status, steps, count)
        if int(count[0]) == 0:
            return
        elapsed = time.perf_counter() - start
        budget = max(1, min(_GROWTH * budget, int(budget * _LAUNCH_SECONDS / max(elapsed, 1e-6))))
        first = 0


def _velocity(data, field):
    """The velocity, read by cell id (``by_id``): point data or cell data of
    3-vectors in the dataset's precision."""
    field = _field(data, field)
    if arrays.width_of(field.values) != 3:
        raise TypeError("advection needs a field of 3-vectors")
    velocity = Field(field.space, _as_vectors(arrays.materialize(field.values), data.dtype))
    by_id = field_by_id(data, velocity)
    if by_id is None:
        raise TypeError(f"advection takes point data (H1 order 1, values on points) or "
                        f"cell data, not {field.space!r}")
    return by_id


def _run(data, field, seeds, step_size, steps, integrator, locator, record, resume=None,
         handoff=False, unit="time"):
    """``resume``, ``(times, steps, status)`` for the seeds, goes on from there:
    ``steps`` counts the steps already taken, and a stopped particle stays.
    ``handoff`` stops a particle after a whole step ending in a ghost cell
    (``HANDOFF``). ``unit`` is what ``step_size`` measures: ``"time"``, or
    ``"length"`` along the path."""
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    if integrator not in ("rk4", "euler"):
        raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
    if unit not in ("time", "length"):
        raise ValueError(f"step_unit is 'time' or 'length', not {unit!r}")
    geometry = data.geometry
    if not isinstance(geometry.space, H1) or geometry.space.order != 1:
        raise TypeError("advection needs an order-1 H1 geometry: one position per point")
    velocity = _velocity(data, field)
    positions = _as_vectors(arrays.materialize(geometry.values), data.dtype)
    from tack.dataset.polyhedra import PolylineTopology

    if (isinstance(data.topology, PolylineTopology)
            or cells_by_id(data, positions, strict=False) is None):
        raise NotImplementedError("advection takes unstructured, polyhedral, polygonal and 2D "
                                  "or 3D structured topologies; not polylines, nor lines")
    locator = locator or cached_locator(data)
    if not locator.serves(data):
        raise ValueError("the locator was built for another topology or geometry, or "
                         "Tack has written the positions since: make a new one")
    mesh = mesh_search(data, locator, positions, strict=False)
    dtype = data.dtype
    # Seeds in an array of 3-vectors are copied on the device; anything else
    # comes from the host.
    # A Tack array holds positions relative to the dataset's origin, as its
    # geometry does; host seeds are world positions, taken relative to it in
    # double precision.
    stored = isinstance(seeds, Array) and seeds._vector_n == 3
    if stored:
        host = None
    elif isinstance(seeds, Array):
        host = np.ascontiguousarray(seeds.numpy().astype(dtype.numpy_dtype).reshape(-1, 3))
    else:
        host = np.ascontiguousarray((np.asarray(seeds, np.float64).reshape(-1, 3)
                                     - np.asarray(data.origin)).astype(dtype.numpy_dtype))
    m = int(seeds.shape[0]) if stored else len(host)
    xs = tack.empty((m,), tack.vec(3, dtype))
    times = tack.empty((m,), dtype)
    taken = tack.empty((m,), tack.i32)
    status = tack.empty((m,), tack.i32)
    lengths = tack.empty((m,), tack.i32)
    capacity = int(steps) + 1 if record else 1
    trace = tack.empty((m * capacity,), tack.vec(3, dtype))
    trace_times = tack.empty((m * capacity,), dtype)
    if m:
        tack.copyto(xs, seeds.to_device(xs.device) if stored else host)
        if resume is None:
            times.fill(0)
            taken.fill(0)
            status.fill(int(SUCCESS))
        else:
            for into, values, kind in zip((times, taken, status), resume,
                                          (dtype, tack.i32, tack.i32)):
                tack.copyto(into, values.to_device(into.device).astype(kind)
                            if isinstance(values, Array)
                            else np.asarray(values, kind.numpy_dtype))
        marks = data.ghosts.cells if handoff and data.ghosts is not None else None
        ghosts = marks if marks is not None else tack.zeros((1,), tack.u8)
        eps = float(np.finfo(dtype.numpy_dtype).eps)
        lengths.fill(0)
        _launches(mesh, velocity, float(step_size), int(steps),
                  (1 if integrator == "rk4" else 0, 1 if unit == "length" else 0, eps, xs, times,
                   taken, status, trace, trace_times, lengths, capacity, 1 if record else 0,
                   ghosts, 1 if marks is not None else 0), xs.device)
    return xs, times, taken, status, (trace, trace_times), lengths, capacity

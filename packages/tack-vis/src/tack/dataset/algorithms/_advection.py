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
from tack.dataset import ids, units
from tack.dataset.algorithms._common import _field, _has_quantity, _length, _time_unit
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
# Not Viskores': it stopped where the velocity's speed was at or below the
# terminal speed (vtkStreamTracer's TerminalSpeed), with TERMINATE.
SLOW_SPEED = tack.constant(256, tack.i32)


class ParticleStatus(enum.IntFlag):
    """The bits of ``advect``'s ``status`` field, Viskores' ParticleStatus:
    ``SUCCESS`` (cleared when the seed was outside the mesh), ``TERMINATE`` (it
    stopped after its steps, at zero velocity, at its path's ``max_length`` or
    below the terminal speed), ``SPATIAL_BOUNDS`` (it left the mesh),
    ``TOOK_ANY_STEPS``, ``ZERO_VELOCITY``; and, not Viskores', ``SLOW_SPEED``
    (it stopped where the speed was at or below ``terminal_speed``)."""

    SUCCESS = 1
    TERMINATE = 2
    SPATIAL_BOUNDS = 4
    TOOK_ANY_STEPS = 16
    ZERO_VELOCITY = 64
    SLOW_SPEED = 256


# What the next point location is for.
_STEP = tack.constant(0, tack.i32)       # a stage of a whole step


_START = tack.constant(1, tack.i32)      # the particle itself, starting the bisection


_CHECK = tack.constant(2, tack.i32)      # a stage of a bisection's trial step


_TRIAL = tack.constant(3, tack.i32)      # where the trial step ends


_LAST = tack.constant(4, tack.i32)       # the last point inside, for the push out


_OUT = tack.constant(5, tack.i32)        # where the push out ends


# The integrators, as the kernel numbers them.
_EULER = tack.constant(0, tack.i32)
_RK4 = tack.constant(1, tack.i32)
_RK45 = tack.constant(2, tack.i32)
_METHODS = {"euler": int(_EULER), "rk4": int(_RK4), "rk45": int(_RK45)}


@tack.kernel
def _advect(mesh, velocity, h, max_steps, method, by_length, eps, xs, times, steps, status,
            trace, trace_times, lengths, capacity, record, ghosts, handoff, budget, first,
            moved, hints, tol, min_step, max_step, max_length, terminal, hs, travelled):
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
    #
    # method: Euler, RK4, or RK45 (Cash-Karp, VTK's vtkRungeKutta45), whose step
    # adapts: a whole step's error estimate over tol shrinks it and tries again,
    # and an accepted step proposes the next, within [min_step, max_step], by
    # VTK's rules (0.9 * h * ratio ** -0.25 down, ** -0.2 up; a step at
    # min_step is taken whatever its error; a push out at the end of a bisection
    # that moves an RK45 particle less than min_step ends it, SPATIAL_BOUNDS). hs[q] is a particle's current step
    # (h for Euler and RK4), kept between launches and pieces; travelled[q] its
    # path's length so far (its steps in the length unit, the distances moved in
    # the time unit), for max_length (0: none), whose last step lands on it in
    # the length unit. terminal (0: none) is vtkStreamTracer's TerminalSpeed: a
    # whole step is not begun where the velocity's speed is at or below it, so a
    # path ends at its first point that slow (the seed's too).
    for q in range(xs.shape[0]):
        x = xs[q]
        t = times[q]
        n = steps[q]
        st = status[q]
        took = moved[q]
        hc = hs[q]
        gone = travelled[q]
        final = 0
        forced = 0
        ratio = h - h
        here = 0
        length = lengths[q]
        if record == 1 and first == 1:
            trace[q * capacity] = x
            trace_times[q * capacity] = t
            length = 1
        k1 = x - x
        k2 = k1
        k3 = k1
        k4 = k1
        k5 = k1
        vel = k1
        w1 = h - h
        w2 = w1
        w3 = w1
        w4 = w1
        w5 = w1
        curr = x
        out = x
        mode = _STEP
        stage = 0
        sub = hc
        r0 = h - h
        r1 = hc
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
            if mode == _STEP and stage == 0 and max_length > 0.0 and by_length == 1:
                # The last step lands on max_length.
                if gone + sub >= max_length:
                    sub = max_length - gone
                    final = 1
            if (mode == _STEP or mode == _CHECK) and method == _RK45:
                if stage == 1:
                    p = x + sub * (0.2 * k1)
                elif stage == 2:
                    p = x + sub * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2)
                elif stage == 3:
                    p = x + sub * (0.3 * k1 - 0.9 * k2 + 1.2 * k3)
                elif stage == 4:
                    p = x + sub * ((-11.0 / 54.0) * k1 + 2.5 * k2 - (70.0 / 27.0) * k3
                                   + (35.0 / 27.0) * k4)
                elif stage == 5:
                    p = x + sub * ((1631.0 / 55296.0) * k1 + (175.0 / 512.0) * k2
                                   + (575.0 / 13824.0) * k3 + (44275.0 / 110592.0) * k4
                                   + (253.0 / 4096.0) * k5)
            elif mode == _STEP or mode == _CHECK:
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
            raw = v.dot(v)            # the speed squared, before the length unit
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
                        r1 = sub
                        div = 1.0
                        final = 0
                    else:
                        r1 = sub
                        bisect = 1
                elif mode == _STEP and stage == 0 and terminal > 0.0 \
                        and raw <= terminal * terminal:
                    # Too slow to begin another step: the path ends here.
                    st = st | SLOW_SPEED | TERMINATE
                    running = 0
                elif (method == _RK4 and stage < 3) or (method == _RK45 and stage < 5):
                    if stage == 0:
                        k1 = v
                        w1 = w
                    elif stage == 1:
                        k2 = v
                        w2 = w
                    elif stage == 2:
                        k3 = v
                        w3 = w
                    elif stage == 3:
                        k4 = v
                        w4 = w
                    else:
                        k5 = v
                        w5 = w
                    stage += 1
                else:
                    vel = v
                    pace = w
                    if method == _RK4:
                        vel = (k1 + 2.0 * k2 + 2.0 * k3 + v) / 6.0
                        pace = (w1 + 2.0 * w2 + 2.0 * w3 + w) / 6.0
                    elif method == _RK45:
                        # Cash-Karp's fifth-order weights, as VTK steps.
                        vel = ((37.0 / 378.0) * k1 + (250.0 / 621.0) * k3
                               + (125.0 / 594.0) * k4 + (512.0 / 1771.0) * v)
                        pace = ((37.0 / 378.0) * w1 + (250.0 / 621.0) * w3
                                + (125.0 / 594.0) * w4 + (512.0 / 1771.0) * w)
                    else:
                        k1 = v
                    stage = 0
                    redo = 0
                    if mode == _STEP and method == _RK45 and tol > 0.0:
                        # The fifth- and fourth-order solutions' difference.
                        e = sub * (((37.0 / 378.0) - (2825.0 / 27648.0)) * k1
                                   + ((250.0 / 621.0) - (18575.0 / 48384.0)) * k3
                                   + ((125.0 / 594.0) - (13525.0 / 55296.0)) * k4
                                   - (277.0 / 14336.0) * k5
                                   + ((512.0 / 1771.0) - 0.25) * v)
                        ratio = e.norm() / tol
                        if ratio > 1.0 and forced == 0 and sub > min_step:
                            # Too large: again from the same point, smaller, its first
                            # stage kept; at min_step it is taken whatever its error.
                            nh = 0.9 * sub * pow(ratio, -0.25)
                            if nh <= min_step:
                                nh = min_step + 0.0 * nh
                                forced = 1
                            hc = nh
                            sub = nh
                            final = 0
                            stage = 1
                            redo = 1
                    if mode == _STEP and redo == 0:
                        start = x
                        x = x + sub * vel
                        if by_length == 1:
                            t += sub * pace
                            gone += sub
                        else:
                            t += sub
                            gone += (x - start).norm()
                        n += 1
                        here += 1
                        took = 1
                        if record == 1:
                            trace[q * capacity + length] = x
                            trace_times[q * capacity + length] = t
                            length += 1
                        if method == _RK45 and tol > 0.0:
                            # The next step, from this one's error.
                            nh = max_step + 0.0 * sub
                            if ratio > 0.0:
                                nh = 0.9 * sub * pow(ratio, -0.2)
                            hc = min(max(nh, min_step), max_step)
                            forced = 0
                        sub = hc
                        if vel.dot(vel) <= eps:
                            st = st | ZERO_VELOCITY | TERMINATE
                            running = 0
                        elif n == max_steps:
                            st = st | TERMINATE
                            running = 0
                        elif final == 1 or (max_length > 0.0
                                            and gone >= max_length - 0.001 * hc):
                            # At max_length, or a sliver of a step short of it (the
                            # steps' sum rounded below it).
                            st = st | TERMINATE
                            running = 0
                        elif here == budget:
                            running = 0
                    elif mode != _STEP:
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
                if by_length == 1:
                    gone += r1
                else:
                    gone += (out - x).norm()
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
                elif method == _RK45 and (out - x).norm() < min_step * (
                        1.0 if by_length == 1 else vel.norm()):
                    # Moved less than RK45's shortest step: at a wall the flow
                    # turns back from, a stage point swings out of the mesh at
                    # every trial size, or the stages' directions cancel, and the
                    # push out lands a hair (an ulp) away, still inside; every
                    # later step would do the same until the steps ran out (1920
                    # steps of about 1e-6 within 1.9e-4 on the aneurysm, on every
                    # device). As vtkStreamTracer's RK45 ends it, the particle
                    # has left.
                    st = st | SPATIAL_BOUNDS
                if vel.dot(vel) <= eps:
                    st = st | ZERO_VELOCITY | TERMINATE
                if n == max_steps or (max_length > 0.0 and gone >= max_length - 0.001 * hc):
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
                sub = hc
                div = 1.0
                stage = 0
                final = 0
            if bisect == 1:
                # The next trial step, or the push out once the bracket is small.
                if r1 - r0 > 10.0 * eps:
                    div = div * 2.0
                    sub = r0 + hc / div
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
        hs[q] = hc
        travelled[q] = gone


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


def _launches(mesh, velocity, step_size, steps, args, device, adaptive):
    """Run ``_advect`` (``args``, its arguments from ``method`` to ``handoff``;
    ``adaptive``, from ``tol`` to ``travelled``) until no particle has steps left
    or is not stopped: on the CPU in one launch, else in launches of a bounded
    number of steps."""
    import time

    xs, taken, status = args[3], args[5], args[6]
    m = int(xs.shape[0])
    moved = tack.zeros((m,), tack.i32)
    hints = tack.empty((m,), ids.at_least(tack.i32, int(mesh.ncells)))
    if device.type == "cpu" and not _CHUNKED_ON_CPU:
        _advect(mesh, velocity, step_size, steps, *args, steps, 1, moved, hints, *adaptive)
        return
    count = tack.empty((1,), tack.i32)
    budget, first = min(_FIRST_STEPS, steps), 1
    while True:
        start = time.perf_counter()
        _advect(mesh, velocity, step_size, steps, *args, budget, first, moved, hints, *adaptive)
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
        raise TypeError(f"advection takes point data, cell data or an interpolated field "
                        f"of one order per shape, not {field.space!r}")
    return by_id


def _parameters(data, field, unit, who, step_size, tolerance=None, min_step=None,
                max_step=None, max_length=None, terminal_speed=None):
    """The step parameters as numbers: ``step_size``, ``min_step`` and ``max_step`` in
    the spatial unit along the path (``unit="length"``) or the field's time unit
    (``"time"``), ``tolerance`` (a distance) and ``max_length`` in the spatial unit,
    ``terminal_speed`` in the velocity's unit; a plain number as given, ``None``
    kept."""
    def step(value, what):
        if value is None:
            return None
        if unit == "length":
            return _length(data, value, f"{who}'s {what}")
        if _has_quantity(value):
            return units.magnitude(value, _time_unit(data, field), f"{who}'s {what}")
        return value

    def length(value, what):
        return None if value is None else _length(data, value, f"{who}'s {what}")

    def speed(value):
        if value is None or not _has_quantity(value):
            return value
        from tack.dataset.partitioned import PartitionedDataSet

        if isinstance(data, PartitionedDataSet):
            every = data.world._allgather([piece.fields[field].unit for piece in data])
            velocity = units.common("the pieces' velocity", [u for part in every for u in part])
        else:
            velocity = _field(data, field).unit
        return units.magnitude(value, velocity, f"{who}'s terminal_speed")

    return (step(step_size, "step size"), length(tolerance, "tolerance"),
            step(min_step, "min_step"), step(max_step, "max_step"),
            length(max_length, "max_length"), speed(terminal_speed))


def _adaptive(integrator, step_size, tolerance, min_step, max_step):
    """``(tolerance, min_step, max_step)`` for the kernel: RK45's, by default VTK's
    vtkStreamTracer's (an error of 1e-6, steps from a fiftieth of the first to
    twice it); ``(0, 0, 0)`` for Euler and RK4, which take none."""
    if integrator not in _METHODS:
        raise ValueError(f"integrator is 'rk4', 'euler' or 'rk45', not {integrator!r}")
    if integrator != "rk45":
        given = [n for n, v in (("tolerance", tolerance), ("min_step", min_step),
                                ("max_step", max_step)) if v is not None]
        if given:
            raise ValueError(f"{', '.join(given)} {'is' if len(given) == 1 else 'are'} for "
                             f"integrator='rk45', not {integrator!r}")
        return 0.0, 0.0, 0.0
    step = float(step_size)
    tolerance = 1e-6 if tolerance is None else float(tolerance)
    min_step = step / 50.0 if min_step is None else float(min_step)
    max_step = 2.0 * step if max_step is None else float(max_step)
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, not {tolerance!r}")
    if not 0.0 < min_step <= step <= max_step:
        raise ValueError(f"RK45 needs 0 < min_step <= step_size <= max_step, not {min_step!r}, "
                         f"{step!r}, {max_step!r}")
    return tolerance, min_step, max_step


def _run(data, field, seeds, step_size, steps, integrator, locator, record, resume=None,
         handoff=False, unit="time", tolerance=None, min_step=None, max_step=None,
         max_length=None, neighbors=False, terminal_speed=None):
    """``resume``, ``(times, steps, status)`` for the seeds -- or with each one's
    current step and length so far, ``(times, steps, status, step, travelled)``
    -- goes on from there: ``steps`` counts the steps already taken, and a
    stopped particle stays. ``handoff`` stops a particle after a whole step
    ending in a ghost cell (``HANDOFF``). ``unit`` is what ``step_size``
    measures: ``"time"``, or ``"length"`` along the path. ``tolerance``,
    ``min_step`` and ``max_step`` are RK45's (``_adaptive``); ``max_length``
    stops a particle whose path has gone that far (``None``: no limit), and
    ``terminal_speed`` one where the velocity's speed is at or below it, before
    another step (``None``: none).
    ``neighbors`` walks from a particle's last cell to its face neighbours before
    searching the bins (``CellLocator.cell_neighbors``).

    Returns ``(xs, times, steps, status, (trace, trace_times), lengths, capacity,
    (step, travelled))``."""
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    tol, low, high = _adaptive(integrator, step_size, tolerance, min_step, max_step)
    if max_length is not None and not float(max_length) > 0.0:
        raise ValueError(f"max_length must be positive, not {max_length!r}")
    if terminal_speed is not None and not float(terminal_speed) >= 0.0:
        raise ValueError(f"terminal_speed must be at least 0, not {terminal_speed!r}")
    if unit not in ("time", "length"):
        raise ValueError(f"step_unit is 'time' or 'length', not {unit!r}")
    geometry = data.geometry
    if not isinstance(geometry.space, H1):
        raise TypeError("advection needs an H1 geometry: positions at its nodes")
    velocity = _velocity(data, field)
    positions = _as_vectors(geometry.values, data.dtype)
    from tack.dataset.polyhedra import PolylineTopology

    if (isinstance(data.topology, PolylineTopology)
            or cells_by_id(data, positions, strict=False) is None):
        raise NotImplementedError("advection takes unstructured, polyhedral, polygonal and 2D "
                                  "or 3D structured topologies; not polylines, nor lines")
    locator = locator or cached_locator(data)
    if not locator.serves(data):
        raise ValueError("the locator was built for another topology or geometry, or "
                         "Tack has written the positions since: make a new one")
    mesh = mesh_search(data, locator, positions, strict=False, neighbors=neighbors)
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
    hs = tack.empty((m,), dtype)
    travelled = tack.empty((m,), dtype)
    capacity = int(steps) + 1 if record else 1
    trace = tack.empty((m * capacity,), tack.vec(3, dtype))
    trace_times = tack.empty((m * capacity,), dtype)
    if m:
        tack.copyto(xs, seeds.to_device(xs.device) if stored else host)
        hs.fill(float(step_size))
        travelled.fill(0)
        if resume is None:
            times.fill(0)
            taken.fill(0)
            status.fill(int(SUCCESS))
        else:
            for into, values, kind in zip((times, taken, status, hs, travelled), resume,
                                          (dtype, tack.i32, tack.i32, dtype, dtype)):
                tack.copyto(into, values.to_device(into.device).astype(kind)
                            if isinstance(values, Array)
                            else np.asarray(values, kind.numpy_dtype))
        marks = data.ghosts.cells if handoff and data.ghosts is not None else None
        ghosts = marks if marks is not None else tack.zeros((1,), tack.u8)
        eps = float(np.finfo(dtype.numpy_dtype).eps)
        lengths.fill(0)
        _launches(mesh, velocity, float(step_size), int(steps),
                  (_METHODS[integrator], 1 if unit == "length" else 0, eps, xs, times,
                   taken, status, trace, trace_times, lengths, capacity, 1 if record else 0,
                   ghosts, 1 if marks is not None else 0), xs.device,
                  (tol, low, high, 0.0 if max_length is None else float(max_length),
                   0.0 if terminal_speed is None else float(terminal_speed), hs, travelled))
    return xs, times, taken, status, (trace, trace_times), lengths, capacity, (hs, travelled)

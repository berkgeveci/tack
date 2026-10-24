"""The advection kernel ``advect`` and ``streamlines`` share, by Viskores' rules:
RK4 or Euler steps; a step leaving the mesh bisected to the boundary and the
particle pushed out; a stop after its steps, outside, or at zero velocity.
Each particle's whole path is one thread of one launch, its stages located
from the particle's side by ``by_id``'s search (over ``CellLocator.cell_bins``,
ranked as the locator ranks cells), its velocity read by cell id there, as
``CellLocator.find`` and ``probe`` do."""

import enum

import numpy as np

import tack
from tack import arrays
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
def _advect(mesh, velocity, h, max_steps, rk4, eps, xs, times, steps, status, trace,
            trace_times, lengths, capacity, record):
    for q in range(xs.shape[0]):
        x = xs[q]
        t = times[q]
        n = steps[q]
        st = status[q]
        took = 0
        length = 0
        if record == 1:
            trace[q * capacity] = x
            trace_times[q * capacity] = t
            length = 1
        k1 = x - x
        k2 = k1
        k3 = k1
        vel = k1
        curr = x
        out = x
        mode = _STEP
        stage = 0
        sub = h
        r0 = h - h
        r1 = h
        div = 1.0
        running = 1
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
            # ahead of coming within the tolerance, then the smaller id -- and
            # the velocity there (by_id).
            best, best_type, best_pc, best_tet = mesh.find(p)
            found = 1 if best < mesh.ncells else 0
            v = p - p
            if found == 1:
                v = velocity.value(mesh, best, best_type, best_pc, best_tet)

            # What it means for the particle.
            bisect = 0
            if mode == _STEP or mode == _CHECK:
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
                    elif stage == 1:
                        k2 = v
                    else:
                        k3 = v
                    stage += 1
                else:
                    vel = v
                    if rk4 == 1:
                        vel = (k1 + 2.0 * k2 + 2.0 * k3 + v) / 6.0
                    else:
                        k1 = v
                    stage = 0
                    if mode == _STEP:
                        x = x + h * vel
                        t += h
                        n += 1
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
                t += r1
                mode = _OUT
            else:
                # Where the push out ends: the step is taken, in or out.
                x = out
                n += 1
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
                if (st & (SPATIAL_BOUNDS | ZERO_VELOCITY | TERMINATE)) != 0:
                    running = 0
                else:
                    mode = _STEP
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


def _run(data, field, seeds, step_size, steps, integrator, locator, record):
    if int(steps) < 1:
        raise ValueError(f"steps must be at least 1, not {steps!r}")
    if integrator not in ("rk4", "euler"):
        raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
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
    stored = isinstance(seeds, Array) and seeds._vector_n == 3
    host = None if stored else np.ascontiguousarray(
        np.asarray(seeds.numpy() if hasattr(seeds, "numpy") else seeds,
                   dtype.numpy_dtype).reshape(-1, 3))
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
        times.fill(0)
        taken.fill(0)
        status.fill(int(SUCCESS))
        eps = float(np.finfo(dtype.numpy_dtype).eps)
        _advect(mesh, velocity, float(step_size), int(steps),
                1 if integrator == "rk4" else 0, eps, xs, times, taken, status, trace,
                trace_times, lengths, capacity, 1 if record else 0)
    return xs, times, taken, status, (trace, trace_times), lengths, capacity

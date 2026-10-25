"""Temporal filters: steps streamed through filters one at a time.

Tack holds no series of steps. A simulation in situ, or a reader's loop, hands
each step to the filters as a dataset with its time (``DataSet.time``, in
``DataSet.time_unit``); stateless filters treat it as any dataset, and the
temporal filters here are objects fed one step at a time (``update(data)``,
which returns that step's output), keeping what they need of earlier steps.

What they keep is a copy by default: in situ, arrays often come without a copy
(DLPack) and the simulation overwrites them at the next step. A filter copies
only what it reads -- a velocity field, positions where the mesh moves -- and
``copy=False`` keeps references where the caller knows its buffers are not
reused. Implicit arrays (an image's coordinates, a constant field) are kept as
they are: nothing writes them.

A ``PartitionedDataSet`` streams piece by piece: each piece keeps its own
state, so every step must bring the same pieces.

- ``RunningStatistics``: per point or cell, the mean, variance, minimum and
  maximum of fields over the steps so far.
- ``TimeDerivative``: each field's rate of change since the previous step.
- ``Advection``: particles carried by a velocity field through the steps, new
  ones injected at any time; ``extend_paths`` joins its particles, step after
  step, into paths; ``Pathlines`` and ``Streaklines`` are built on them.
"""

import math

import numpy as np

import tack
from tack import arrays
from tack.dataset import units
from tack.dataset.dataset import DataSet, Field
from tack.dataset.partitioned import PartitionedDataSet

__all__ = ["Advection", "Pathlines", "RunningStatistics", "Streaklines", "TimeDerivative",
           "extend_paths"]


def _elapsed(earlier, later, unit=None):
    """``later``'s time less ``earlier``'s, in ``unit`` (by default ``later``'s
    time unit): each a dataset or a ``_Step``. Both must have a time; their
    units are converted where both are known, and two steps of which one has a
    unit and the other none are refused."""
    for step, which in ((earlier, "the earlier step"), (later, "the later step")):
        if step.time is None:
            raise ValueError(f"{which} has no time: give each step one (DataSet.time)")
    if (earlier.time_unit is None) != (later.time_unit is None):
        raise ValueError(f"one step's time is in {earlier.time_unit or later.time_unit!r} "
                         "and the other's unknown: give both a time unit, or neither")
    t0, t1 = earlier.time, later.time
    if earlier.time_unit is not None and earlier.time_unit != later.time_unit:
        factor, offset = units.conversion(earlier.time_unit, later.time_unit)
        t0 = t0 * factor + offset
    dt = t1 - t0
    target = unit if unit is not None else later.time_unit
    if target is not None and later.time_unit is not None and target != later.time_unit:
        factor, _ = units.conversion(later.time_unit, units.time_unit(target))
        dt *= factor
    elif target is not None and later.time_unit is None:
        raise ValueError("the steps' times have no unit to convert from")
    if not math.isfinite(dt):
        raise ValueError("the time between the steps is not finite")
    return dt


def _kept(values, copy):
    """``values`` as a filter keeps them: a copy of a stored array (unless not
    ``copy``), an implicit array as it is."""
    if not copy or isinstance(values, arrays._Implicit):
        return values
    out = tack.empty_like(values)
    tack.copyto(out, values)
    return out


class _Step:
    """What a temporal filter keeps of one step: its time and time unit, its
    topology (by reference, to tell a step of another mesh), and of ``fields``
    (names) and the geometry, if ``positions``, the values -- copies, unless
    ``copy`` is false. Each kept field keeps its space and unit."""

    def __init__(self, data, fields=(), positions=False, copy=True):
        self.time = data.time
        self.time_unit = data.time_unit
        self.topology = data.topology
        self.origin = data.origin
        self.unit = data.unit
        self.fields = {}
        for name in fields:
            if name not in data.fields:
                raise KeyError(f"no field {name!r}: the step has "
                               f"{', '.join(map(repr, data.fields)) or 'none'}")
            field = data.fields[name]
            self.fields[name] = Field(field.space, _kept(field.values, copy), field.unit)
        geometry = data.geometry
        self.geometry = Field(geometry.space, _kept(geometry.values, copy),
                              geometry.unit) if positions else None

    def same_mesh(self, data):
        """Whether ``data`` is on this step's mesh: the same topology object."""
        return data.topology is self.topology


# -- the same mesh ---------------------------------------------------------------

@tack.kernel
def _differences(a, b, out):
    for i in range(out.shape[0]):
        out[i] = 1.0 if a[i] != b[i] else 0.0


def _equal(a, b):
    """Whether two id arrays (stored or implicit) hold the same values, compared on
    their device."""
    if a is b:
        return True
    if arrays.size_of(a) != arrays.size_of(b):
        return False
    n = arrays.size_of(a)
    if not n:
        return True
    device = a.device if isinstance(a, tack.Array) else b.device
    flags = tack.empty((n,), tack.f32, device=device)
    _differences(a, b, flags)
    return flags.max() == 0.0


def _same_mesh(a, b):
    """Whether topologies ``a`` and ``b`` are one mesh: the same object, or of one
    kind with the same sizes and arrays. In situ a simulation may hand each step
    a new topology over the same mesh."""
    if a is b:
        return True
    if type(a) is not type(b) or a.num_points != b.num_points or a.num_cells != b.num_cells:
        return False
    if hasattr(a, "point_dims"):
        return a.point_dims == b.point_dims and a.offset == b.offset
    return all(_equal(getattr(a, name), getattr(b, name)) for name in a._STORED)


def _refuse_another_mesh(what, kept, data):
    if not _same_mesh(kept, data.topology):
        raise ValueError(f"{what} needs one mesh over the steps, and this step is on "
                         "another: remesh it onto the first step's mesh first")


# -- streaming ------------------------------------------------------------------

class _Streamed:
    """A temporal filter over a dataset, or piece by piece over a partitioned
    dataset (each piece its own state)."""

    def __init__(self, fields):
        self.fields = [fields] if isinstance(fields, str) else list(fields)
        if not self.fields:
            raise ValueError("name at least one field")
        self._states = None

    def update(self, data):
        """This step's output (``None`` where the filter has none yet)."""
        if isinstance(data, PartitionedDataSet):
            pieces = list(data)
            if self._states is None:
                self._states = [self._new_state() for _ in pieces]
            if len(pieces) != len(self._states):
                raise ValueError(f"the steps bring {len(self._states)} pieces, then "
                                 f"{len(pieces)}: each piece keeps its own state")
            outs = [self._on_device(piece, state) for piece, state in zip(pieces, self._states)]
            return None if any(o is None for o in outs) else \
                PartitionedDataSet(outs, world=data.world)
        if self._states is None:
            self._states = [self._new_state()]
        if len(self._states) != 1:
            raise ValueError("the steps brought pieces before: keep streaming pieces")
        return self._on_device(data, self._states[0])

    def _on_device(self, data, state):
        # What a step's update allocates, it allocates on the step's device.
        with data.device:
            return self._update(data, state)

    def _fields_of(self, data):
        for name in self.fields:
            if name not in data.fields:
                raise KeyError(f"no field {name!r}: the step has "
                               f"{', '.join(map(repr, data.fields)) or 'none'}")
        return [data.fields[name] for name in self.fields]


def _accumulator_type(data, values):
    """Sums run in f64 where the device has it, else in f32."""
    if tack.f64 in data.device._backend.supported_dtypes:
        return tack.f64
    return tack.f32


def _like(values, dtype):
    width = arrays.width_of(values)
    return tack.empty((arrays.size_of(values),), tack.vec(width, dtype) if width else dtype)


@tack.kernel
def _start(x, mean, m2, lo, hi):
    for i in range(mean.shape[0]):
        mean[i] = x[i]
        m2[i] = x[i] * 0.0
        lo[i] = x[i]
        hi[i] = x[i]


@tack.kernel
def _welford(x, n, mean, m2, lo, hi):
    # Welford's update, the step's value the n-th.
    for i in range(mean.shape[0]):
        v = x[i]
        delta = v - mean[i]
        mean[i] = mean[i] + delta / n
        m2[i] = m2[i] + delta * (v - mean[i])
        lo[i] = min(lo[i], v)
        hi[i] = max(hi[i], v)


@tack.kernel
def _scaled_copy(src, out, factor):
    for i in range(out.shape[0]):
        out[i] = src[i] * factor


class _Running:
    def __init__(self):
        self.topology = None
        self.count = 0
        self.sums = {}


class RunningStatistics(_Streamed):
    """Per point or cell, statistics of ``fields`` (a name or names) over the steps
    so far: each ``update(data)`` adds the step's values and returns the step with
    ``<name>_mean``, ``<name>_variance`` (the population variance, ``M2 / n``),
    ``<name>_min`` and ``<name>_max`` beside its fields, computed by Welford's
    method on the device (in f64 where the device has it). Vector fields are taken
    component by component.

    The steps must be on one mesh -- the same topology, or one of the same kind,
    sizes and arrays -- or the update is refused: remesh onto the first step's mesh
    first. Nothing of earlier steps is kept but the sums, so nothing is copied.
    The mean, minimum and maximum are in the field's unit, the variance in its
    square (kelvin squared for degrees Celsius). ``count`` is the steps added."""

    def _new_state(self):
        return _Running()

    @property
    def count(self):
        return self._states[0].count if self._states else 0

    def _update(self, data, state):
        fields = self._fields_of(data)
        if state.topology is None:
            state.topology = data.topology
        else:
            _refuse_another_mesh("RunningStatistics", state.topology, data)
        state.count += 1
        out = dict(data.fields)
        for name, field in zip(self.fields, fields):
            values = arrays.materialize(field.values)
            if arrays.dtype_of(values).numpy_dtype.kind not in "fiu":
                raise TypeError(f"field {name!r} is not numeric")
            if name not in state.sums:
                dtype = _accumulator_type(data, values)
                state.sums[name] = [_like(values, dtype) for _ in range(4)]
                if arrays.size_of(values):
                    _start(values, *state.sums[name])
            elif arrays.size_of(values):
                _welford(values, state.count, *state.sums[name])
            mean, m2, lo, hi = state.sums[name]
            n = arrays.size_of(values)
            copies = []
            for source, factor in ((mean, 1.0), (m2, 1.0 / state.count), (lo, 1.0), (hi, 1.0)):
                copy = _like(source, arrays.dtype_of(source))
                if n:
                    _scaled_copy(source, copy, factor)
                copies.append(copy)
            unit = field.unit
            square = units.derived((units.delta(unit), 2))
            for suffix, values_out, u in zip(("mean", "variance", "min", "max"), copies,
                                             (unit, square, unit, unit)):
                out[f"{name}_{suffix}"] = Field(field.space, values_out, u)
        from tack.dataset.ghosts import keep_marks

        return keep_marks(data, DataSet(data.topology, data.geometry, fields=out,
                                        sets=dict(data.sets)))


@tack.kernel
def _rate(x, previous, inverse, out):
    for i in range(out.shape[0]):
        out[i] = (x[i] - previous[i]) * inverse


class _Previous:
    def __init__(self):
        self.step = None


class TimeDerivative(_Streamed):
    """Each of ``fields``' rate of change since the previous step: ``update(data)``
    returns the step with ``<name>_dt``, ``(value - previous) / (time - previous
    time)`` per point or cell (a backward difference, in f64 where the device has
    it), in the field's unit over the time unit (a field in degrees Celsius gives
    kelvin per second). The first step has no previous one: its update returns
    ``None``.

    The previous step's values are kept as copies (``copy=False`` keeps
    references, where the caller's buffers are not overwritten between steps).
    The steps must be on one mesh and in increasing time, with times in units
    that convert (``_elapsed``)."""

    def __init__(self, fields, copy=True):
        super().__init__(fields)
        self.copy = copy

    def _new_state(self):
        return _Previous()

    def _update(self, data, state):
        fields = self._fields_of(data)
        previous = state.step
        state.step = _Step(data, self.fields, copy=self.copy)
        if previous is None:
            return None
        _refuse_another_mesh("TimeDerivative", previous.topology, data)
        dt = _elapsed(previous, data)
        if not dt > 0:
            raise ValueError(f"the steps' times do not increase: {previous.time} then "
                             f"{data.time}")
        out = dict(data.fields)
        for name, field in zip(self.fields, fields):
            values = arrays.materialize(field.values)
            before = arrays.materialize(previous.fields[name].values)
            rate = _like(values, _accumulator_type(data, values))
            if arrays.size_of(values):
                _rate(values, before, 1.0 / dt, rate)
            unit = units.derived(units.delta(field.unit), (data.time_unit, -1))
            out[f"{name}_dt"] = Field(field.space, rate, unit)
        from tack.dataset.ghosts import keep_marks

        return keep_marks(data, DataSet(data.topology, data.geometry, fields=out,
                                        sets=dict(data.sets)))


# -- advection ------------------------------------------------------------------

def _kept_dataset(data, field, copy):
    """What advection keeps of a step: a dataset of the step's topology (by
    reference) with copies of its positions and velocity (``_Step``), at its
    origin, spatial unit and time; piece by piece for pieces."""
    if isinstance(data, PartitionedDataSet):
        return PartitionedDataSet([_kept_dataset(piece, field, copy) for piece in data],
                                  world=data.world)
    step = _Step(data, [field], positions=True, copy=copy)
    return DataSet(step.topology, step.geometry, fields=dict(step.fields), **data._frame())


def _frame_of(data):
    """A dataset's, or the first piece's, ``_frame()`` and device."""
    first = next(iter(data)) if isinstance(data, PartitionedDataSet) else data
    return first._frame(), first.device


def _velocities(data, field, x):
    """The velocity at world points ``x`` (``(n, 3)`` f64) and whether a cell holds
    each: ``probe``, over pieces routed to the rank whose cells hold them."""
    from tack.dataset.algorithms.probe import probe

    if not len(x):
        return np.zeros((0, 3)), np.zeros(0, bool)
    out = probe(data, x, fields=[field])
    values = np.asarray(out.fields[field].values.numpy(), np.float64).reshape(len(x), -1)
    return values, out.fields["valid"].values.numpy() == 1


class Advection:
    """Particles carried by ``field`` (a velocity field's name) through the steps:
    each ``update(data)`` moves every particle from its time to the step's, with
    the velocity linear in time between the previous step (kept, as a copy unless
    ``copy`` is false) and this one, and returns the particles at their current
    points -- a point cloud at the step's time with ``seed`` (each particle's
    number), ``source`` (the seed point it came from, or -1), ``release`` (its
    release time), ``time`` and ``status`` (``ParticleStatus`` bits, as
    ``advect``'s).

    ``seeds`` (an ``(n, 3)`` array of world positions) are released at the first
    step's time; ``inject(points, time=None)`` adds particles at any later time
    (by default the last step's). The velocity is the probe of each step's own
    mesh, so the mesh may change between steps, and over a ``PartitionedDataSet``
    each point is probed on the rank whose cells hold it: the particles stay on
    the rank that released them. ``integrator`` is ``"rk4"`` or ``"euler"``;
    ``step_size`` (in the steps' time unit) bounds each substep, by default one
    substep from step to step. A particle leaving the mesh stops where it last
    was, marked ``SPATIAL_BOUNDS``; one released outside has ``SUCCESS``
    cleared. Times convert between the steps' unit and the velocity's (a
    velocity in m/s over steps in minutes)."""

    def __init__(self, field, seeds=None, integrator="rk4", step_size=None, copy=True):
        if integrator not in ("rk4", "euler"):
            raise ValueError(f"integrator is 'rk4' or 'euler', not {integrator!r}")
        if step_size is not None and not step_size > 0:
            raise ValueError(f"step_size must be positive, not {step_size!r}")
        self.field, self.integrator = field, integrator
        self.step_size, self.copy = step_size, copy
        self.x = np.zeros((0, 3))
        self.time = np.zeros(0)
        self.release = np.zeros(0)
        self.seed = np.zeros(0, np.int64)
        self.source = np.zeros(0, np.int64)
        self.status = np.zeros(0, np.int32)
        self._pending = []                       # (points, time or None, sources)
        self._previous = None                    # the kept previous step
        self._last = None                        # its time, in the stream's unit
        self._unit = None                        # the stream's time unit
        self._released = 0
        if seeds is not None:
            points = np.asarray(seeds, np.float64).reshape(-1, 3)
            self.inject(points, None, np.arange(len(points)))

    def inject(self, points, time=None, sources=None):
        """Release particles at world ``points`` at ``time`` (in the steps' time
        unit; by default the last step's, or the first step's before any); the next
        update carries each from its release time. ``sources`` numbers the seed
        point each came from (``-1`` by default). A time before the last step is
        refused: that interval has been integrated."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        if time is not None and self._last is not None and time < self._last:
            raise ValueError(f"particles released at {time}, before the last step's time "
                             f"{self._last}: that interval is integrated")
        sources = np.full(len(points), -1, np.int64) if sources is None else \
            np.asarray(sources, np.int64).reshape(-1)
        if len(sources) != len(points):
            raise ValueError("one source per point")
        self._pending.append((points, None if time is None else float(time), sources))

    def _stream_time(self, data):
        """``data``'s time in the stream's time unit (the first step's)."""
        frame, _ = _frame_of(data)
        if frame["time"] is None:
            raise ValueError("advection needs each step's time (DataSet.time)")
        if self._last is None:
            self._unit = frame["time_unit"]
            return frame["time"]
        unit = frame["time_unit"]
        if (unit is None) != (self._unit is None):
            raise ValueError("the steps' times change from a unit to none, or back")
        if unit is not None and unit != self._unit:
            factor, offset = units.conversion(unit, self._unit)
            return frame["time"] * factor + offset
        return frame["time"]

    def _velocity_scale(self, data):
        """Steps' time unit to the velocity's: the factor to multiply a time by."""
        from tack.dataset.algorithms._common import _time_unit

        if self._unit is None:
            return 1.0
        velocity_time = _time_unit(data, self.field)
        if velocity_time is None:
            return 1.0
        return units.conversion(self._unit, velocity_time)[0]

    def _release(self, data, t1):
        """Particles pending with a release time at or before ``t1``."""
        keep = []
        for points, time, sources in self._pending:
            time = (self._last if self._last is not None else t1) if time is None else time
            if time > t1:
                keep.append((points, time, sources))
                continue
            if self._last is None and time < t1:
                raise ValueError(f"particles released at {time}, before the first step's "
                                 f"time {t1}")
            n = len(points)
            ids = self._released + np.arange(n, dtype=np.int64)
            if isinstance(data, PartitionedDataSet):
                ids += np.int64(data.world.rank) << np.int64(40)
            self._released += n
            # Released outside any cell: SUCCESS cleared, as advect's seeds.
            where = self._previous if self._previous is not None and time < t1 else data
            _, inside = _velocities(where, self.field, points)
            status = np.where(inside, 1, 0).astype(np.int32)
            for name, values in (("x", points), ("time", np.full(n, time)),
                                 ("release", np.full(n, time)), ("seed", ids),
                                 ("source", sources), ("status", status)):
                setattr(self, name, np.concatenate([getattr(self, name), values]))
        self._pending = keep

    def update(self, data):
        """Carry the particles to ``data``'s time and return them (see the class)."""
        t1 = self._stream_time(data)
        if self._last is not None and not t1 > self._last:
            raise ValueError(f"the steps' times do not increase: {self._last} then {t1}")
        self._release(data, t1)
        if self._previous is not None:
            self._advance(self._previous, data, self._last, t1)
        self._previous = _kept_dataset(data, self.field, self.copy)
        self._last = t1
        return self._particles(data)

    def _advance(self, before, now, t0, t1):
        scale = self._velocity_scale(now)
        span = t1 - t0

        def velocity(x, t):
            a = np.clip((t - t0) / span, 0.0, 1.0)[:, None]
            v0, in0 = _velocities(before, self.field, x)
            v1, in1 = _velocities(now, self.field, x)
            return (1.0 - a) * v0 + a * v1, in0 & in1

        limit = self.step_size if self.step_size is not None else span
        while True:
            moving = np.flatnonzero(((self.status & 1) != 0) & ((self.status & 2) == 0)
                                    & (self.time < t1 - 1e-12 * max(abs(t1), 1.0)))
            if not len(moving):
                break
            x, t = self.x[moving], self.time[moving]
            h = np.minimum(limit, t1 - t)
            k1, ok = velocity(x, t)
            if self.integrator == "euler":
                step = k1
            else:
                k2, ok2 = velocity(x + 0.5 * (h * scale)[:, None] * k1, t + 0.5 * h)
                k3, ok3 = velocity(x + 0.5 * (h * scale)[:, None] * k2, t + 0.5 * h)
                k4, ok4 = velocity(x + (h * scale)[:, None] * k3, t + h)
                ok = ok & ok2 & ok3 & ok4
                step = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            end = x + (h * scale)[:, None] * step
            _, inside = velocity(end, t + h)
            ok &= inside
            # A particle whose step leaves the mesh stops where it was.
            left = moving[~ok]
            self.status[left] |= 2 | 4
            went = moving[ok]
            self.x[went] = end[ok]
            self.time[went] = (t + h)[ok]
            self.status[went] |= 16

    def _particles(self, data):
        from tack.dataset.dataset import point_cloud

        frame, device = _frame_of(data)
        fields = {"seed": self.seed, "source": self.source, "release": self.release,
                  "time": self.time, "status": self.status}
        time_units = {"release": self._unit, "time": self._unit} if self._unit else {}
        cloud = point_cloud(self.x, fields=fields, origin=frame["origin"], device=device,
                            unit=frame["unit"], units=time_units)
        cloud.time, cloud.time_unit = frame["time"], frame["time_unit"]
        if isinstance(data, PartitionedDataSet):
            return PartitionedDataSet([cloud], world=data.world)
        return cloud


def _rows(particles):
    """A particle cloud's (or a piece list's) world positions and fields, host."""
    pieces = list(particles) if isinstance(particles, PartitionedDataSet) else [particles]
    x = np.concatenate([p.world_positions() for p in pieces]) if pieces else np.zeros((0, 3))
    get = lambda name: np.concatenate([np.asarray(p.fields[name].values.numpy())
                                       for p in pieces]) if pieces else np.zeros(0)
    return x, {name: get(name) for name in ("seed", "time", "status")}, pieces


def _lines(groups, x, frame, like, point_fields):
    """A dataset of the world points ``x`` (sorted by group) and a polyline through
    each group of two or more; a group of one point keeps its point, in no line."""
    from tack.dataset.polyhedra import PolylineTopology
    from tack.dataset.spaces import H1, Values

    _, starts, counts = np.unique(groups, return_index=True, return_counts=True)
    lined = counts >= 2
    offsets = np.concatenate([[0], np.cumsum(counts[lined])]).astype(np.int64)
    members = np.concatenate([np.arange(a, a + c) for a, c in
                              zip(starts[lined], counts[lined])]) if lined.any() else \
        np.zeros(0, np.int64)
    with like.device:
        topology = PolylineTopology(offsets, members.astype(np.int64), num_points=len(x))
        origin = np.asarray(frame["origin"])
        out = DataSet(topology, (x - origin).astype(like.dtype.numpy_dtype), dtype=like.dtype,
                      **frame)
        for name, (values, unit) in point_fields.items():
            kind = values.dtype.kind
            stored = values.astype(out.dtype.numpy_dtype if kind == "f" else np.int64)
            out.fields[name] = Field(H1(out) if kind == "f" else Values(out, "points"),
                                     tack.array(stored), unit)
    return out


def extend_paths(paths, particles):
    """``paths`` (``None``, or what this returned before) with ``particles`` -- an
    ``Advection``'s output -- appended: each particle's points in time (``seed``
    and ``time`` on the points), and a polyline through those of each particle
    with two or more. Each step adds the particles that moved since (a
    stopped one adds nothing). It stands alone, so particles written step by
    step can be joined into paths afterwards."""
    x, rows, pieces = _rows(particles)
    seed, time = rows["seed"].astype(np.int64), rows["time"].astype(np.float64)
    if paths is not None:
        old_seed = paths.fields["seed"].values.numpy().astype(np.int64)
        old_time = paths.fields["time"].values.numpy().astype(np.float64)
        # A particle's point is new when it is later than its path's last.
        order = np.lexsort((old_time, old_seed))
        last_of = dict(zip(old_seed[order].tolist(), old_time[order].tolist()))
        new = np.array([t_ > last_of.get(s_, -np.inf)
                        for s_, t_ in zip(seed.tolist(), time.tolist())], bool)
        x = np.concatenate([paths.world_positions(), x[new]])
        seed = np.concatenate([old_seed, seed[new]])
        time = np.concatenate([old_time, time[new]])
    order = np.lexsort((time, seed))
    like = pieces[0]
    unit = like.fields["time"].unit
    return _lines(seed[order], x[order], like._frame(), like,
                  {"seed": (seed[order], None), "time": (time[order], unit)})


class Pathlines:
    """Each particle's path through the steps: ``Advection`` (its arguments) and
    ``extend_paths``, each ``update(data)`` returning the current pathlines -- a
    polyline per particle (``seed`` and ``time`` on the points), a particle's
    first point alone until it has moved.
    ``inject`` releases particles as ``Advection.inject`` does."""

    def __init__(self, field, seeds=None, integrator="rk4", step_size=None, copy=True):
        self.advection = Advection(field, seeds, integrator, step_size, copy)
        self.paths = None

    def inject(self, points, time=None):
        self.advection.inject(points, time)

    def update(self, data):
        self.paths = extend_paths(self.paths, self.advection.update(data))
        return self.paths


class Streaklines:
    """The lines of particles released at ``seeds`` (world positions) through the
    steps: every update releases ``releases`` particles at each seed, at times
    spread evenly over the interval since the previous step (one at each step's
    time by default), carries them all (``Advection``), and returns a polyline per
    seed (``source`` on the cells) through the particles it released still in the
    mesh, newest (at the seed) first, with ``source``, ``release`` and ``time`` on
    the points; a seed with one particle so far keeps its point, in no line."""

    def __init__(self, field, seeds, releases=1, integrator="rk4", step_size=None,
                 copy=True):
        if int(releases) < 1:
            raise ValueError("releases is at least 1")
        self.seeds = np.asarray(seeds, np.float64).reshape(-1, 3)
        self.releases = int(releases)
        self.advection = Advection(field, None, integrator, step_size, copy)

    def update(self, data):
        adv = self.advection
        t1 = adv._stream_time(data)
        sources = np.arange(len(self.seeds))
        if adv._last is None:
            adv.inject(self.seeds, t1, sources)
        else:
            for j in range(1, self.releases + 1):
                adv.inject(self.seeds, adv._last + (t1 - adv._last) * j / self.releases,
                           sources)
        particles = adv.update(data)
        x, _, pieces = _rows(particles)
        alive = ((adv.status & 1) != 0) & ((adv.status & 4) == 0) & (adv.source >= 0)
        x, source = x[alive], adv.source[alive]
        release, time = adv.release[alive], adv.time[alive]
        order = np.lexsort((-release, source))          # newest, at the seed, first
        like = pieces[0]
        unit = like.fields["time"].unit
        return _lines(source[order], x[order], like._frame(), like,
                      {"source": (source[order], None), "release": (release[order], unit),
                       "time": (time[order], unit)})


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
"""

import math

import tack
from tack import arrays
from tack.dataset import units
from tack.dataset.dataset import DataSet, Field
from tack.dataset.partitioned import PartitionedDataSet

__all__ = ["RunningStatistics", "TimeDerivative"]


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


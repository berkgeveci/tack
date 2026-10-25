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

This module holds the parts the filters share (``_Step``, ``_elapsed``); the
filters themselves follow (running statistics, time derivatives, advection,
pathlines).
"""

import math

import tack
from tack import arrays
from tack.dataset import units
from tack.dataset.dataset import Field

__all__ = []


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

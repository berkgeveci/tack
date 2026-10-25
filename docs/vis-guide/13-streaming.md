# Streaming Steps

A simulation run in situ hands its analysis one step at a time. Tack keeps no
series of steps: each step is a dataset at its **time** (`data.time`, in
`data.time_unit`), every filter keeps that time as it keeps the origin, and
stateless filters treat each step like any dataset. Filters that need more
than the current step are small objects fed one step at a time: `update(data)`
returns that step's output, and the object keeps what it needs of earlier
steps.

<!-- tack-check: streaming -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

stats = td.RunningStatistics("T")       # per point, over the steps so far
rate = td.TimeDerivative("T")           # since the previous step

for step in range(5):
    # A simulation's step: its mesh, a temperature field and the time.
    data = td.uniform_grid((8, 8, 8), spacing=(0.1, 0.1, 0.1), unit="m")
    data.time, data.time_unit = 0.5 * step, "s"
    x = data.positions()
    temperature = 20.0 + 5.0 * np.sin(x[:, 0] * 3.0 + 0.4 * step)
    data.fields["T"] = td.Field(td.H1(data), tack.array(temperature.astype(np.float32)),
                                unit="degC")

    summary = stats.update(data)        # the step, with T_mean, T_variance, T_min, T_max
    change = rate.update(data)          # the step with T_dt; None on the first step
    hot = td.threshold(summary, "T_max", 24.0, 30.0)      # filters as on any dataset

print(stats.count, "steps;", summary.fields["T_variance"].unit, change.fields["T_dt"].unit)
assert summary.time == 2.0 and change.fields["T_dt"].unit == "K/s"
assert np.all(summary.fields["T_min"].values.numpy() <= summary.fields["T_mean"].values.numpy())
```

## Time on a dataset

`DataSet(..., time=, time_unit=)` (or setting `data.time`, `data.time_unit`):
a double on the host or `None`, in a unit of time or `None`. Every filter keeps
both; `probe` gives back its target, with the target's time. The pieces of a
`PartitionedDataSet` and the boxes of an `AMRDataSet` are at one time
(compared in one unit, so 1 s and 1000 ms agree); an octree's is its roots'.
The VTK interop maps VTK's `DATA_TIME_STEP` both ways; VTK has no time unit.

## Running statistics

`RunningStatistics(fields)` adds each step's values, per point or cell, and
returns the step with `<name>_mean`, `<name>_variance` (the population
variance), `<name>_min` and `<name>_max` beside its fields, computed on the
device by Welford's method, in f64 where the device has it. Vector fields are
taken component by component. It keeps only its sums, so nothing of earlier
steps is copied. The mean and extremes are in the field's unit, the variance
in its square (kelvin squared for degrees Celsius).

The steps must be on one mesh: the same topology, or one of the same kind,
sizes and arrays, as a simulation that builds a new dataset each step over the
same mesh gives. A step on another mesh is refused: remesh it onto the first
step's mesh first (`probe` onto it, say).

## Time derivatives

`TimeDerivative(fields)` returns the step with `<name>_dt`, the backward
difference `(value - previous) / (time - previous time)` per point or cell,
in the field's unit over the time unit (`K/s` for a field in degrees Celsius).
The first step has no previous one, and its update returns `None`. The steps
must be on one mesh and in increasing time; times in two units (seconds, then
minutes) are converted.

## What a filter keeps

A filter that needs an earlier step's values keeps a **copy** of them: in
situ, arrays often come without a copy (DLPack) and the simulation overwrites
them at the next step, so a reference would see the new values. `copy=False`
keeps references where the caller knows its buffers are not reused between
steps. Each filter keeps only what it reads -- a field, positions where the
mesh moves -- and an image's coordinates or a constant field, which nothing
writes, are kept as they are.

## Pieces

A `PartitionedDataSet` streams piece by piece: each piece keeps its own state,
so every step must bring the same pieces, and the output is pieces too. Over
MPI ranks each rank feeds its own pieces; the statistics and derivatives need
nothing from other ranks.

Advection of particles through the steps, with new particles injected at any
time, and pathlines built from it, are the next phase.

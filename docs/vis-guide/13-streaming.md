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

The [model](12-units.md#time-on-a-dataset) defines `time` and `time_unit`,
including timestamps on pieces and hierarchies. In a streaming loop, feed
one step at a time in increasing physical time; an individual filter may
also require an unchanged mesh. The sections below state those requirements.

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

## Advection, pathlines and streaklines

`Advection(field, seeds)` carries particles through the steps: each update
moves every particle from its time to the step's, the velocity linear in time
between the previous step (kept as a copy) and this one, and returns the
particles at their current points -- a point cloud at the step's time with
`seed` (each particle's number), `source`, `release`, `time` and `status`
(`ParticleStatus` bits, as `advect`'s). `inject(points, time=)` releases more
particles at any time from the last step's on; each starts at its own release
time, and one released after the coming step waits for a step that reaches it.

<!-- tack-check: pathlines -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")


def step(t):
    """A swirl that speeds up: a simulation's step at time t (seconds)."""
    data = td.uniform_grid((21, 21, 3), spacing=(0.5, 0.5, 0.5), unit="m")
    data.time, data.time_unit = t, "s"
    x = data.positions().astype(np.float64) - [5.0, 5.0, 0.0]
    v = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x))], axis=1) * (0.2 + 0.1 * t)
    data.fields["v"] = td.Field(td.H1(data), tack.array(v.astype(np.float32), tack.vec3()),
                                unit="m/s")
    return data


seeds = np.array([[7.0, 5.0, 0.5], [5.0, 8.0, 0.5]])
pathlines = td.Pathlines("v", seeds, step_size=0.1)
streaklines = td.Streaklines("v", seeds[:1], releases=4)
for t in np.arange(0.0, 3.01, 0.5):
    data = step(t)
    if t == 1.0:
        pathlines.inject([[3.0, 5.0, 0.5]])          # a particle released mid-run
    paths = pathlines.update(data)                   # every particle's path so far
    streak = streaklines.update(data)                # the line of the seed's particles

print(paths.num_cells, "paths,", paths.num_points, "points;", streak.num_points, "in the streak")
radius = np.linalg.norm(paths.world_positions()[:, :2] - [5.0, 5.0], axis=1)
assert np.allclose(radius[paths.fields["seed"].values.numpy() == 0], 2.0, atol=1e-3)
assert paths.num_cells == 3 and streak.num_cells == 1
```

The velocity is the probe of each step's own mesh, so the mesh may change
between steps; over a `PartitionedDataSet`, each point is probed on the rank
whose cells hold it, so particles stay on the rank that released them. A
particle that leaves the mesh stops where it last was, marked
`SPATIAL_BOUNDS`. `integrator` is `"rk4"` or `"euler"`; `step_size` (in the
steps' time unit) bounds each substep. Times convert between the steps' unit
and the velocity's: steps in milliseconds with a velocity in metres per second
work.

`extend_paths(paths, particles)` joins one step's particles onto the paths so
far, by `seed`: a polyline through each particle's points in time (`seed` and
`time` on the points), a particle's first point alone until it has moved. It
stands alone, so particles written step by step can be joined afterwards.
`Pathlines` is `Advection` and `extend_paths` together, returning the current
pathlines every step. `Streaklines` releases particles at its seeds every step
(`releases` of them, spread over the interval since the previous step) and
returns a polyline per seed through its particles still in the mesh, newest
first.

The particles live on the host in double precision, and each substep probes
both steps for every particle: point location runs on the device, but each
probe is its own set of launches, so this suits thousands of particles more
than millions.

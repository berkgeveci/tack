# Sample and trace fields

A field is a function of position, and many questions ask for its value
somewhere the mesh stores nothing: along a line, at a sensor, on another grid.
Others follow a velocity field: where do particles released here go? This page
shows Tack's filters for both, for readers who know VTK's `vtkProbeFilter`,
`vtkResampleToImage` and `vtkStreamTracer`. After it you can probe a dataset
at points or onto another dataset, put a mesh's fields on an image, and trace
streamlines and particles, choosing the integrator, the step and the device.

## Probe a dataset at points

`probe(data, where)` evaluates `data`'s fields at the points of `where`, as
VTK's `vtkProbeFilter` does. `where` is an `(n, 3)` array of points or another
dataset. Each point is located in the cell that holds it, and each field is
evaluated there through its own interpolation. The result is `where` with
every field as point data, and a field `valid`: 1 where a cell holds the
point, 0 where none does.

<!-- tack-check: probe -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 11), origin=(-1, -1, -1), spacing=(0.2, 0.2, 0.2))
x = data.positions()
data.fields["T"] = td.Field(td.H1(data), tack.array(300 + 10 * x[:, 0]))

points = np.array([[0.31, 0.2, 0.0], [-0.5, 0.1, 0.7], [5.0, 0.0, 0.0]])
samples = td.probe(data, points)
valid = samples.fields["valid"].values.numpy()
T = samples.fields["T"].values.numpy()
print(valid, T)                                   # [1 1 0] [303.1 295.    0. ]
assert list(valid) == [1, 1, 0] and np.allclose(T[:2], [303.1, 295.0])

line = td.uniform_grid((21,), origin=(-1, 0.1, 0.0), spacing=(0.1, 0, 0))   # a line of points
along = td.probe(data, line)                      # keeps the line's topology
assert np.allclose(along.fields["T"].values.numpy(), 300 + 10 * line.positions()[:, 0], atol=1e-4)
```

Check `valid` before reading a sample. A point outside the mesh gets 0 in
every field, and 0 may also be a real value.

- **Points as an array** become vertex cells, one per point. Their positions
  are world coordinates, converted in double precision.
- **A dataset** keeps its own topology and geometry. Probing onto a line of
  points samples along the line, and probing onto an image (`td.uniform_grid`)
  resamples the mesh onto it, as VTK's `vtkResampleToImage` and
  `vtkResampleWithDataSet` do. A probed field replaces a field of the same
  name in `where`.
- **Cell data** (a `Constant` field) is copied from the cell that holds the
  point. A field with an interpolation, such as point data (an
  [`H1` field](../reference/glossary.md#h1) of order 1) or an `H1` or `L2` field
  of any order, is evaluated in that cell. An integer field is rounded to the nearest integer, as VTK's probe
  rounds it.

Where a point lies in several cells, on a face or an edge they share, the
lowest cell id wins. At a discontinuity this decides which side is sampled.
It is not an average across the interface.

### Reuse the locator

Finding the cell that holds a point takes a
[locator](../reference/glossary.md#locator), a search structure over the
cells, as VTK's `vtkStaticCellLocator` is. Without one, `probe` uses the
locator the dataset's topology keeps, so probing the same mesh again does not
build it again. `td.CellLocator(data)` builds one you can pass to `probe`,
`advect` and `streamlines` as `locator=`. It serves any dataset of the same
cells and positions.

A locator follows the positions. After Tack writes them (a kernel, `fill`,
`copyto`), the kept one is rebuilt, and one you built is refused, since its
bins describe the old mesh. Positions changed through memory shared with
another library are not seen, so build a new locator then.

`probe` also takes AMR datasets and octrees, as VTK's `vtkAMRResampleFilter`
and `vtkHyperTreeGridProbeFilter` do ([adaptive data](adaptive-data.md)), and
datasets in pieces over MPI ranks ([filters over pieces](over-pieces.md)). See
the [reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.probe.probe).

## Put a mesh's fields on an image

Probing onto an image samples one point per image point. `voxelize(data,
grid)` instead gives each *voxel* (a cell of the image) the share of its
volume the data covers, as the cell field `fraction`, and each field's mean
over that share. That is what a simulation's fields are on a uniform grid, for
analysis or for machine learning.

<!-- tack-check: voxelize -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A ball of tetrahedra, cut from a cube, with a temperature rising along x.
mesh = td.tetrahedralize(td.uniform_grid((11, 11, 11), spacing=(0.1, 0.1, 0.1)))
x = mesh.positions().astype(np.float64)
mesh.fields["r"] = td.Field(td.H1(mesh), tack.array(np.linalg.norm(x - 0.5, axis=1)))
mesh.fields["T"] = td.Field(td.H1(mesh), tack.array(300.0 + 10.0 * x[:, 0]), unit="K")
ball = td.clip(mesh, "r", 0.4, inside=True)

voxels = td.uniform_grid((6, 6, 6), spacing=(0.2, 0.2, 0.2))
image = td.voxelize(ball, voxels, samples=6)
covered = image.fields["fraction"].values.numpy().sum() * 0.2 ** 3
volume = td.cell_geometry(ball).measure.values.numpy().sum()
print(f"covered {covered:.4f} of the ball's {volume:.4f}")
assert abs(covered - volume) < 0.02 * volume
assert image.fields["T"].unit == "K"
```

Each voxel is sampled at `samples` points along each axis (4 by default, so
64 in 3D), at the centres of its sub-voxels, through `probe`. So `fraction` is
right to about one sample's share, and a mean is the samples' mean, not an
integral. More samples cost more probes. Voxels the data does not reach are
hidden. `voxelize` takes anything `probe` takes. See the
[reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.voxelize.voxelize).

## Trace streamlines

`streamlines(data, field, seeds, step_size, steps)` follows a steady velocity
field from each seed, as VTK's `vtkStreamTracer` does. The result is one
polyline per seed, with the cell field `seed` (which seed the path started
from) and the point field `time` (when the particle reached each point).
`advect` takes the same arguments and returns only where each particle
stopped.

<!-- tack-check: streamlines -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 3), origin=(-1, -1, -0.1), spacing=(0.2, 0.2, 0.1))
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x), np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)))

seeds = np.array([[0.5, 0, 0], [0, 0.75, 0]], np.float32)
paths = td.streamlines(data, "velocity", seeds, step_size=0.05, steps=20)
ends = td.advect(data, "velocity", seeds, step_size=0.05, steps=20)

lengths = td.cell_geometry(paths).measure.values.numpy()
print(paths.num_cells, "lines of length", lengths.round(3))     # 2 lines of length [0.5 0.75]
# The flow turns at unit angular speed, so time 1 is one radian round.
assert np.allclose(lengths, [0.5, 0.75], atol=1e-3)
assert np.allclose(ends.positions()[0, :2], [0.5 * np.cos(1), 0.5 * np.sin(1)], atol=1e-4)
assert np.allclose(ends.fields["time"].values.numpy(), 1.0, atol=1e-5)
```

The velocity may be point data, cell data, or a field of any order, on a
straight or curved mesh. `seeds` is an `(n, 3)` array or a field of 3-vectors.
A seed that cannot take a step, outside the mesh say, keeps a point in the
output and makes no line.

### Choose the step

By default `step_size` is a time step: each step integrates `dx/dt = velocity`
over that time, so its length depends on the speed. `step_unit="length"`
makes it an arc length along the path instead, as `vtkStreamTracer` steps by
default. The particle then follows the velocity's direction, slow flow takes
steps as long as fast flow, and `steps * step_size` is the longest path. Where
speeds vary widely, length steps reach as far in far fewer steps. `time` on
the points is the particle's time either way.

The integrator is `"rk4"` (fixed-step fourth-order Runge-Kutta, the default),
`"euler"`, or `"rk45"`, which adapts the step as VTK's RK45 (Cash-Karp) does:

- `step_size` is the first step.
- A step whose error estimate exceeds `tolerance` (a distance, 1e-6 by
  default) is tried again, smaller.
- Each step taken proposes the next, between `min_step` and `max_step`. By
  default they are a fiftieth of `step_size` and twice it, VTK's ratios.

Smooth stretches take long steps and sharp turns short ones. On a swirl at
VTK's settings, `rk45` takes about half of fixed RK4's points and lies as
close to the path as VTK's RK45 does.

Two options end a path early, whatever the integrator. `max_length` stops a
path that has gone that far. `terminal_speed`, in the velocity's unit, as in
`vtkStreamTracer`, stops it at the first point where the speed is at or below
that value, so a path stops where the flow does. A step size, a length or a
speed may be a quantity such as `"5 mm"`, converted to the dataset's units.

### Read why a particle stopped

`advect` returns a dataset of the particles where they stopped, one vertex
cell each, with three point fields: `steps` (how many steps it took), `time`
and `status`. `status` holds the bits of `td.ParticleStatus`:

| Bit | Means |
|---|---|
| `SUCCESS` | the particle was traced |
| `TERMINATE` | its path ended inside the mesh: its steps were done, it reached `max_length`, or a bit below ended it |
| `SPATIAL_BOUNDS` | it left the mesh |
| `TOOK_ANY_STEPS` | it moved at all |
| `ZERO_VELOCITY` | it stopped where the velocity is zero |
| `SLOW_SPEED` | it stopped at `terminal_speed` |

Check `status` and `time` before treating a short path as a sign of slow flow.

`advect` also continues from its own output. Pass the returned dataset as
`seeds`, and each particle goes on from where it stopped, with its time. The
`steps` argument counts the steps taken before, so pass the new total:

<!-- tack-check: advect-resume -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 3), origin=(-1, -1, -0.1), spacing=(0.2, 0.2, 0.1))
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x), np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)))

seeds = np.array([[0.5, 0, 0], [0.9, 0.9, 0]], np.float32)
first = td.advect(data, "velocity", seeds, step_size=0.05, steps=20)
later = td.advect(data, "velocity", first, step_size=0.05, steps=40)    # 20 more

status = later.fields["status"].values.numpy()
print(later.fields["steps"].values.numpy(), later.fields["time"].values.numpy().round(3))
assert later.fields["steps"].values.numpy()[0] == 40
assert status[1] & td.ParticleStatus.SPATIAL_BOUNDS                  # the corner seed left
```

The full signatures are in the reference:
[`streamlines`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.streamlines.streamlines),
[`advect`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.advect.advect)
and [`ParticleStatus`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms._advection.ParticleStatus).

### At a wall

When a step would leave the mesh, `advect` and `streamlines` halve it until it
stays inside, as Viskores' `SmallStep` does, then push the particle out along
the velocity there. The path ends at the mesh's edge (`SPATIAL_BOUNDS`) only
if that push leaves the mesh. At a no-slip wall the flow runs along the wall,
so the particle stays in the fluid and goes on downstream.

VTK's `vtkStreamTracer` ends a line as soon as a step leaves the mesh, so near
walls its lines stop sooner. In the wake of the AhmedML car body (72 seeds
behind the body, length steps of 0.005), VTK ends 29 lines within 0.02 of the
body and Tack 1. Tack's wake lines are longer, a median of 9.0 against 6.5. A
`terminal_speed` does not make the two agree: at 5% of the free stream VTK
still ends 22 lines by the body and Tack 6, because the flow where VTK's steps
cross the wall is faster than that.

## Make tracing fast

**Choose the device by the number of seeds.** Each particle is one thread that
takes all its steps. A few seeds therefore use a few threads of a GPU built for
thousands, and one GPU thread is slower than a CPU core. Below a few thousand
seeds the CPU is usually faster, and above, the GPU. On a 28-core AMD EPYC
against an NVIDIA RTX 6000 Ada (ms, warm):

| mesh | seeds | CPU | CUDA |
|---|---|---|---|
| aneurysm, 8.7M tetrahedra | 25 | 8.4 | 32.0 |
| | 2,000 | 57.2 | 75.8 |
| | 20,000 | 871 | 331 |
| ONERA M6 wing, 46.9M cells | 17 | 41.7 | 84.4 |
| | 1,024 | 92.9 | 149 |
| | 16,384 | 1,190 | 108 |

For a few seeds on a mesh that lives on a GPU, trace on a CPU copy of it,
`data.to_device("cpu")`. The locator is built where the mesh is.

**Choose the locator by the mesh.** The default locator bins the cells on two
levels, as Viskores' `CellLocatorTwoLevel` does, so a mesh graded toward a
body in a large far field is not crowded into a few bins. On a small mesh of
cells of about one size, one uniform grid of bins holds fewer candidates a
bin, as VTK's `vtkStaticCellLocator` does. Build it with `levels=1` and pass
it in.

**Walk to the neighbours.** `neighbors=True` tries the face neighbours of the
cell a particle was last in before searching the locator. On large meshes this
takes about half the time, for a table of face neighbours (six ids a cell)
that the locator keeps. On surfaces and polyhedra it changes nothing.

Neither choice changes the result. Both locators find the same cells, and the
walk finds the cell the search would, so the paths are the same, bit for bit:

<!-- tack-check: locators -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((11, 11, 3), origin=(-1, -1, -0.1), spacing=(0.2, 0.2, 0.1))
x = data.positions()
rotation = np.stack([-x[:, 1], x[:, 0], np.zeros(len(x), np.float32)], axis=1)
data.fields["velocity"] = td.Field(td.H1(data), tack.array(rotation, dtype=tack.vec3(tack.f32)))
seeds = np.array([[0.5, 0, 0], [0, 0.75, 0]], np.float32)

default = td.streamlines(data, "velocity", seeds, 0.05, 100)
uniform = td.streamlines(data, "velocity", seeds, 0.05, 100,
                         locator=td.CellLocator(data, levels=1))
walking = td.streamlines(data, "velocity", seeds, 0.05, 100, neighbors=True)
assert np.array_equal(uniform.positions(), default.positions())
assert np.array_equal(walking.positions(), default.positions())
```

## Details

**`rk45` and divergent flow.** The adaptive integrator's step control makes
yes-or-no choices. Where paths diverge, in a recirculation or at a wall, a
change the size of rounding can send two runs, or two devices, along paths of
different lengths, each within the tolerance. A looser `tolerance` makes that
rarer.

**Pieces.** Over a `PartitionedDataSet`, give the field by name. Particles
cross from piece to piece, over MPI ranks or not, and each piece holds the
stretches of path it traced ([filters over pieces](over-pieces.md)).

**Unsteady flow.** `advect` and `streamlines` follow one steady field.
Repeating them step by step does not trace a particle through a flow that
changes in time. Pathlines and streaklines do that, on
[steps over time](over-time.md).

**Supported cells.** Probing and tracing take every shape, polyhedra (located
in the tetrahedra they are fanned into) and polygons (in the triangles they
are fanned into). Probing also takes polylines, on their segments, but
tracing does not. A polyhedron or polygon must be star-shaped from the mean of
its points, and the locator refuses one that is not.

## Where to go next

- [Steps over time](over-time.md): pathlines and streaklines through a field
  that changes from step to step.
- [Filters over pieces](over-pieces.md): probing and tracing across pieces and
  MPI ranks.
- [Points and particles](points-and-particles.md), when the data are the
  particles themselves.

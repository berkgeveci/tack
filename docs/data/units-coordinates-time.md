# Coordinates, units and time

Besides its cells and fields, a dataset records three facts about its data:
where its positions are measured from, in what units its positions and
fields are, and when the data are. This page is for readers who know VTK,
which keeps a unit in an array's `UNITS_LABEL` and a time in
`DATA_TIME_STEP`. After it you can hold coordinates far from zero without
losing precision, give positions and fields their units, and stamp a
dataset with its time.

## A survey far from zero, in metres, at one time

The example holds three LiDAR returns in projected coordinates, millions of
metres from the datum. It gives the positions a unit, gives the temperature
field a unit, and stamps the dataset with a time. Single precision keeps
about seven digits, so it cannot hold the quarter metre in 4,580,000.25.
The dataset keeps it anyway.

<!-- tack-check: survey-frame -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

world = np.array([[4580000.25, 512000.50, 10.0],       # metres, in double precision
                  [4580010.00, 512005.00, 12.0],
                  [4580020.75, 512010.25, 11.5]])
cloud = td.point_cloud(world, fields={"temperature": [12.5, 12.0, 11.8]},
                       unit="m", units={"temperature": "degC"})
cloud.time, cloud.time_unit = 3600.0, "s"

print(f"{np.float32(world[0, 0]):.2f}")  # 4580000.00: single precision drops the quarter metre
print(cloud.dtype, cloud.origin)        # tack.f32 (4580000.0, 512000.0, 10.0)
print(cloud.positions()[0])             # [0.25 0.5  0.  ]: single precision, from the origin
print(cloud.unit, cloud.fields["temperature"].unit, cloud.time, cloud.time_unit)   # m °C 3600.0 s

assert np.array_equal(cloud.world_positions(), world)    # every digit kept
```

The positions are stored in single precision, but relative to an origin that
is held in double precision. Their sum, `world_positions()`, is the input
exactly. The next sections explain the origin, the units and the time.

## Positions are relative to an origin

A dataset's **origin** is three double-precision numbers kept on the host.
Its geometry stores each position relative to the origin, in the geometry's
own type, which is often single precision. Kernels see only these relative
values. Since they are small, single precision keeps the detail that matters:
the spacing between returns, a cell's thin edge, a gradient's small
differences.

A dataset gives you its positions both ways:

- `data.positions()` returns the stored values, relative to the origin.
- `data.world_positions()` adds the origin in double precision and returns
  positions in the world's coordinates.

An origin of zero, the default, makes the two the same. You can set one with
`DataSet(..., origin=)`. `point_cloud` chooses one when it is given
double-precision positions far from zero for their size. It uses the lower
corner of their box, rounded down to whole units, as in the example.

An image keeps its own corner in its `UniformCoordinates`, in the geometry's
type. So a raster placed at survey coordinates loses digits in single
precision. `rebased(origin)` moves the image's corner into the dataset's
origin, in double precision, and keeps it an image:

<!-- tack-check: rebased-image -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

raster = td.uniform_grid((4, 3), origin=(4580000.0, 512000.0, 0.0), spacing=(0.1, 0.1), unit="m")
print(raster.world_positions()[1])     # [4580000.  512000.  0.]: the 0.1 m is lost

near = raster.rebased((4580000.0, 512000.0, 0.0))
print(near.positions()[1])             # [0.1 0.  0. ]
print(type(near.geometry.values).__name__)                  # UniformCoordinates
assert abs(near.world_positions()[1, 0] - 4580000.1) < 1e-6
```

Filters keep their input's origin in what they return. Anything you give a
filter in world coordinates, such as a probe point, a seed or the centre of
an implicit sphere, is moved to the dataset's origin in double precision
first. Datasets whose positions meet, such as the pieces of one mesh, share
one origin.

## Units on fields and positions

A field's **unit** is a string such as `"K"`, `"m/s"` or `"kg/m**3"`, given
as `Field(space, values, unit=)`. The **spatial unit**, `data.unit`, is the
unit of the positions and of the origin. It belongs to the geometry field,
and the grid and point cloud constructors take it as `unit=`. Arrays have no
unit, and kernels never see one. A unit is meaning, kept beside the values.

`None` means the unit is unknown, and it is the default. It is not the same
as `"dimensionless"`, the unit of a pure number. Nothing derived from an
unknown unit is guessed, so a gradient of a field of unknown unit has an
unknown unit too.

Tack parses and compares units with the [unyt](https://unyt.readthedocs.io)
library, and stores each in unyt's spelling. So `"m/s"`, `"m*s**-1"` and
`"m/s^1"` are one unit, and a string that is not a unit is refused when the
field is made. `td.units` has the helpers you need to work with them:

<!-- tack-check: unit-strings -->
```python
import tack
import tack.dataset as td

tack.set_default_device("cpu")

print(td.units.canonical("m/s^1"))              # m/s
print(td.units.canonical("degC"))               # °C
print(td.units.conversion("degC", "K"))         # (1.0, 273.15): K = 1.0 * °C + 273.15
print(td.units.same("km", "1000*m"))            # True
print(td.units.derived(("K", 1), ("m", -1)))    # K/m, the unit of a temperature gradient

data = td.uniform_grid((3, 2, 2), unit="km")
data.fields["density"] = td.Field(td.H1(data), tack.ones((data.num_points,)), unit="kg/m^3")
assert data.unit == "km" and data.fields["density"].unit == "kg/m**3"
```

Simulation code units are scaled units, such as `"3.0857e21*cm"` for a
kiloparsec in centimetres, and need nothing registered. Besides unyt's
units, Tack knows the US survey foot, `"ftUS"`, which State Plane
coordinates and older LiDAR use.

## What filters do with units

Units matter most in the Visualization layer, which this page only
previews. A filter that keeps a quantity keeps its unit, so a contour's
carried temperature is still in kelvin. A filter that computes a new
quantity derives its unit, so a gradient of temperature is in kelvin per
metre. A difference of Celsius temperatures is in kelvin. Parameters may be
quantities, such as a radius of `"10 cm"` on a mesh in metres. Positions in
two different known units are refused where they meet.
`td.convert(data, unit)` scales a field or the positions to another unit.
The [units reference](../vis-guide/reference/units.md) lists every rule.

## A dataset's time

`DataSet(..., time=, time_unit=)` records when the data are. Both may also be
set later, as `data.time` and `data.time_unit`. The time is one
double-precision number, or `None`. The time unit must be a unit of time,
such as `"s"` or `"ms"`, or `None` when it is unknown.

A dataset is one step of a simulation, not a series. Tack keeps no
collection of time steps, as a VTK reader's time steps would be. Filters
keep their input's time, so a step's results carry its time through a
pipeline. Filters that need earlier steps, such as a running average or
particle paths, keep what they need from each step you give them.
[Steps over time](../visualization/over-time.md) describes them. The pieces of one
partitioned dataset are at one time.

## Details

**Converting the positions.** Converting the spatial unit converts the
origin too, in double precision. Convert through the dataset, with
`td.convert(data, "m")`, not by converting the geometry field alone, which
would leave the origin in the old unit.

**Offset units.** Degrees Celsius and Fahrenheit have a zero that is not
zero. Tack does not multiply them, so a product of a Celsius field with
another is refused until you convert to kelvin.

**VTK.** The VTK interop reads and writes each array's `UNITS_LABEL` and the
dataset's `DATA_TIME_STEP`. VTK has no time unit and nowhere to keep an
image's spatial unit, so those are lost on the way to VTK. A label that is
not a unit is dropped with a warning. See
[VTK](../visualization/vtk.md).

**Cost.** unyt is imported the first time a unit is given, so a program that
gives none pays nothing for it.

## Where to go next

- [Cells and meshes](cells-and-meshes.md) lists the cell types and the kinds
  of mesh a dataset can hold.
- The [units reference](../vis-guide/reference/units.md) has the full rules
  for units kept, derived and refused.
- [Particles and point clouds](particles.md) builds on `point_cloud`.

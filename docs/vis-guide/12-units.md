# Coordinates, units and time

Knowing where values live is not enough: physical interpretation also needs
a coordinate frame, units and the time of the dataset. These belong to the
data, while conversions and temporal state are explicit operations.

## Relative and world positions

A dataset's `origin` is three doubles on the host. Geometry values give
positions relative to it. Kernels see those relative positions, keeping
nearby detail accurate in single precision even far from world zero.
`positions()` returns relative corner positions for supported H1 geometry;
`world_positions()` adds the origin in double precision.

The origin inside an image's `UniformCoordinates` formula is separate: it
locates the grid in the dataset's relative frame. Adding the dataset origin
gives world coordinates. `rebased(origin)` can change that frame while
preserving the world positions and implicit image representation.

Parameters that are physical positions, such as probe points, seeds and
implicit-function origins, use world coordinates and are moved to the dataset
frame before the device calculation. Pieces whose positions meet need a
consistent frame. Higher-order geometry has more coefficients than corner
positions; [geometry's space](00-function-spaces.md#geometry-is-a-field-too)
determines their meaning.

## Far from zero: the origin

A survey in projected coordinates sits hundreds of kilometres from zero,
and single precision has about seven digits: at 4,580,000 m its spacing is
half a metre. So a dataset's positions are an **origin**, three doubles on
the host, plus values in the geometry's type. `point_cloud` given
double-precision positions picks the lower corner of their box, in whole
units, as the origin; the values are then offsets of at most the tile's
size, and keep millimetres in single precision. Kernels see only the
offsets. `positions()` gives them; `world_positions()` gives the world
coordinates back in double precision.

Filters keep the origin, and anything given in world coordinates (a probe
point, a seed, an implicit function, a viewpoint) is moved to it in double
precision. A grid to deposit onto is rebased to the same origin, as shown in the [LiDAR recipe](08-point-clouds.md#clean-thin-and-grid-a-lidar-tile).
Pieces whose positions meet share one origin.


## Units are on fields

`Field(..., unit="K")` describes temperature. `data.unit` describes spatial
coordinates, including the dataset origin. Arrays themselves have no physical
unit. Canonical spellings are supplied by unyt; `None` means unknown, which
differs from `"dimensionless"`.

Filters keeping a quantity keep its unit. New quantities derive units:
temperature gradient is temperature per length, cell volume is length cubed,
and a difference of Celsius temperatures is in kelvin. Knowing a unit does
not identify whether a face array holds normal velocity or integrated flux.

Compatible physical quantities may still need conversion before being
combined. A quantity-valued parameter can be converted to the field's unit;
plain numbers are interpreted in that unit. See [the unit reference](reference/units.md)
for the complete rules and [the conversion recipe](connect-simulation-data.md#convert-units-before-combining-data)
for checked examples. Convert a dataset's spatial unit through the dataset
so its origin changes with its geometry coefficients.

## Time on a dataset

`DataSet(..., time=, time_unit=)` (or setting `data.time`, `data.time_unit`):
a double on the host or `None`, in a unit of time or `None`. Every filter keeps
both; `probe` gives back its target, with the target's time. The pieces of a
`PartitionedDataSet` and the boxes of an `AMRDataSet` are at one time
(compared in one unit, so 1 s and 1000 ms agree); an octree's is its roots'.
The VTK interop maps VTK's `DATA_TIME_STEP` both ways; VTK has no time unit.


## Metadata and pipeline results

A returned dataset carries the frame and metadata according to the operation's
rules. It can share topology or storage with its input. Sharing is not a deep
snapshot, and an old contour does not recompute when a simulation changes.
[Connect Simulation Data](connect-simulation-data.md) explains the update and
buffer-lifetime decisions; [Stream Steps](13-streaming.md) explains cached history.

You now have the common model. Choose [a task](introduction.md#choose-a-task),
read about [collections](collections-and-partitions.md), or continue into
[spaces and higher-order geometry](00-function-spaces.md).

??? info "Related detailed sections"
    These topics are developed in their own chapters.

    <span id="physical-units" data-guide-forward="#coordinates-units-and-time"></span>
    [Physical Units](12-units.md#coordinates-units-and-time).

    <span id="where-units-come-from" data-guide-forward="../reference/units/#where-units-come-from"></span>
    [Where units come from](reference/units.md#where-units-come-from).

    <span id="what-keeps-a-unit" data-guide-forward="../reference/units/#what-keeps-a-unit"></span>
    [What keeps a unit](reference/units.md#what-keeps-a-unit).

    <span id="derived-units" data-guide-forward="../reference/units/#derived-units"></span>
    [Derived units](reference/units.md#derived-units).

    <span id="quantities-as-parameters" data-guide-forward="../reference/units/#quantities-as-parameters"></span>
    [Quantities as parameters](reference/units.md#quantities-as-parameters).

    <span id="mismatches-are-refused" data-guide-forward="../reference/units/#mismatches-are-refused"></span>
    [Mismatches are refused](reference/units.md#mismatches-are-refused).

    <span id="converting" data-guide-forward="../connect-simulation-data/#converting"></span>
    [Converting](connect-simulation-data.md#converting).

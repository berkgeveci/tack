# Physical Units

A field's values are numbers; what they measure is its **unit**. Tack keeps
units on fields, not arrays: an array is storage and a field is meaning, so
kernels, devices and implicit arrays never see a unit. A unit is a string such
as `"m/s"`, `"K"` or `"Msun/kpc**3"`, kept in [unyt](https://unyt.readthedocs.io)'s
canonical spelling, so `"m/s"`, `"m*s**-1"` and `"m/s^1"` are one unit. `None`
is **unknown**, which is not the same as `"dimensionless"`: nothing derived
from an unknown unit is guessed.

The **spatial unit** is the geometry field's (`data.unit`), and the dataset's
`origin` is in it.

<!-- tack-check: units -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A raster at State Plane coordinates, in US survey feet, with a temperature.
image = td.uniform_grid((5, 4, 3), spacing=(10.0, 10.0, 10.0), unit="ftUS")
image.origin = (1_400_000.0, 600_000.0, 0.0)
celsius = np.linspace(0.0, 30.0, image.num_points).astype(np.float32)
image.fields["T"] = td.Field(td.H1(image), tack.array(celsius), unit="degC")
print(image.unit, image.fields["T"].unit)

# Filters that keep a quantity keep its unit, and the spatial unit with it.
cut = td.clip(image, "T", 15.0)
print(cut.unit, cut.fields["T"].unit)
assert cut.unit == "ftUS" and cut.fields["T"].unit == "°C"

# convert scales on the device; the origin converts in double precision.
metres = td.convert(image, "m")
kelvin = td.convert(metres, "K", field="T")
assert isinstance(metres.geometry.values, tack.arrays.UniformCoordinates)  # still an image
assert np.allclose(metres.world_positions(), image.world_positions() * 1200 / 3937)
assert np.allclose(kelvin.fields["T"].values.numpy(), celsius + 273.15)

# Derived quantities derive their units; a difference of degrees Celsius is in
# kelvin, so a temperature gradient is in kelvin per foot.
assert td.gradients(image, "T").unit == "K/ftUS"
stats = td.statistics(image, "T")
print(stats.unit_of("mean"), stats.unit_of("population_variance"))
assert stats.unit_of("population_variance") == "K**2"

# Parameters take quantities: a threshold in degrees Celsius on a field in
# kelvin, a radius in metres on positions in feet.
warm = td.threshold(kelvin, "T", "20 degC", "30 degC")
assert warm.num_cells == td.threshold(kelvin, "T", 293.15, 303.15).num_cells
cloud = td.point_cloud(image.world_positions(), unit="ftUS")
coarse = td.voxel_downsample(cloud, "10 m")             # 32.8 US survey feet
assert coarse.num_points == td.voxel_downsample(cloud, 10 * 3937 / 1200).num_points

# Positions in two different units do not meet.
try:
    td.probe(image, td.point_cloud(np.array([[1_400_010.0, 600_010.0, 5.0]]), unit="m"))
except ValueError as error:
    print("refused:", error)
```

## Where units come from

- `Field(space, values, unit=)`; `DataSet(..., unit=)`, `uniform_grid`,
  `rectilinear_grid` and `point_cloud` take `unit=` for the geometry, and
  `point_cloud(..., units={name: unit})` for its fields. A geometry `Field`
  brings its own unit.
- `OctreeDataSet(..., units={name: unit})` (and `from_cells`): octree fields
  are arrays, so their units are a mapping; the spatial unit is the roots'.
- The VTK interop reads and writes each array's `UNITS_LABEL`, the one unit
  VTK 9.7.1 keeps: on point and cell data, the points' array, a rectilinear
  grid's axes and a HyperTreeGrid's arrays and axes. VTK has nowhere to keep
  an image's spatial unit. A label that is no unit (VTK's labels are free
  text) is dropped with a warning; `^` is read as a power, as LLNL units and
  VTK write it.

Code units are scaled units, `"3.0857e21*cm"`, which need no registry of
their own. Besides unyt's units, Tack knows the US survey foot, `ftUS`
(1200/3937 m): State Plane coordinates and older LiDAR use it, and it differs
from the international foot by 2 parts per million, metres at survey
coordinates.

## What keeps a unit

Filters carry fields through `carry`, so every field a filter takes along
keeps its unit: clip, slice, threshold, extracts, contour (with
`contour_value` in the contoured field's unit), tetrahedralize, shrink, warp,
transform, probe, pieces, ghost cells, merging and `to_device`. So do the
filters whose result is the same quantity: `project`, `values_at_centers`,
traces, `magnitude`, `voxel_downsample`'s means, SPH interpolation, and
`cell_centers` and `face_centers` (positions, in the spatial unit).

Cylindrical and spherical coordinates give an unknown spatial unit, since an
angle and lengths are no one unit.

## Derived units

Filters that compute a new quantity derive its unit from their inputs':

| Filter | Unit |
|---|---|
| `gradients` | the field's per spatial unit; `flow_quantities`' divergence and vorticity the gradient's, the Q-criterion its square |
| `jump` | the field's |
| `dot`, `cross` | the product of the two, by its SI name where it has one (`N` times `m` is `J`) |
| `composite` | the fields' one unit (two known units are refused) |
| `log_values` | a pure number, of a pure number (a dimensioned field is refused) |
| `cell_geometry` | measures in the spatial unit to the cells' dimension (`m**3`, `m**2`, `m`), centroids in it |
| `face_geometry`, `edge_lengths` | areas in the spatial unit to the faces' dimension (an edge's length on a surface), unit normals pure numbers; lengths in it |
| `net_flux`, `divergence`, `perot` | the flux's; it over the cells' measure; the faces' components' |
| `statistics` | `Statistics.unit` and `weight_unit`, and `unit_of(name)` for each number: the mean, extremes and deviations in the field's unit, the variances in its square, `sum` times the weights', skewness and kurtosis pure numbers |
| `deposit` | counts pure numbers, sums, means and extremes in the field's; `per_volume` over the cells' measure (`1/m**3`, `kg/m**3`) |
| `sph_density`, `sph_interpolate` | the mass per volume; the field's, times `m / (rho L^d)` (a pure number for consistent units) |
| `halo_catalog`, `outliers`, `point_normals` | masses in the mass field's, radii and centres and mean distances in the spatial unit; normals pure numbers |
| `advect`, `streamlines` | `time` in the spatial unit over the velocity's |
| `interpolate(..., unit=)` | what the caller says: a function's code does not say its unit |

A difference of values with an offset zero -- degrees Celsius or Fahrenheit --
is in the absolute unit of the same size, kelvin or Rankine: gradients, jumps,
variances and deviations. Such values are not multiplied (`dot` refuses them):
convert to kelvin first. Labels, ids, flags and counts that are not
quantities (`region`, `cell_ids`, `outlier`, `neighbors`) have no unit, and a
result computed from a bare number -- `upwind_flux`'s constant velocity, a
mass given as one number, the potential's `G` -- is unknown.

## Quantities as parameters

A parameter in a unit takes a quantity: a string such as `"10 cm"`, a pair
`(value, unit)` (the value a number or an array), or a unyt or pint quantity.
It is converted to what the parameter is compared with; a plain number is
taken to be in that unit already, as before.

| Parameter | Converted to |
|---|---|
| `voxel_downsample`'s size; `radius=` of `outliers`, `connected_components`, `point_normals`, `ghost_points`; `periodic=` boxes; `PointLocator`'s bin size and `neighbors`' radius; SPH smoothing lengths given as one number; the halo catalogue's softening; `slice_plane`'s origin and distances | the spatial unit |
| `threshold`'s and `threshold_points`' bounds, `contour`'s values, `clip`'s value by a field | the field's unit, offsets included (`"20 degC"` on a field in kelvin) |
| `advect`'s and `streamlines`' step size | the spatial unit over the velocity's |
| a mass or density given as one number to SPH or the halo catalogue, `upwind_flux`'s velocity | none: the quantity brings its unit, so the result has one (`mass=(1e10, "Msun")` makes densities in `Msun/kpc**3`) |

A quantity needs a unit to convert to: one given for a dataset or field of
unknown unit is refused, as is one for an implicit function's values, which
have none. A string that names a field (`smoothing_length="h"`) is the
field.

## Mismatches are refused

Positions compared or moved between datasets must be in one spatial unit, as
they must be relative to one origin: probe's points, pieces of a
`PartitionedDataSet`, the boxes of an `AMRDataSet` and an octree's probe
refuse two known units, naming both and `convert`. An unknown unit cannot be
checked, so it is not refused.

## Converting

`convert(data, unit)` gives the dataset with its geometry in `unit`, and
`convert(data, unit, field=name)` with that field in it: a new dataset, the
values scaled (and offset, from degrees Celsius to kelvin, say) on the device,
everything else kept. The geometry's conversion scales the origin on the host
in double precision, so survey coordinates keep their digits; an image stays
an image and a rectilinear grid keeps its axes. The unit converted from must
be known. Integer fields convert only by relabelling (`"1000*m"` to `"km"`).

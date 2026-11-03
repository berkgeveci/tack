# Points and particles

This page is for anyone who analyses data that comes as points rather than
cells: LiDAR and photogrammetry scans, or the particles of SPH, N-body and
molecular dynamics codes. It shows how to find each point's neighbours, mark
noise, thin a cloud, estimate normals, put points on a grid, resample SPH
particles and find dark matter halos. Each algorithm names its VTK or
Viskores counterpart, so you can find the filter you already use.

A point cloud in Tack is a dataset whose cells are its points, one vertex
cell each. Its fields are point data. How a cloud is stored, and how it keeps
its precision far from zero through its origin, is explained in
[Particles and point clouds](../data/particles.md).

<!-- tack-check: points-first -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A noisy scan of a tilted plane: 20,000 points, 30 stray returns above it.
rng = np.random.default_rng(0)
xy = rng.uniform(0.0, 10.0, (20000, 2))
z = 0.1 * xy[:, 0] + rng.normal(0.0, 0.01, len(xy))
z[:30] += 5.0
scan = td.point_cloud(np.column_stack([xy, z]), fields={"z": z})

marked = td.outliers(scan, radius=0.2, min_neighbors=3)    # adds "outlier", 1 for noise
clean = td.threshold_points(marked, "outlier", 0, 0)        # keeps the rest
thin = td.voxel_downsample(clean, 0.25)                     # one point per 25 cm voxel
oriented = td.point_normals(thin, k=10)                     # adds "normals"

print(scan.num_points, "->", clean.num_points, "->", thin.num_points, "points")
normals = oriented.fields["normals"].values.numpy()
assert scan.num_points - clean.num_points >= 30
assert np.median(normals[:, 2]) > 0.99                      # up, off the plane
```

```text
20000 -> 19970 -> 1865 points
```

## The algorithms

Every algorithm here takes a point cloud and returns a new dataset or field.
Most of them search each point's neighbourhood through a
[locator](../reference/glossary.md), a structure that bins the points so a
search visits only nearby ones.

| Task | Algorithm | VTK or Viskores counterpart |
|---|---|---|
| Find the points within a radius, or the k nearest | [`PointLocator`](../vis-guide/reference/data-model.md#tack.dataset.PointLocator) | `vtkStaticPointLocator` |
| Mark noise | [`outliers`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.outliers.outliers) | `vtkRadiusOutlierRemoval`, `vtkStatisticalOutlierRemoval` |
| Keep the points whose field lies in a range | [`threshold_points`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.threshold_points.threshold_points) | `vtkThresholdPoints` |
| Thin to one point per voxel | [`voxel_downsample`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.voxel_downsample.voxel_downsample) | `vtkVoxelGrid` |
| Estimate surface normals | [`point_normals`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.point_normals.point_normals) | `vtkPCANormalEstimation` |
| Group points closer than a distance | [`connected_components(radius=)`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.connected_components.connected_components) | `vtkEuclideanClusterExtraction` |
| Count, sum, average or take extremes on a grid | [`deposit`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.deposit.deposit) | Viskores' `ParticleDensityNearestGridPoint`, `ParticleDensityCloudInCell` |
| SPH density | [`sph_density`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.sph.sph_density) | none |
| SPH resampling onto other points | [`sph_interpolate`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.sph.sph_interpolate) | `vtkSPHInterpolator` |
| A catalogue of halos | [`halo_catalog`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.halo_catalog.halo_catalog) | Viskores' cosmotools |

`extract_points` and `mask_points`, VTK's `vtkExtractPoints` and
`vtkMaskPoints`, also take point clouds; see the
[algorithms reference](../vis-guide/reference/algorithms.md#point-clouds).
Each algorithm here is deterministic. None of them sums with atomics, so the
same input gives the same values on every run.

## Find neighbours with a point locator

`PointLocator` sorts a cloud's points into uniform bins. `neighbors(radius)`
returns every point within a radius of each query, and `nearest(k)` the k
nearest points, nearest first. By default the queries are the cloud's own
points, each leaving itself out. Pass other points to query them instead.

<!-- tack-check: points-locator -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

rng = np.random.default_rng(1)
cloud = td.point_cloud(rng.uniform(0.0, 1.0, (5000, 3)))
locator = td.PointLocator(cloud)

offsets, ids = locator.neighbors(0.05)         # within 0.05 of each point
nearest, squared = locator.nearest(8)          # the 8 nearest, nearest first
probes = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
closest, _ = locator.nearest(3, probes)        # the 3 nearest to other points

offsets, nearest = offsets.numpy(), nearest.numpy()
print("point 0 has", offsets[1] - offsets[0], "neighbours within 0.05")
x = cloud.positions()
assert np.array_equal(nearest[0], np.argsort(((x - x[0]) ** 2).sum(axis=1))[1:9])

# Filters take the locator, so several of them bin the cloud once.
marked = td.outliers(cloud, k=8, deviations=2.0, locator=locator)
```

`neighbors` returns CSR arrays: the neighbours of query `q` are
`ids[offsets[q]:offsets[q + 1]]`. `nearest` returns `(m, k)` arrays of ids
and squared distances, with -1 where a cloud has fewer than k points. Query
points given as a NumPy array are world positions.

A filter given no locator uses the one the dataset's topology keeps, so a
chain of filters on one cloud bins it once. A locator records the positions
it was built from and is refused once Tack writes to them. For kernels of
your own, `locator.grid()` gives the bins as a
[device class](../core/device-classes.md).

## Mark and remove outliers

`outliers` marks points rather than removing them, as PDAL does. It adds
`outlier`, a `u8` point field that is 1 for noise. `threshold_points(data,
"outlier", 0, 0)` keeps the rest, and the marks stay for inspection. It has
two tests:

- The **radius test** (`radius` and `min_neighbors`) marks a point with fewer
  than `min_neighbors` other points within `radius`, as VTK's
  `vtkRadiusOutlierRemoval` does. It adds `neighbors`, each point's count.
- The **statistical test** (`k` and `deviations`) marks a point whose mean
  distance to its k nearest is more than `deviations` standard deviations
  above the mean of those distances, as PCL does. It adds `mean_distance`.

VTK's `vtkStatisticalOutlierRemoval` also removes points that many deviations
*below* the mean, such as duplicates in dense spots. Tack marks only the far
ones. The two agree on the mean distances.

## Thin a cloud to one point per voxel

`voxel_downsample(data, size)` keeps one point per occupied voxel of a grid
with sides of `size`, anchored at the points' lower corner, as VTK's
`vtkVoxelGrid` and PDAL's `filters.voxeldownsize` do. `keep` chooses the
point:

| `keep` | The point kept | Its fields |
|---|---|---|
| `"centroid"` (default) | a new point at the mean of the voxel's points | floating-point fields averaged; integer fields, such as a classification, from the voxel's first point |
| `"first"` | the voxel's lowest-numbered point | the point's own |
| `"nearest"` | the voxel's point nearest the centroid | the point's own |

Point fields come along, chosen by `fields=` as for every filter. Cell fields
do not, since the output's cells are new.

## Estimate normals

`point_normals` fits each point's neighbourhood and adds `normals`, unit
vectors in the direction the neighbourhood spreads least, as VTK's
`vtkPCANormalEstimation` does. The neighbourhood is the `k` nearest points or
those within `radius`. A normal points toward `viewpoint` when you give one,
otherwise along `direction`, which is up by default: the outside of terrain.
A point with fewer than three neighbours, or whose neighbours lie on a line,
gets a zero normal.

## Put points on a grid

`deposit(data, grid, field, reduce=)` sums points into an image, a dataset
made by `uniform_grid`. With `method="nearest"` each point goes to the cell
holding it, and the result is a cell field. With `method="cic"` (cloud in
cell) each point is spread over the corners of its cell by trilinear weights,
and the result is a point field. Without `field` it counts points. With a
field, `reduce` is `"sum"`, `"mean"`, `"count"`, or for `"nearest"` also
`"min"` and `"max"`. `per_volume=True` divides by each cell's volume, which
turns counts or masses into densities.

The highest return in each cell of a LiDAR survey is a surface model, and the
lowest ground return a terrain model:

<!-- tack-check: points-deposit -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A 100 m survey tile in UTM coordinates: sloping ground and a building.
rng = np.random.default_rng(1)
east, north = rng.uniform(0.0, 100.0, (2, 40000))
height = 0.02 * east + rng.normal(0.0, 0.02, len(east))
roof = (np.abs(east - 50.0) < 10.0) & (np.abs(north - 50.0) < 10.0)
height[roof] += 8.0
world = np.column_stack([east + 431000.0, north + 4580000.0, height + 120.0])
tile = td.point_cloud(world, fields={"z": world[:, 2]})
print(tile.origin)                                   # the tile's corner, in whole metres

# A 2D image of 2 m cells over the tile, moved to the tile's origin.
image = td.uniform_grid((51, 51), origin=(431000.0, 4580000.0), spacing=(2.0, 2.0))
image = image.rebased(tile.origin)
surface = td.deposit(tile, image, "z", reduce="max").values.numpy().reshape(50, 50)
print(f"roof {surface[25, 25]:.1f} m, ground {surface[5, 5]:.1f} m")
assert abs(surface[25, 25] - 129.0) < 0.2 and abs(surface[5, 5] - 120.2) < 0.2
```

A cell no point reaches holds 0 for counts and sums, and NaN for the other
reductions. Points outside the image are left out.

The survey's coordinates are millions of metres from zero. `point_cloud`
gives such a cloud an origin at its lower corner, so its positions keep
millimetres in single precision. An image needs the same care, since its
corner is stored in the geometry's precision too. `rebased(tile.origin)`
moves the image's corner next to zero in double precision and keeps it an
image. [Coordinates, units and time](../data/units-coordinates-time.md)
explains origins.

## Resample SPH particles

`sph_interpolate(particles, target)` evaluates the particles' fields at a
target's points by SPH kernel sums, as VTK's `vtkSPHInterpolator` does. Each
particle `j` contributes `(m_j / rho_j) A_j W(|x - x_j|, h_j)` to a point
`x`, out to its kernel's support. The target is any dataset (an image, a
slice, another cloud) or an `(n, 3)` array of world positions. When a code
writes no density, `sph_density` computes one, adding `density` to the
particles.

<!-- tack-check: points-sph -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Particles of equal mass on a jittered lattice filling the unit cube: density 1.
rng = np.random.default_rng(2)
side = np.arange(0.0, 1.0, 0.05) + 0.025
x = np.stack(np.meshgrid(side, side, side, indexing="ij"), axis=-1).reshape(-1, 3)
x += rng.uniform(-0.005, 0.005, x.shape)
particles = td.point_cloud(x, origin=(0.0, 0.0, 0.0),
                           fields={"h": np.full(len(x), 0.06), "mass": np.full(len(x), 0.05**3),
                                   "T": 300.0 + 10.0 * x[:, 0]})
particles = td.sph_density(particles)                           # adds "density"
inner = np.all((x > 0.2) & (x < 0.8), axis=1)
assert abs(np.median(particles.fields["density"].values.numpy()[inner]) - 1.0) < 0.02

image = td.uniform_grid((11, 11, 11), origin=(0.0, 0.0, 0.0), spacing=(0.1, 0.1, 0.1))
resampled = td.sph_interpolate(particles, image, fields=["T"], normalize=True)
T = resampled.fields["T"].values.numpy()
shepard = resampled.fields["shepard"].values.numpy()
print(f"coverage {shepard.min():.2f} at a corner, {np.median(shepard):.2f} inside")
p = image.positions()
inside = np.all((p > 0.15) & (p < 0.85), axis=1)
assert np.abs(T - (300.0 + 10.0 * p[:, 0]))[inside].max() < 0.05
```

The result also has `shepard`, the sum of `(m / rho) W` at each point. Near 1
the particles cover the point, and near 0 they do not. `normalize=True`
divides by it, as VTK's Shepard processing does, which keeps a constant exact
out to the particles' edge.

The main options:

- `kernel` is `"cubic"` (the default), `"quartic"`, `"quintic"` or
  `"wendland"`, normalized in `dimension` dimensions. They are VTK's kernels,
  and the results agree with `vtkSPHInterpolator` to rounding.
- `smoothing_length`, `mass` and `density` name point fields, `h`, `mass` and
  `density` by default. Each also takes one number for every particle.
- `periodic` reaches across the faces of a periodic box.

`sph_density` takes the same options. Each particle's density sums over the
particles within its own smoothing length's reach, itself included: the
gather form, as SPLASH computes it.

## Find halos in a periodic box

A halo finder groups dark matter particles closer than a linking length.
That is friends of friends, and in Tack it is `connected_components` with the
linking length as `radius`. On a point cloud, whose cells are its points,
`connected_components(radius=)` is Euclidean clustering, as VTK's
`vtkEuclideanClusterExtraction` does. It adds `point_region`, each point's
group, numbered from 0, or -1 for none. `min_size` drops groups of fewer
points and numbers the rest again. `halo_catalog` then summarizes each group.

<!-- tack-check: points-halos -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Three halos in a periodic box of side 50, one across a face, and a background.
rng = np.random.default_rng(3)
box = 50.0
centres = np.array([[10.0, 10.0, 10.0], [30.0, 25.0, 40.0], [49.5, 20.0, 20.0]])
clumps = [c + rng.normal(0.0, 0.4, (n, 3)) for c, n in zip(centres, (3000, 1500, 800))]
x = np.concatenate(clumps + [rng.uniform(0.0, box, (20000, 3))]) % box
v = rng.normal(0.0, 1.0, x.shape)
particles = td.point_cloud(x, fields={"v": v}, origin=(0.0, 0.0, 0.0))

linking = 0.2 * box / len(x) ** (1 / 3)
groups = td.connected_components(particles, radius=linking, min_size=100, periodic=box)
catalog = td.halo_catalog(groups, mass=1.0e10, velocity="v", periodic=box)
counts = catalog.fields["count"].values.numpy()
print(catalog.num_points, "halos of", sorted(counts.tolist()), "particles")
assert catalog.num_points == 3 and counts.min() > 700
```

`periodic` is a box: its side, its three sides, or its two corners. Distances
are then to a point's nearest image, so a halo across a face of the box is
one halo. `PointLocator`, `outliers`, `point_normals`, `connected_components`
and the SPH algorithms take `periodic` too.

The catalogue is a point cloud with one point per halo, at its most bound
particle, or at its centre of mass with `center="com"`. Its fields are:

| Field | Meaning |
|---|---|
| `halo`, `count`, `mass` | the label, the number of particles, their total mass |
| `com` | the centre of mass |
| `mbp`, `potential` | the most bound particle's index and its potential |
| `velocity`, `dispersion` | the mass-weighted mean velocity and the 1D velocity dispersion (with `velocity=`) |
| `radius` | the distance to the halo's farthest particle |

`mass` is one number for every particle or a point field. The most bound
particle is exact for every halo: large halos are pruned by bounds, not
estimated. `softening` and `G` set the potential's softening length and
gravitational constant.

## Point clouds in pieces

A cloud split over ranks, or a survey cut into tiles, needs each piece's
neighbours from the other pieces. `ghost_points` copies them in, and
`connected_components`, `outliers`, `voxel_downsample`, `deposit` and
`halo_catalog` take the pieces directly and give the whole cloud's answer.
[Filters over pieces](over-pieces.md#point-clouds-in-pieces) shows how.

## Details

**Reproducible sums.** `voxel_downsample`, `deposit`, the SPH sums and
`halo_catalog` add their terms in point order, so a run gives the same bits
every time. Counts, minima and maxima are the same on every device. Sums agree
to rounding, since a GPU may fuse a multiply and an add. Every distance that
decides something, such as a voxel's nearest point or the k nearest, is
computed alike on every device, so the same points are chosen everywhere.

**Precision.** The statistical outlier test's mean and deviation are taken in double
precision on the host. `point_normals` cannot tell a strip about 4% as wide
as it is long from a line in single precision, and gives it a zero normal.

**Integer fields** keep their type through `point_cloud`. A classification
or an intensity stays an integer, while floating-point fields take the
cloud's precision.

## Where to go next

- [Particles and point clouds](../data/particles.md) explains the model
  behind these algorithms.
- [Sample and trace](sample-and-trace.md) probes meshes at points and traces
  particles through velocity fields.
- [Filters over pieces](over-pieces.md) runs these algorithms on clouds split
  over ranks.

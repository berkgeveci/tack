# Point clouds and particles

A point cloud is a dataset whose cells are its points: one vertex cell per
point, stored nowhere. LiDAR surveys, SPH particles, N-body particles and
tracers from a simulation are all point clouds, with fields per point. The
filters that work on points (`threshold_points`, `probe`'s target,
`statistics`, `transform`) work on them as on any dataset, and a
**point locator** adds the filters that need each point's neighbours:
cleaning and thinning, normals, clusters, deposition onto a grid, SPH
interpolation and cosmology halos.

| Task | Algorithm |
|---|---|
| Make a cloud from positions and fields | `point_cloud` |
| Find the points within a radius, or the k nearest | `PointLocator` |
| Mark noise | `outliers` (radius or statistical) |
| Thin to one point per voxel | `voxel_downsample` |
| Estimate surface normals | `point_normals` |
| Group points closer than a distance | `connected_components(radius=)` |
| Count, sum, average, or take the extremes on a grid | `deposit` |
| Density and kernel-weighted values of SPH particles | `sph_density`, `sph_interpolate` |
| Halos of dark matter particles | `connected_components(radius=, periodic=)`, `halo_catalog` |

The [reference](reference/dataset-api.md#origins-and-point-clouds) has every
argument. Each filter is deterministic, with no atomics in its sums: the
same input gives the same values every run.

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
precision. A grid to deposit onto takes the same origin, as below. Pieces
whose positions meet share one origin.

## Clean, thin and grid a LiDAR tile

<!-- tack-check: lidar -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# A 100 m survey tile in UTM coordinates: terrain, a building, and noise.
rng = np.random.default_rng(1)
n = 40000
east, north = rng.uniform(0.0, 100.0, n), rng.uniform(0.0, 100.0, n)
height = 0.02 * east + rng.normal(0.0, 0.02, n)
roof = (np.abs(east - 50.0) < 10.0) & (np.abs(north - 50.0) < 10.0)
height[roof] += 8.0
height[:40] += rng.uniform(20.0, 40.0, 40)                 # birds and spurious returns
world = np.stack([east + 431000.0, north + 4580000.0, height + 120.0], axis=1)
tile = td.point_cloud(world, fields={"z": world[:, 2],
                                    "classification": np.where(roof, 6, 2).astype(np.uint8)})
print(tile.origin)                     # the box's lower corner in whole metres
assert np.abs(tile.world_positions() - world).max() < 1e-3  # millimetres, in f32

# Mark the outliers, keep the rest, thin to 0.5 m voxels.
marked = td.outliers(tile, radius=1.0, min_neighbors=3)
kept = td.threshold_points(marked, "outlier", 0, 0)
thin = td.voxel_downsample(kept, 0.5)
print(kept.num_points, "kept,", thin.num_points, "after thinning")
assert tile.num_points - kept.num_points >= 40

# Normals point up on the ground and the roof.
normals = td.point_normals(thin, k=12).fields["normals"].values.numpy()
assert np.median(normals[:, 2]) > 0.99

# A surface model: the highest return in each 2 m cell of a 2D image.
grid = td.uniform_grid((51, 51), spacing=(2.0, 2.0, 1.0))  # 100 m from the tile's corner
image = td.DataSet(grid.topology, grid.geometry, origin=tile.origin)
dsm = td.deposit(kept, image, "z", reduce="max").values.numpy().reshape(50, 50)
print(f"roof {dsm[25, 25]:.1f} m, ground {dsm[5, 5]:.1f} m")
assert abs(dsm[25, 25] - 129.0) < 0.2 and abs(dsm[5, 5] - 120.2) < 0.2
```

`outliers` **marks** points (`outlier`, 1 for noise) rather than removing
them, as PDAL does: `threshold_points` keeps the rest, and the marks stay
for inspection. The radius test marks points with fewer than
`min_neighbors` others within `radius`, as VTK's `vtkRadiusOutlierRemoval`;
`outliers(data, k=8)` is the statistical test instead, a mean distance to
the k nearest more than `deviations` standard deviations above the mean,
as VTK's `vtkStatisticalOutlierRemoval` and PCL.

`voxel_downsample` keeps one point per occupied voxel, of exactly the size
asked: the voxel's centroid (floating-point fields averaged, integer ones
such as a classification taken from its first point), its first point, or
its point nearest the centroid (`keep=`). `point_normals` fits each point's
neighbourhood, its k nearest or those within a radius, and signs the normal
upward or toward a `viewpoint`.

`deposit` puts each point in the image cell holding it (`method="nearest"`,
a field on the cells), or spreads it over the cell's corners by trilinear
weights (`"cic"`, cloud in cell, on the points). Without a field it counts;
with one it takes the `"sum"`, `"mean"`, `"min"` or `"max"`: the highest
return per cell is a surface model, the lowest ground return a terrain
model. Cells nothing reaches hold 0 for counts and sums, NaN otherwise.

The image here is built in the tile's frame and given the tile's origin;
an image whose own coordinates start at 431,000 m would hold them in single
precision.

## Neighbours, directly

The filters above share one structure, which you can use yourself:

```python
locator = td.PointLocator(tile)                      # uniform bins, sorted by bin
offsets, ids = locator.neighbors(1.0)                # within 1 m of each point (CSR)
nearest, d2 = locator.nearest(8)                     # the 8 nearest, nearest first
```

`neighbors` and `nearest` take other query points too, world positions or
offsets. A filter given no locator uses the one the dataset's topology
keeps, so several filters on one cloud bin it once; a locator is refused
once the positions change. Its device class (`locator.grid()`) gives your
own kernels the bins.

## SPH particles

<!-- tack-check: sph -->
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
particles = td.point_cloud(x.astype(np.float32), origin=(0.0, 0.0, 0.0),
                           fields={"h": np.full(len(x), 0.06), "mass": np.full(len(x), 0.05**3),
                                   "T": 300.0 + 10.0 * x[:, 0], "one": np.ones(len(x))})
particles = td.sph_density(particles)                       # adds "density"
inner = np.all((x > 0.2) & (x < 0.8), axis=1)
rho = particles.fields["density"].values.numpy()
assert abs(np.median(rho[inner]) - 1.0) < 0.02

# Resample onto an image. The Shepard sum says how well the particles cover
# each point; normalize divides by it, which keeps a constant exact out to the
# particles' edge, where the plain sum falls off.
image = td.uniform_grid((11, 11, 11), origin=(0.0, 0.0, 0.0), spacing=(0.1, 0.1, 0.1))
out = td.sph_interpolate(particles, image, fields=["T", "one"], normalize=True)
T, one = (out.fields[name].values.numpy() for name in ("T", "one"))
shepard = out.fields["shepard"].values.numpy()
print(f"coverage {shepard.min():.2f} at a corner, {np.median(shepard):.2f} inside")
assert np.abs(one - 1.0).max() < 1e-5
inside = np.all((image.positions() > 0.15) & (image.positions() < 0.85), axis=1)
assert np.abs(T - (300.0 + 10.0 * image.positions()[:, 0]))[inside].max() < 0.05
```

`sph_interpolate` sums `(m_j / rho_j) A_j W(|x - x_j|, h_j)` over the
particles reaching each target point (the scatter form, each particle out to
its kernel's support), onto any dataset's points: an image, a slice, another
cloud, or an array of world positions. `kernel` is `"cubic"` (the default),
`"quartic"`, `"quintic"` or `"wendland"`, VTK's, normalized in `dimension`
dimensions; the results agree with VTK's `vtkSPHInterpolator` to float
rounding. Smoothing length, mass and density are point fields, by name, or
one number for all. `sph_density` computes the density (the gather form,
SPLASH's) when a code does not write it.

## Halos in a periodic box

<!-- tack-check: halos -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Dark matter particles in a periodic box of side 50: three halos, one of them
# across a face of the box, and a smooth background.
rng = np.random.default_rng(3)
box = 50.0
centres = np.array([[10.0, 10.0, 10.0], [30.0, 25.0, 40.0], [49.5, 20.0, 20.0]])
parts = [c + rng.normal(0.0, 0.4, (n, 3)) for c, n in zip(centres, (3000, 1500, 800))]
x = np.concatenate(parts + [rng.uniform(0.0, box, (20000, 3))]) % box
v = rng.normal(0.0, 1.0, x.shape)
particles = td.point_cloud(x.astype(np.float32), fields={"v": v}, origin=(0.0, 0.0, 0.0))

# Friends of friends: particles closer than the linking length are one halo.
linking = 0.2 * box / len(x) ** (1 / 3)
fof = td.connected_components(particles, radius=linking, min_size=100, periodic=box)
catalog = td.halo_catalog(fof, mass=1.0e10, velocity="v", periodic=box)
counts = catalog.fields["count"].values.numpy()
print(catalog.num_points, "halos of", sorted(counts.tolist()), "particles")
print(catalog.positions())                                  # their most bound particles
assert catalog.num_points == 3 and counts.min() > 700
gap = catalog.positions()[:, None, :] - centres[None, :, :]
gap -= box * np.round(gap / box)                            # across the box's faces
assert (np.linalg.norm(gap, axis=2).min(axis=1) < 0.5).all()
```

With `periodic` (a box's side, its three sides, or two corners), distances
are to the nearest image, so a halo across a face is one halo. Friends of
friends is `connected_components` with the linking length as `radius`;
`min_size` drops the small groups, renumbering the rest. `halo_catalog`
gives one point per halo, at its most bound particle (`center="com"` for
the centre of mass), with its `count`, `mass`, centre of mass, the most bound
particle's index and potential, the mass-weighted mean `velocity` and the
velocity `dispersion`, and its `radius`. The most bound particle is exact
for every halo: large halos are pruned by bounds, not estimated. On
ParaView's HACC test data the halos, their counts and their most bound
particles are those of ParaView's ANL halo finder, which counts the halos
across a face of the box twice.

## Point clouds in pieces

A cloud cut into pieces, by a simulation's ranks or a survey's tiles, needs
each piece's neighbours from the others. `ghost_points` gives them, and
clusters, catalogues, outliers, voxels and deposits take the pieces: see
[Point clouds in pieces](09-pieces-across.md#point-clouds-in-pieces).

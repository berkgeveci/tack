# Reduce: statistics and regions

Some answers are a few numbers, not a new mesh: the range and mean of a field,
how its values spread, how many separate regions a threshold leaves and how
big each is. This page shows Tack's reductions for readers who know VTK's
`vtkDescriptiveStatistics` and `vtkConnectivityFilter`: `statistics`,
`entropy` and `connected_components`. After it you can summarize a field,
weight the summary by volume or mass, and label the connected regions of a
mesh or a point cloud.

## Summarize a field

`statistics(data, field)` returns a `Statistics` of a scalar field's values:
their count `n`, `min`, `max`, `sum` and `mean`, the central moments, the
sample and population variances and standard deviations, the skewness and the
kurtosis. It computes them as Viskores' `Statistics` filter does, with the
moments taken about the mean in a second pass.

<!-- tack-check: statistics -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((21, 21, 21), origin=(-1, -1, -1), spacing=(0.1, 0.1, 0.1))
r = np.linalg.norm(data.positions(), axis=1)
data.fields["r"] = td.Field(td.H1(data), tack.array(r))

s = td.statistics(data, "r")
print(f"n {s.n}, range {s.min:.3f} to {s.max:.3f}, mean {s.mean:.4f}, "
      f"std {s.sample_stddev:.4f}, skewness {s.skewness:.3f}")
# n 9261, range 0.000 to 1.732, mean 1.0077, std 0.2909, skewness -0.364
assert s.n == data.num_points
assert np.isclose(s.mean, r.mean(dtype=np.float64), rtol=1e-6)
assert np.isclose(s.sample_variance, r.astype(np.float64).var(ddof=1), rtol=1e-5)
```

The numbers are Python floats, so the call waits for the device work it
reads. Their definitions:

| Name | Is |
|---|---|
| `m2`, `m3`, `m4` | the sums of `(x - mean)^k` |
| `sample_variance`, `population_variance` | `m2 / (n - 1)` and `m2 / n` |
| `sample_stddev`, `population_stddev` | their square roots |
| `skewness` | `sqrt(n) m3 / m2^1.5`, 0 for a constant field |
| `kurtosis` | `n m4 / m2^2`, not the excess kurtosis, and 0 for a constant field |
| `unit` | the field's unit, which `mean`, `min` and the others are in |

Integer fields are summarized in floating point.

**Ghosts are left out.** Cells and points that the dataset's
[ghost marks](../reference/glossary.md#ghost-marks) mark as copies owned by another piece,
or as hidden, are not counted. A `PartitionedDataSet` is summarized whole:
`field` names a field every piece has, each piece reduces what it owns on its
own device, and the parts are combined in the pieces' order over every MPI
rank. The result is the same however the pieces are shared out among ranks.
`statistics` also takes AMR datasets, counting each region of space once at
its finest level, and octrees, counting their leaves
([adaptive data](adaptive-data.md)).

See the [reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.statistics.statistics)
for the full signature, and
[`Statistics`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.statistics.Statistics)
for every field it holds.

## Weight by volume or mass

`statistics` summarizes the stored values, each counted once. On a mesh whose
cells differ in size, that is not the mean over the volume: a region of many
small cells counts for more than a region of a few large ones. `weights`
weighs each value:

- `weights="volume"` weighs each cell by its measure. The field must be cell
  data (a [`Constant` field](../reference/glossary.md#constant)), whose values belong to
  the cells. Point data is
  refused.
- `weights=` a field's name weighs each value by that field, on the same
  entities. A density gives mass-weighted moments.

With weights, `mean` is `sum(w x) / W`, `sum` is `sum(w x)`, and `weight` is
`W`, the sum of the weights. The example's grid is fine near `x = 0` and coarse
toward `x = 8`:

<!-- tack-check: weights -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

axis = np.concatenate([np.linspace(0, 1, 11), [2.0, 4.0, 8.0]])   # graded along x
data = td.rectilinear_grid(axis, [0.0, 1.0], [0.0, 1.0], unit="m")
data.fields["T"] = td.Field(td.H1(data), tack.array(data.positions()[:, 0].copy()), unit="K")
data.fields["T_cell"] = td.project(data, "T", td.Constant(data))   # each cell's average

samples = td.statistics(data, "T")                                # the points, each once
by_volume = td.statistics(data, "T_cell", weights="volume")
print(f"mean of the samples {samples.mean:.3f} K, over the volume {by_volume.mean:.3f} K")
# mean of the samples 1.393 K, over the volume 4.000 K
totals = td.integrate(data, ["T"])
assert np.isclose(by_volume.mean, totals.values["T"] / totals.measure)
assert np.isclose(by_volume.weight, 8.0) and by_volume.weight_unit == "m**3"
```

The volume-weighted mean equals the integral over the measure, which
[`integrate`](derive.md#integrate-fields) gives directly. With weights the
population variance is `m2 / W`, and the sample variance treats the weights
as reliability weights, as NumPy's `aweights` does.

## Measure how a field's values spread

`entropy(data, field, bins=10)` is the Shannon entropy, in bits, of the field's
histogram: `-sum(p log2 p)` over the shares of the values in `bins` equal bins
over their range, as Viskores' `Entropy` filter computes it. A constant field
has entropy 0, and values spread evenly over `2^k` bins have entropy `k`.

<!-- tack-check: entropy -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

values = np.random.default_rng(0).uniform(size=100_000).astype(np.float32)
cloud = td.point_cloud(np.c_[values, values, values])
cloud.fields["u"] = td.Field(td.H1(cloud), tack.array(values))

bits = td.entropy(cloud, "u", bins=8)
print(f"{bits:.3f} bits")                      # 3.000 bits: 8 equally full bins
assert abs(bits - 3.0) < 0.01
```

`entropy` leaves out ghosts, combines pieces and takes AMR datasets and
octrees as `statistics` does. Over pieces, the bins span the whole dataset's
range. `weights` works as for `statistics`: each bin's share is the sum of its
weights over the sum of all of them. See the
[reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.entropy.entropy).

## Label connected regions

`connected_components(data)` finds which cells are connected through shared
points, as VTK's `vtkConnectivityFilter` does with all regions extracted. It
returns the dataset with two new fields: `region` on the cells and
`point_region` on the points. Regions are numbered from 0 in the order of
their lowest cell id. A point no cell uses, and a hidden cell, get -1.

To pick one region, threshold `region`:

<!-- tack-check: regions -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

data = td.uniform_grid((21, 21, 21), origin=(-1, -1, -1), spacing=(0.1, 0.1, 0.1))
x = data.positions()
to_nearer_centre = np.minimum(np.linalg.norm(x - [0.5, 0, 0], axis=1),
                              np.linalg.norm(x + [0.5, 0, 0], axis=1))
data.fields["d"] = td.Field(td.H1(data), tack.array(to_nearer_centre))

blobs = td.threshold(data, "d", 0.0, 0.3)               # two separate balls of cells
labelled = td.connected_components(blobs)
regions, sizes = np.unique(labelled.fields["region"].values.numpy(), return_counts=True)
print(dict(zip(regions.tolist(), sizes.tolist())))      # {0: 48, 1: 48}

largest = regions[np.argmax(sizes)]
one_ball = td.threshold(labelled, "region", largest, largest)
assert len(regions) == 2 and one_ball.num_cells == sizes.max()
```

It runs on the device, by union-find over the cells and their points. The
regions are the same on every device and every run, since each is numbered by
its lowest cell.

### Cluster points by distance

With `radius`, points within that distance of each other are connected too.
On a point cloud, whose cells are its points, that is Euclidean clustering, as
VTK's `vtkEuclideanClusterExtraction` does it. With a linking length it is
friends-of-friends halo finding. `min_size` gives the regions of fewer points
the label -1, and numbers the rest again in the same order.

<!-- tack-check: clusters -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

rng = np.random.default_rng(0)
points = np.concatenate([rng.normal(0.0, 0.05, (200, 3)),       # a cluster of 200
                         rng.normal(1.0, 0.05, (300, 3)),       # a cluster of 300
                         [[5.0, 5.0, 5.0]]]).astype(np.float32)  # a stray point
cloud = td.point_cloud(points)

clusters = td.connected_components(cloud, radius=0.1, min_size=10)
labels, sizes = np.unique(clusters.fields["region"].values.numpy(), return_counts=True)
print(dict(zip(labels.tolist(), sizes.tolist())))      # {-1: 1, 0: 200, 1: 300}
assert dict(zip(labels.tolist(), sizes.tolist())) == {-1: 1, 0: 200, 1: 300}
```

`periodic=` joins points across the faces of a periodic box, as a cosmology
box needs. The [points and particles page](points-and-particles.md) covers
clustering and halo catalogues in full.

### Regions over pieces

Over a `PartitionedDataSet`, regions are numbered across every piece of every
MPI rank. Each piece labels its own cells, and pieces join the regions that
meet at the points they share. With global ids on every piece, regions are
numbered by their lowest global cell id, which is the whole dataset's
numbering. Without them, they are numbered by their lowest (piece, cell).
[Filters over pieces](over-pieces.md) explains pieces, ghost cells and global
ids. See the
[reference](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.connected_components.connected_components).

## Details

**Pieces give the whole's answer.** Where every shared cell and point has one
owner, as the ghost marks arrange, statistics over pieces equal the whole
dataset's up to rounding. The order of combination is fixed by the pieces'
order, not by which rank finished first.

**Stored values, not functions.** `statistics` and `entropy` count the values a
field stores. For an [`H1` field](../reference/glossary.md#h1) of order 1 (point data), those
are the point values. For a
field of higher order, they are its node values, or a Bernstein basis's
coefficients, and neither is a set of samples spread evenly over the cells.
For integrals and true averages of any order, use
[`integrate`](derive.md#integrate-fields) or
[`project`](derive.md#move-fields-between-point-and-cell-data) into cell data.

## Where to go next

- [Derive quantities](derive.md): integrals, cell measures and the fields
  worth summarizing.
- [Points and particles](points-and-particles.md): outliers, downsampling,
  deposits and halo catalogues.
- [Filters over pieces](over-pieces.md): how reductions combine pieces over
  MPI ranks.

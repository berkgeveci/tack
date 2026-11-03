# Adaptive data

This page is for anyone who analyses the output of an adaptive simulation:
block-structured AMR from codes such as those built on AMReX, and tree-based
AMR from codes such as RAMSES. It shows how to filter each box of an AMR
hierarchy, add ghost cells, average fine data down to the coarse levels,
resample onto an image and reduce with weights. For octrees it shows how to
turn the leaves into cells and probe them. After it, you can analyse adaptive
data at the finest resolution it has, counting each region of space once.

Tack has two adaptive models. An [AMR dataset](../reference/glossary.md) is
levels of structured boxes, each level finer than the one before, as VTK's
`vtkOverlappingAMR`. An [octree](../reference/glossary.md) is a grid of
trees whose leaves are cells of different sizes, as VTK's
`vtkHyperTreeGrid`. [AMR hierarchies and octrees](../data/amr-and-octrees.md)
explains both, including the ghost marks that say which cells a finer level
covers.

```python
--8<-- "docs/examples/vis2_amr.py:pulse"
```

```text
level 0: 25 cells
level 1: 24 cells
level 1: 24 cells
integral 0.6633 over an area of 6.25
```

The integral counts each region once, at its finest level. The root image
covers 6.25 square units, and the two fine boxes replace the parts of it they
refine.

## The algorithms

| Task | Algorithm | Takes | VTK or other counterpart |
|---|---|---|---|
| Filter each box | any filter, box by box | a box | the same filter on each block |
| Grow each box by ghost cells | [`ghost_cells`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.ghost_cells.ghost_cells) | AMR | AMReX's `FillBoundary`, `FillPatchTwoLevels` |
| Refresh ghost values | [`update_ghosts`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.ghost_cells.update_ghosts) | AMR | none |
| Average fine data into coarse cells | [`average_down`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.amr_ghosts.average_down) | AMR | AMReX's `average_down` |
| Probe or resample | [`probe`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.probe.probe) | AMR, octree | `vtkAMRResampleFilter`, `vtkHyperTreeGridProbeFilter` |
| Summarize a field | [`statistics`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.statistics.statistics), [`entropy`](../vis-guide/reference/algorithms.md#tack.dataset.algorithms.entropy.entropy) | AMR, octree | `vtkDescriptiveStatistics`; Viskores' `Statistics`, `Entropy` |
| Leaves as cells | [`OctreeDataSet.leaves`](../vis-guide/reference/data-model.md#tack.dataset.OctreeDataSet) | octree | `vtkHyperTreeGridToUnstructuredGrid` |

## Filter each box

An AMR dataset is not a dataset. It is a container of boxes, and iterating
over it gives `(level, box)` pairs, coarsest level first. Each box is an
ordinary dataset, so every filter runs on it. Filters leave out the cells a
finer level covers, which the hierarchy marks `REFINEDCELL`. Their data are
valid but superseded.

This hierarchy puts a refined box across the seam of two root boxes. The
scalar's peak lies inside the refined region:

```python
--8<-- "docs/examples/vis2_amr.py:boxes"
```

```python
--8<-- "docs/examples/vis2_amr.py:filter"
```

```text
level 0: 32 cells, 8 refined
level 0: 32 cells, 8 refined
level 1: 64 cells, 0 refined
[0, 0, 28] segments
```

The root boxes' contours are empty, because the 0.5 level lies where the fine
box refines them. Each box keeps its `offset`, the index of its first cell in
its level, which places it in the hierarchy. Its geometry may be an image, a
rectilinear grid or a curvilinear grid.

## Add ghost cells and average down

A filter that reads each point's neighbours, such as point gradients, needs
the cells around a box's edge. `ghost_cells(amr, layers=1)` grows every box by
a layer of cells within its level's domain. Each ghost value comes from the
best source the hierarchy has:

- A ghost a sibling box holds is copied from it.
- A ghost no sibling holds comes from the coarser level. A cell takes its
  coarse cell's value, and a point interpolates the coarse cell's corners
  linearly.
- A ghost neither holds is hidden.

Positions are filled the same way, so rectilinear and curvilinear boxes grow
as images do. `update_ghosts(amr, "name")` refreshes the ghost values after a
filter changed the owned ones. `average_down(amr, "name")` puts the finer
levels' data into the coarse cells and points they cover, in place:

```python
--8<-- "docs/examples/vis2_amr.py:ghosts"
```

`average_down` gives each refined coarse cell the volume-weighted average of
the fine cells over it, and each refined coarse point the value of the fine
point at its location. It runs finest level first, so a value reaches the
coarsest level in one call. Call `update_ghosts` after it, since the ghost
copies of the coarse values it changed are otherwise stale.

## Probe and resample

`probe(amr, where)` evaluates each point in the finest level's cell holding
it. Probing onto an image therefore resamples the hierarchy, as VTK's
`vtkAMRResampleFilter` does:

```python
--8<-- "docs/examples/vis2_amr.py:probe"
```

Inside the fine box, the resampled values are exactly those of probing the
fine box alone. `valid` is 0 where no box holds a point.

## Reduce with weights

`statistics` and `entropy` take an AMR dataset or an octree directly. They
leave out refined cells and points, so each region of space counts once, at
its finest level. They count values, though, and the cells of an adaptive
mesh differ in size. The plain mean of a field weighs a small dense cell like
a large empty one. `weights="volume"` weighs each cell by its measure, which
gives the field's average over the domain. Then `sum` is the integral and
`weight` the domain's measure, as the first example shows.

`weights` also takes a field's name. With a density field, a volume-weighted
sum is a mass, and a field of mass per cell gives mass-weighted moments.

## Summarize an octree

An octree comes from a simulation reader as cells with levels and positions,
or from VTK as a `vtkHyperTreeGrid`. `OctreeDataSet.from_cells` builds one from
each cell's level and a position inside it, as a RAMSES reader or yt gives
them, and adds the ancestors no cell gives. This octree refines one root cell
to level 4 near a dense blob:

<!-- tack-check: adaptive-octree -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Whole octs, as a RAMSES reader gives them: each cell's level and centre.
levels, centres = [], []


def refine(lo, size, level):
    mid = lo + size / 2
    if level < 2 or (level < 4 and np.linalg.norm(mid - 0.3) < 0.25):
        for child in range(8):
            offset = np.array([child & 1, (child >> 1) & 1, (child >> 2) & 1]) * size / 2
            refine(lo + offset, size / 2, level + 1)
    else:
        levels.append(level)
        centres.append(mid)


refine(np.zeros(3), 1.0, 0)
centres = np.array(centres)
rho = np.exp(-40 * ((centres - 0.3) ** 2).sum(axis=1))       # a dense blob
roots = td.uniform_grid((2, 2, 2))                            # one root cell, the unit cube
octree = td.OctreeDataSet.from_cells(roots, levels, centres, fields={"rho": rho})

plain = td.statistics(octree, "rho")
by_volume = td.statistics(octree, "rho", weights="volume")
print(f"mean {plain.mean:.4f}, by volume {by_volume.mean:.4f}, mass {by_volume.sum:.4f}")
leaf = octree.is_leaf
assert np.isclose(by_volume.sum, octree.fields["rho"].numpy()[leaf] @ octree.measures()[leaf])
assert np.isclose(by_volume.weight, 1.0) and plain.mean > by_volume.mean
```

The leaves of the refined region are small and dense, so the plain mean is
far above the volume-weighted one. Reductions of an octree read its leaves.
`level=` reduces the nodes of one level instead. `is_leaf` and `measures()`
give each node's role and volume as NumPy arrays, for checks of your own.

## Turn octree leaves into cells and probe them

`octree.leaves()` gives the leaves as an unstructured dataset of voxels, or of
pixels in 2D, with every field as cell data. Cell `c` is the `c`-th leaf in
the nodes' order, `np.flatnonzero(octree.is_leaf)[c]`. Corners that coincide
are one point, so the cells are connected. Corners on a finer neighbour's
face stay hanging, as the tree has them. Every filter that reads cell data
works on the result: `threshold`, `cell_centers`, `extract_cells`, `probe`.

`probe(octree, points)` gives each point the values of the leaf holding it, as
VTK's `vtkHyperTreeGridProbeFilter` does. Values are constant over a leaf.
`valid` is 0 outside the roots, on hidden leaves and off a 2D octree's plane.

<!-- tack-check: adaptive-leaves -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Two root cells side by side; the first refined once, one of its children again.
roots = td.uniform_grid((3, 2, 2))
tree = [0] + [0] * 8 + [0] * 8 + [1]
level = [0] + [1] * 8 + [2] * 8 + [0]
key = [0] + list(range(8)) + [(7 << 3) | c for c in range(8)] + [0]
octree = td.OctreeDataSet(roots, tree, level, key,
                          fields={"level": np.array(level, np.float32)})

leaves = octree.leaves()
print(leaves.num_cells, "cells,", leaves.num_points, "points")
assert leaves.num_cells == 16
# 27 corners in the refined root, 19 more in its refined child, 4 more for the
# coarse root beside it, whose shared face's corners are already there.
assert leaves.num_points == 27 + 19 + 4

probed = td.probe(octree, np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9],
                                    [1.5, 0.5, 0.5], [3.0, 0.5, 0.5]]))
print(probed.fields["level"].values.numpy(), probed.fields["valid"].values.numpy())
assert list(probed.fields["level"].values.numpy()) == [1, 2, 0, 0]
assert list(probed.fields["valid"].values.numpy()) == [1, 1, 1, 0]
```

The octree constructor takes each node's tree, level and Morton key, as the
[octree model](../data/amr-and-octrees.md) explains. A probe onto an image
resamples the octree.

## Over MPI ranks

The boxes of a hierarchy may live on several devices and several MPI ranks.
Each rank passes its own boxes of each level and the
[world](../reference/glossary.md):
`td.AMRDataSet(levels, ratios, world=tc.World(comm))`. The hierarchy gathers
every rank's boxes when it is made. Its marks, ghost cells, `average_down`,
probes and reductions then work across the ranks. An octree's ranks each hold
whole trees. Its reductions combine every rank's leaves, and a probe sends
each point to the rank whose trees hold it.
[Filters over pieces](over-pieces.md) covers worlds and MPI.

## Limits

- An AMR contour is one piece per box, and the pieces of different levels do
  not join at level boundaries. Crack-free contouring is not part of Tack
  yet.
- An octree's leaves keep their hanging corners. Face-based algorithms treat a
  coarse-fine interface as boundary, since such faces are not yet paired.
- An AMR dataset refuses boxes that overlap within a level, boxes not properly
  nested in the level below, image boxes off their level's lattice, and boxes
  that already have ghost cells.

## Where to go next

- [AMR hierarchies and octrees](../data/amr-and-octrees.md) explains the
  models, their marks and how to build them.
- [Sample and trace](sample-and-trace.md) covers probing in general.
- [VTK](vtk.md) converts `vtkOverlappingAMR` and `vtkHyperTreeGrid` both ways.

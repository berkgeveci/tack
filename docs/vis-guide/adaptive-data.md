# Analyze adaptive data

Choose [AMR boxes](10-amr.md) or [octree leaves](11-octrees.md) according to
the simulation's representation. A hierarchy's refined parents are not extra
physical volume. The following recipes construct data, check ownership and
compute results at the finest resolution available.

## Construct and filter an AMR hierarchy

This complete example puts a refined box across the seam of two coarse boxes.
The scalar peak lies inside the refined region, so its contour should come
from the fine box alone. The model explains [the refinement marks](10-amr.md).

<!-- tack-check: amr -->
```python
import numpy as np

import tack
import tack.dataset as td
from tack.dataset import Ghosts

tack.set_default_device("cpu")


def box(lo, cells, spacing):
    """An image box of ``cells`` cells, its first cell at index ``lo`` of its level."""
    corner = tuple(i * spacing for i in lo)
    grid = td.uniform_grid(tuple(n + 1 for n in cells), origin=corner, spacing=(spacing,) * 2)
    topology = td.StructuredTopology(tuple(n + 1 for n in cells), offset=lo)
    data = td.DataSet(topology, grid.geometry.values)
    x = data.positions()
    data.fields["f"] = td.Field(td.H1(data), tack.array(
        np.exp(-((x[:, 0] - 4) ** 2 + (x[:, 1] - 4) ** 2) / 4).astype(np.float32)))
    return data


# A root of two 4 x 8 boxes, and a box at level 1 across their seam (ratio 2).
amr = td.AMRDataSet([[box((0, 0), (4, 8), 1.0), box((4, 0), (4, 8), 1.0)],
                     [box((4, 4), (8, 8), 0.5)]], ratios=[2])
for level, piece in amr:
    refined = (piece.ghosts.cells.numpy() & Ghosts.REFINEDCELL) != 0
    print(f"level {level}: {piece.num_cells} cells, {refined.sum()} refined")

# Each location once, at the finest level holding it.
assert td.statistics(amr, "f").n == sum(
    ((p.ghosts.points.numpy() & Ghosts.SKIPPED_POINTS) == 0).sum() for _, p in amr)

# Filters run per box and leave refined cells out. The 0.5 contour lies where
# the fine box refines the root, so only the fine box gives segments.
lines = [td.contour(piece, "f", 0.5) for _, piece in amr]
print([line.num_cells for line in lines], "segments")
assert lines[0].num_cells == lines[1].num_cells == 0 and lines[2].num_cells > 0
```

The root boxes' contours are empty: the 0.5 level lies where the fine box
refines them. Pieces of a contour at different levels do not join at level
boundaries; crack-free contouring is not part of Tack yet.

## Ghost cells

`ghost_cells(amr, layers=1)` grows every box by a layer within its level's
domain. A ghost a sibling box holds is copied from it; one no sibling holds
comes from the coarser level, the coarse cell's value for a cell and an
interpolation of the coarse cell's corners for a point. A grid's positions are
filled the same way, so curvilinear and rectilinear boxes grow as images do.
A ghost neither holds is hidden. Afterwards, `update_ghosts(amr, "name")`
refreshes the ghosts after a filter changed the owned values, and
`average_down(amr, "name")` puts the fine levels' data into the coarse cells
and points they cover:

```python
grown = td.ghost_cells(amr)                # every box a layer wider
for level, box in grown:
    box.fields["g"] = td.gradients(box, "f", at="points")   # right where owned
td.update_ghosts(grown, "g")               # the ghosts' from their owners
td.average_down(grown, "g")                # coarse cells under fine ones: the fine average
```

## Probe and resample

`probe(amr, where)` evaluates each point in the finest level's cell holding it,
so probing onto an image resamples the hierarchy, as VTK's
`vtkAMRResampleFilter` does:

```python
image = td.uniform_grid((64, 64), origin=(0.0, 0.0), spacing=(8 / 63, 8 / 63, 1.0))
resampled = td.probe(amr, image)           # each point from the finest level holding it
```

## In pieces

The boxes of a hierarchy may be on several devices and spread over MPI ranks:
each rank passes its own boxes of each level and the world
(`td.AMRDataSet(levels, ratios, world=tc.World(comm))`). The hierarchy gathers
every rank's boxes when it is made, and its marks, ghost cells,
`average_down`, probes and reductions work across the ranks.


## Construct and summarize an octree

This example starts from cell levels and positions, as a simulation reader
might supply them. It compares sample and volume weighting independently.

<!-- tack-check: octree -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Cells as a RAMSES reader gives them: whole octs, each cell's level and centre.
# One root cell refined to level 2 everywhere, and to level 4 near a blob.
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
rho = np.exp(-40 * ((centres - 0.3) ** 2).sum(axis=1))      # a dense blob
roots = td.uniform_grid((2, 2, 2))                           # one root cell, the unit cube
octree = td.OctreeDataSet.from_cells(roots, levels, centres, fields={"rho": rho})
print(octree)

# Leaves differ in size: the plain mean of rho counts a small dense cell like a
# large empty one; the volume-weighted mean is rho's average over the cube.
plain = td.statistics(octree, "rho")
by_volume = td.statistics(octree, "rho", weights="volume")
print(f"mean rho {plain.mean:.4f}, by volume {by_volume.mean:.4f}, mass {by_volume.sum:.4f}")
leaf = octree.is_leaf
volume = octree.measures()[leaf]
assert np.isclose(by_volume.sum, octree.fields["rho"].numpy()[leaf] @ volume)
assert np.isclose(by_volume.weight, 1.0) and plain.mean > by_volume.mean
```

## Weights

An octree's leaves differ in size by a factor of 8 per level, so the plain
mean of a field over the leaves weighs a tiny dense cell like a huge empty
one. `weights="volume"` weighs each cell by its measure, and a field's name
(`weights="rho"`) by that field: with a density, the volume-weighted `sum` is
the mass and a mass-weighted mean takes `weights=` a field of mass per cell.
`entropy` takes the same weights, and so do datasets, pieces and AMR
hierarchies, whose levels differ in size too. `level=` reduces one level's
nodes instead of the leaves.

## Leaves as cells, and probes

`octree.leaves()` is the leaves as an unstructured `DataSet` of voxels (pixels
in 2D) with every field as cell data: cell `c` is the `c`-th leaf in the
nodes' order (`np.flatnonzero(octree.is_leaf)[c]`), and hidden leaves are
`HIDDENCELL`. Corners that coincide are one point, found by sorting their
indices on the finest level's lattice, so the cells are connected; corners on
a finer neighbour's face stay hanging, as the tree has them. Every filter that
reads cell data works on it: `threshold`, `cell_centers`, `extract_cells`,
`probe`.

`probe(octree, points)` gives each point the values of the leaf holding it,
found by one binary search over the keys per point: the point's key on the
finest level is compared with each node's key carried there, and in pre-order
the last node at or below it is the leaf. Values are constant over a leaf, as
VTK's `vtkHyperTreeGridProbeFilter` gives them; `valid` is 0 outside, on
hidden leaves and off a 2D octree's plane. A probe onto an image resamples the
octree. Over MPI ranks, each point goes to the rank whose trees hold it.

<!-- tack-check: octree-leaves -->
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
# coarse root beside it (its face's 4 corners are already there).
assert leaves.num_points == 27 + 19 + 4

# Each point takes its leaf's value.
probed = td.probe(octree, np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [1.5, 0.5, 0.5],
                                    [3.0, 0.5, 0.5]]))
print(probed.fields["level"].values.numpy(), probed.fields["valid"].values.numpy())
assert list(probed.fields["level"].values.numpy()) == [1, 2, 0, 0]
assert list(probed.fields["valid"].values.numpy()) == [1, 1, 1, 0]
```


## Import existing adaptive data

Use [Connect Simulation Data](connect-simulation-data.md#adaptive-data) for
VTK AMR and HyperTreeGrid entry points. Octree `from_cells` accepts root-grid,
level and position arrays; its [model page](11-octrees.md#from-codes-and-vtk)
describes missing ancestors. For complete exchanges and ghost ownership see
[Data in Pieces](07-partitioned.md), rather than inferring ownership from
the number of local boxes.

## Limits to check before interpreting results

AMR contour pieces can have cracks at level boundaries. Octree leaf meshes
retain hanging nodes; a coarse/fine interface is not yet a paired face for
face-based algorithms. Probing may be valid while conservative face exchange
is unsupported. See the [AMR limits](10-amr.md#not-here),
[octree limits](11-octrees.md#not-here-yet), and [capability catalog](16-cells-and-bases.md).

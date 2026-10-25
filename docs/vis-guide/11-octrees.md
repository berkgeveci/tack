# Octrees

Tree-based AMR codes -- RAMSES and its kind, and CEA's codes, which ParaView
reads as VTK's HyperTreeGrid -- refine cell by cell: each cell is split in two
along each axis wherever more resolution is needed, into a tree of cells. An
`OctreeDataSet` holds a grid of such trees, laid out for GPUs as a *linear
octree*: every node is three numbers -- its tree, its level, and its Morton
key, its integer coordinates in the tree interleaved bit by bit -- and the
nodes are sorted so each subtree is one contiguous run. Whether a node is a
leaf, and its position and size, come from the keys; nothing about position is
stored.

This first pass holds cell data (one value per node, refined ones included, as
the codes store them) and is about **statistics**. Nodes that are not leaves are
`REFINEDCELL`, so reductions count each region once, over the leaves.

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
leaves = octree.leaves()
volume = octree.measures()[leaves]
assert np.isclose(by_volume.sum, octree.fields["rho"].numpy()[leaves] @ volume)
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

## From codes and VTK

- `OctreeDataSet.from_cells(roots, level, position, fields)` takes each cell's
  level and a position in it, as a RAMSES reader or yt gives them (whole octs),
  over a grid of root cells (an image or a rectilinear grid); it adds the
  ancestors no cell gives, their values NaN.
- `vtk_to_dataset` takes a `vtkHyperTreeGrid` (branching factor 2) and
  `dataset_to_vtk` gives one back, a mask as `HIDDENCELL`.

## Not here yet

Leaves as an unstructured grid (so every cell-data filter works on them),
probing, point data and contours (the dual grid) come later.

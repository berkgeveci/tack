# AMR hierarchies

Block-structured adaptive mesh refinement covers a domain with boxes, and
some regions with finer boxes, level after level. An `AMRDataSet` holds the
levels, each a `PartitionedDataSet` of structured boxes, and the refinement
ratio between them. It follows VTK's `vtkOverlappingAMR` and AMReX's design:
the hierarchy is defined in index space alone -- a box is its first cell's
index in its level (its topology's `offset`) and its cell counts -- so a box
can be an image, a rectilinear grid or a curvilinear grid.

Making the hierarchy marks what a finer level supersedes:

- `REFINEDCELL` on each cell a finer box covers;
- `REFINEDPOINT` on each point at whose location a finer level has a point,
  the edge of a refined region included;
- `DUPLICATEPOINT` on points two sibling boxes share, owned by the first.

So every region and location belongs to one box, at the finest level holding
it, and reductions over the hierarchy count it once. Filters run per box, and
those that make cells leave refined cells out.

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

## Requirements

Boxes of a level may not overlap, and each fine box, coarsened, must lie
within its coarser level's boxes (proper nesting). A level's image boxes
share one spacing and lie on one lattice, as VTK's model has it; rectilinear
and curvilinear boxes have no such constraint. Ratios may differ per axis.

## VTK

`vtk_to_dataset` takes a `vtkOverlappingAMR`, and `dataset_to_vtk` gives one
back for images and rectilinear grids; a hierarchy with curvilinear boxes goes
to VTK as a `vtkPartitionedDataSetCollection` of a partitioned dataset per
level. `amr_gaussian_pulse` makes VTK's Gaussian pulse hierarchy, for tests.

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

## Not here

Crack-free contouring across levels, periodic domains, tagging and regridding,
and conservative interpolation are not part of Tack yet.

# Work with data in pieces

Large data often arrives in pieces: a reader's partitions, a simulation's
blocks, the share of each GPU or MPI rank. Tack keeps each piece a whole
`DataSet` and adds three things: a container for the pieces, marks saying
which cells and points another piece owns, and reductions that combine the
pieces into the whole's answer.

- **A `PartitionedDataSet`** is an ordered list of datasets, as VTK's
  `vtkPartitionedDataSet`. Each piece has its own topology, numbering,
  geometry, fields, sets and device. Iterate over it, index it, take its
  length.
- **Work on one piece takes the piece.** Filters, analyses and `for_each`
  take a `DataSet`, as everywhere else, and you loop over the pieces. There
  is no "apply to every piece" wrapper: the loop says what it does.
- **Only algorithms whose answer needs every piece take the container:**
  `statistics` and `entropy`, given a field's name.

## Cut a grid into pieces

`split` cuts a structured grid by index ranges. `ghost_layers=1` adds a
layer of each neighbour's cells around every piece:

```python
--8<-- "docs/examples/data_partitioned.py:split"
```

Each piece of this 20 × 20 × 20-cell grid owns 2000 cells and holds 420
more from its neighbours. Every cell and point that several pieces hold has
exactly **one owner**; in every other piece its copy is a **ghost**. The
owned cells of the four pieces are the grid's 8000 cells, and the owned
points its 9261 points.

An unstructured mesh is cut by VTK or by the application that made it (see
[Pieces from VTK](#pieces-from-vtk) below).

## Ghost marks

`data.ghosts` is a `tack.dataset.Ghosts`, or `None` for data with no
pieces. It holds one `u8` per cell (`ghosts.cells`) and one per point
(`ghosts.points`), either of which may be `None`, with VTK's
`vtkGhostType` bits:

| Bit | Value | Meaning |
|---|---|---|
| `Ghosts.DUPLICATECELL` | 1 | a copy of a cell another piece owns |
| `Ghosts.HIDDENCELL` | 32 | a cell whose data is not valid here: an AMR cell a finer level covers, a blanked cell |
| `Ghosts.DUPLICATEPOINT` | 1 | a copy of a point another piece owns |
| `Ghosts.HIDDENPOINT` | 2 | a point whose data is not valid here |

The other cell bits (`HIGHCONNECTIVITYCELL`, `LOWCONNECTIVITYCELL`,
`REFINEDCELL`, `EXTERIORCELL`) are carried along unread.
`Ghosts.SKIPPED_CELLS` and `Ghosts.SKIPPED_POINTS` are the bits a reduction
leaves out. The marks are kept apart from the fields, as the geometry is,
so no algorithm mistakes them for data. `to_device` moves them;
`vtk_to_dataset` and `dataset_to_vtk` map them to and from `vtkGhostType`
arrays. Set them yourself with `data.ghosts = td.Ghosts(cells=..., points=...)`;
marks of the wrong length or type are refused.

## Work on each piece

Loop over the pieces:

```python
--8<-- "docs/examples/data_partitioned.py:loop"
```

The ghost layer is why this works near a piece's edge. A point gradient at
the edge of a piece averages over every cell around the point, and the
ghost layer supplies the cells across the cut, so each piece's gradients at
its own points are the whole grid's.

Filters process the ghost cells too, and **carry the marks** to what they
make, as VTK's do. What a filter makes takes the owner of what it came from:

| Output | Owner |
|---|---|
| a cell (a contour's triangles, a clip's pieces, a kept cell) | its source cell's |
| a point a filter keeps | its own |
| a point on an input edge (a contour's, a clip's crossings) | the edge's: the owner of its smallest point |
| a point inside a cell (a clip's centroids) | the cell's |
| a surface face | the cell it is a side of |
| a face two cells share, an edge, a vertex | the owner of its smallest point |

"Smallest" goes by position, so every piece that holds a shared edge or
face decides alike, with no communication. With a ghost layer, the piece
that owns a point holds every cell around it. So **the owned part of a
filter's output, over all pieces, is the whole's output**, cell for cell
and point for point. That holds for every filter that makes cells or
points, chained filters included. The ghost output itself is what the
next step near the edge needs: point normals, smoothing, a second
filter that looks at neighbours.

## Reduce over the pieces

```python
--8<-- "docs/examples/data_partitioned.py:reduce"
```

`statistics(pieces, name)` takes each piece's moments on its own device and
combines them pairwise; `entropy(pieces, name, bins)` sums each piece's
histogram over the whole's range. Both count only what each piece owns
(cells and points not marked duplicate or hidden), on a single dataset too.
So the statistics of the pieces' isosurfaces are the whole grid's
isosurface's, and so is the owned triangle count.

An `L2` field goes by its cells' marks. A field on faces or edges, or of
order 2, is refused on a dataset with marks, since who owns those values is
not decided yet.

## Remove the ghosts at the end

Before rendering, writing one file, or handing a piece to code that does
not know ghost marks, take the ghost output away:

```python
--8<-- "docs/examples/data_partitioned.py:remove"
```

`remove_ghosts` drops the duplicate and hidden cells, and the points no
remaining cell uses, as VTK's `vtkRemoveGhosts`. The points it keeps keep
their marks: a point a piece shares is still its owner's, so reductions
over the result are unchanged. A point the piece owns stays even when only
ghost cells used it. A contour point on an edge belongs to the owner of
the edge's smallest point, and that piece's triangles around the edge may
all be ghosts. A structured piece whose remaining cells form a block stays
a structured grid. `dataset_to_vtk` keeps the marks, so pieces with ghosts
can also go to VTK as they are; VTK leaves ghosts out when it renders.

## Pieces need ghost cells

Without ghost cells, a piece cannot tell what its neighbours make or keep
from the cells, faces and edges they share. Take a contour point on an edge
in the cut: it belongs to the owner of the edge's smallest point, and that
piece may have no cell containing the edge. Or a shared point only a
neighbour's kept cells use: it disappears from its owner's threshold. Tack
does not guess:

```python
--8<-- "docs/examples/data_partitioned.py:no-ghosts"
```

Such outputs record that they lost the marks, and `statistics` and
`entropy` refuse them with `NotImplementedError`, naming the cause. What
does not depend on neighbours still works on pieces without ghost cells:

- reductions of the pieces themselves;
- filters that keep the cells and points (`warp`, `transform`,
  `cylindrical`, `spherical`, `shrink`, `as_polyhedra`, `as_polygons`, and
  `probe`'s target);
- points chosen by their own values (`threshold_points`, `extract_points`,
  `mask_points`).

Setting an output's `ghosts` clears the record.

## Hidden cells

Filters never process hidden cells: their data is not valid there. Every
filter that makes cells or points leaves them out, as if they were not
there:

- `contour`, `slice`, `clip`, `threshold`, `extract_cells`,
  `extract_geometry`, `mask`, `extract_surface`, `extract_edges`,
  `triangulate`, `tetrahedralize`;
- the source dataset of `probe`, `advect` and `streamlines`.

A field passed as an object rather than by name follows along. Two things
cannot leave hidden cells out yet and are refused on such data: faces or
edges chosen by id, and a reused `CellLocator`. Algorithms whose output is
on the same cells (gradients, projections, calculators) still compute
there, and the marks stay on those cells.

## Pieces from VTK

`vtk_to_dataset` turns a `vtkPartitionedDataSet` into a
`PartitionedDataSet`, and a `vtkPartitionedDataSetCollection` into a `dict`
of names to partitioned datasets; `dataset_to_vtk` does the reverse. To cut
an unstructured mesh with ghost cells, use VTK's filters (no MPI needed):

```python
from vtkmodules.vtkFiltersParallelDIY2 import (
    vtkGhostCellsGenerator,
    vtkRedistributeDataSetFilter,
)

from tack.interop.vtk import vtk_to_dataset

cut = (vtkRedistributeDataSetFilter(number_of_partitions=4,
                                    preserve_partitions_in_output=True)
       >> vtkGhostCellsGenerator(number_of_ghost_layers=1, build_if_required=False))
pieces = vtk_to_dataset(cut(mesh))                 # mesh: any vtkDataSet
```

The filters' rules hold over these pieces as over `split`'s.

## Generate ghost cells

Unstructured pieces without ghost cells, from VTK's redistribution, a
reader or your own partitioner, get a layer from `ghost_cells`, as VTK's
`vtkGhostCellsGenerator` would give them, marks and values alike:

```python
plain = vtk_to_dataset(vtkRedistributeDataSetFilter(number_of_partitions=4,
                                                    preserve_partitions_in_output=True)(mesh))
pieces = td.ghost_cells(plain)                     # one layer, marked
deeper = td.ghost_cells(plain, layers=2)           # two, for two neighbour filters in a row
```

Structured pieces, `split`'s or VTK's image pieces, stay structured: each
becomes its block grown by the layers, with the right offset and
coordinates.

Pieces neighbour where they share points at exactly the same position, and
a shared point belongs to the lowest piece holding it. The pieces may be on
different devices, or on different MPI ranks (each rank passes its own
pieces and a `tack.communication.World`; see the API reference).

A filter that needs each entity's neighbours is right on what a piece owns
but wrong on the rim of the ghost layer, whose neighbours the piece lacks.
Before a second such filter reads those values, copy them from their
owners:

```python
for piece in pieces:
    piece.fields["p"] = td.project(piece, "c", td.H1(piece))  # a cell field "c" to points
td.update_ghosts(pieces, "p")                      # ghost values from their owners
for piece in pieces:
    piece.fields["grad"] = td.gradients(piece, "p", at="points")   # right where owned
```

`ghost_cells` keeps the exchanges, so each `update_ghosts` is one copy per
neighbour, not a new matching.

## Global ids

Each piece numbers its points and cells from 0. Global ids number them
across all the pieces, one id per point and per cell wherever it is held:
`piece.global_ids.points` and `.cells`, `i64` arrays (`td.GlobalIds`).
VTK's `GlobalIds` arrays come in as them, and `generate_global_ids` makes
them, as `vtkGenerateGlobalIds` does:

```python
pieces = td.generate_global_ids(td.ghost_cells(plain))
ids = pieces[0].global_ids.points                  # the same at a shared point
```

Filters keep the ids of the points and cells they keep (`threshold`,
`extract_cells`, `remove_ghosts`); what they make has none.

## Pieces on several devices

A piece is a dataset, so it can live on any device:
`td.PartitionedDataSet(piece.to_device(f"cuda:{i % 2}") for i, piece in
enumerate(pieces))` spreads them over two GPUs. Each piece's work runs on
its own device, and `statistics` and `entropy` reduce each piece where it
is. `packages/tack-vis/examples/44_partitioned_data.py` does this when there
are several devices of the chosen type.

## Particles across pieces

`advect` and `streamlines` take a partitioned dataset too, and particles
cross from piece to piece:

```python
pieces = td.split(data, 8, ghost_layers=1)
particles = td.advect(pieces, "velocity", seeds, 0.01, 500)  # a dataset per piece
paths = td.streamlines(pieces, "velocity", seeds, 0.01, 500) # each piece's stretches
```

With a ghost layer wider than a step (`step_size` times the speed), the
paths are the whole mesh's up to rounding: a particle that ends a step in a
ghost cell is handed to the cell's owner, which goes on with whole steps.
Without ghost cells it crosses at the boundary by a shortened step. Each
particle is in one piece of the result, with its `seed`; a path's stretches
are in the pieces that traced them, each after the first starting at a
point marked `DUPLICATEPOINT`.

## Probing pieces

`probe` takes a partitioned source: each point is evaluated by the piece
that holds it, and the values come back in your points' order, equal to
probing the whole dataset. Ghost cells are not needed for it.

```python
values = td.probe(pieces, points)                  # points: an array or a dataset
values.fields["RTData"], values.fields["valid"]    # valid 0 outside every piece
```

## Connected regions across pieces

`connected_components` labels cells connected through shared points; given
pieces, it numbers the regions across all of them, so a region that crosses
a cut has one number everywhere:

```python
regions = td.connected_components(pieces)          # region, point_region on each piece
```

With global ids on the pieces, regions are numbered as on the whole dataset.
`min_size` drops regions of fewer points, each point counted once.

## Point clouds in pieces

A point cloud's pieces share no points, so a filter over a point's
neighbours needs copies of the points near it that other pieces hold.
`ghost_points` appends them, marked as ghosts, either every point within a
radius of the piece or exactly those among its points' k nearest. Then the
neighbour filters run per piece, as mesh filters do after `ghost_cells`, and
each owned point gets the whole's answer:

```python
ghosted = td.ghost_points(pieces, k=10)
normals = [td.point_normals(piece, k=10) for piece in ghosted]

ghosted = td.ghost_points(pieces, radius=2 * h_max)            # an SPH kernel's reach
for piece in ghosted:
    piece.fields["density"] = td.sph_density(piece).fields["density"]
td.update_ghosts(ghosted, "density")                           # the copies' from their owners
```

Clusters, statistical outliers, voxel downsampling, deposition onto a grid
and halo catalogues need every piece, so they take the pieces, here or across
MPI ranks:

```python
halos = td.connected_components(pieces, radius=linking, min_size=100, periodic=box)
catalog = td.halo_catalog(halos, mass=particle_mass, velocity="v", periodic=box)
density = td.deposit(pieces, grid, method="cic")               # the whole grid, on every rank
```

In a periodic box the copies across a face are images, moved by the box's
side, so run the per-piece filters on `ghost_points`' pieces without
`periodic`.

## One dataset again

`merge_pieces` gives the pieces back as one dataset, each cell and point
once, ghosts dropped: to write one file, or to hand to code that knows no
pieces. Structured pieces come back as one structured grid; across MPI ranks
the result is on the root rank.

```python
whole = td.merge_pieces(pieces)                    # None on the other ranks
```

## Not here yet

Asynchronous exchange, should particles across MPI ranks need it, and
rebalancing pieces are not planned for now.

## Run and check

```bash
uv run python docs/examples/data_partitioned.py --check
uv run python packages/tack-vis/examples/44_partitioned_data.py --pieces 8
```

The checks compare the pieces against the whole grid: owned cells and
points, statistics of the grid and of its gradients, and the isosurface's
triangles and points, before and after `remove_ghosts`.
[Download the complete program](../examples/data_partitioned.py).

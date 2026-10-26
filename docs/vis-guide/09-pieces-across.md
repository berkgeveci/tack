# Algorithms across pieces

[Work with Data in Pieces](07-partitioned.md) cuts a grid, marks the ghosts
and loops over the pieces in one process. This chapter gets pieces from
elsewhere -- VTK, a partitioner, other devices, other MPI ranks -- and
covers the algorithms whose answer crosses pieces: ghost generation, global
ids, particles, probes, connected regions and point clouds.

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

## Across MPI ranks

Over MPI each rank holds its own pieces, and the container says where the
others are: a `tack.communication.World` made from an mpi4py communicator.
The algorithms that take the container then communicate; the per-piece
loops do not change.

```python
from mpi4py import MPI

import tack.dataset as td
from tack import communication as tc

world = tc.World(MPI.COMM_WORLD)
every = td.split(data, 8, ghost_layers=1)                  # or each rank reads its own
mine = [every[i] for i in range(world.rank, 8, world.size)]
pieces = td.PartitionedDataSet(mine, world)
print(td.statistics(pieces, "RTData").mean)                # over all eight pieces
```

```bash
mpiexec -n 4 python script.py
```

Tack imports no MPI module: you make the communicator, and mpi4py must be
built against your MPI (`uv sync --extra mpi`). Results that sum over pieces
add in the pieces' global order, so they have the same bits however the
pieces are shared out. With an MPI that reads device memory,
`tc.World(comm, gpu_aware=True)` sends device arrays without staging them
through the host; make the rank's device (`tack.device("cuda:N")`) before
importing `mpi4py.MPI`, which starts MPI.

Every algorithm here is written on `tack.communication`, and so can one of
your own: [Communicate Between Blocks](14-communication.md).

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
`ghost_points` appends them, marked as ghosts: every point within a radius
of the piece (`radius=`), or exactly those among its owned points' k
nearest (`k=`). The neighbour filters then run per piece, as mesh filters
do after `ghost_cells`, and each owned point gets the whole's answer.
Clusters, statistical outliers, voxel downsampling, deposition onto a grid
and halo catalogues need every piece, so they take the pieces:

<!-- tack-check: point-pieces -->
```python
import numpy as np

import tack
import tack.dataset as td

tack.set_default_device("cpu")

# Clumps in a periodic box, and the same cloud cut into four pieces.
rng = np.random.default_rng(4)
box = 10.0
x = np.concatenate([c + rng.normal(0.0, 0.3, (400, 3)) for c in rng.uniform(1, 9, (12, 3))]
                   + [rng.uniform(0.0, box, (2000, 3))]) % box
x = x.astype(np.float32)
whole = td.point_cloud(x, origin=(0.0, 0.0, 0.0))
pieces = td.PartitionedDataSet([td.point_cloud(part, origin=(0.0, 0.0, 0.0))
                                for part in np.array_split(x, 4)])

# Per piece, after copying in each owned point's 10 nearest from elsewhere.
ghosted = td.ghost_points(pieces, k=10)
normals = np.concatenate([td.point_normals(piece, k=10).fields["normals"].values.numpy()
                          [:len(own)] for piece, own in zip(ghosted, np.array_split(x, 4))])
expected = td.point_normals(whole, k=10).fields["normals"].values.numpy()
assert np.abs(normals - expected).max() < 1e-3

# Over the pieces: the whole's clusters, numbered as the whole numbers them.
groups = td.connected_components(pieces, radius=0.15, min_size=50, periodic=box)
labels = np.concatenate([p.fields["point_region"].values.numpy() for p in groups])
assert np.array_equal(labels, td.connected_components(
    whole, radius=0.15, min_size=50, periodic=box).fields["point_region"].values.numpy())
print(labels.max() + 1, "clusters")
```

After a filter that is right only where a piece owns its points,
`update_ghosts` brings the copies' values from their owners: SPH densities
before an interpolation that reads them, for instance.

```python
ghosted = td.ghost_points(pieces, radius=2 * h_max)        # an SPH kernel's reach
for piece in ghosted:
    piece.fields["density"] = td.sph_density(piece).fields["density"]
td.update_ghosts(ghosted, "density")                       # the copies' from their owners
```

`deposit` of pieces gives every rank the whole grid, and `halo_catalog` of
pieces a catalogue piece per piece, the halos in label order across them.
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
uv run python docs/examples/validate.py --arch cpu
uv run python packages/tack-vis/examples/44_partitioned_data.py --pieces 8
```

`validate.py` runs the point-cloud example above, comparing the pieces'
normals and clusters with the whole cloud's; the test suite compares every
algorithm here with the whole, on two devices and on two and three MPI
ranks.

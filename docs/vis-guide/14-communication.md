# Pieces and blocks

The algorithms that take a `PartitionedDataSet` are written on
`tack.communication`, the layer in `tack` for moving arrays between *blocks*
and combining values over them. The User's Guide's
[Communicate Between Blocks](../users-guide/18-communication.md) explains it
with plain arrays: blocks and worlds, collectives, exchanges, `Plan`,
`Routing`, `sort_by_key` and MPI. This page is about using it with a
dataset in pieces, for an algorithm of your own whose answer on a piece
needs another piece's data.

## Blocks from a dataset's pieces

An algorithm over a `PartitionedDataSet` makes one block per piece, in the
pieces' order, in the dataset's world:

```python
from tack import communication as tc

blocks = tc.Blocks([piece.device for piece in pieces], pieces.world)
mine = dict(zip(blocks.local, pieces))           # global id -> this rank's piece
```

A piece's global id is then its index among every rank's pieces, as VTK
numbers a partitioned dataset's pieces, and the numbering that ghost marks,
global ids and `statistics` use. Positions are relative to each piece's
origin, so compare boxes and points across pieces only where the pieces
share one, as Tack's algorithms check they do (`partitioned.common_origin`).

## What the built-in algorithms use

Each pattern of the User's Guide chapter has built-ins written on it, which
are worth reading before writing one like them:

| Pattern | Built-ins |
|---|---|
| Collectives (`all_reduce`, `all_gather`, `exclusive_scan`) | `statistics` and `entropy` over pieces, `generate_global_ids`' numbering |
| `Plan` | `ghost_cells`' and `ghost_points`' plans, which `update_ghosts` reruns; `generate_global_ids`' ids for copies, from their owners; AMR ghost cells and `average_down` |
| `Routing` | `probe` across pieces (the cell that holds a point outranks one within tolerance); particle advection and streamlines, in rounds |
| `sort_by_key` | ghost cells' boundary-point matching, `generate_global_ids`, `merge_pieces`, `voxel_downsample`, `halo_catalog`, `connected_components` over pieces |

Results that combine pieces do it in the pieces' global order, so they have
the same bits however the pieces are shared out over ranks.
[Algorithms Across Pieces](09-pieces-across.md) runs them, and
[Write a Filter](15-write-a-filter.md) shows a reduction over pieces written
with public names.

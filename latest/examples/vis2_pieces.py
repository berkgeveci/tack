# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Filters over pieces: ghost cells, reductions, global ids, merging, probes and
particles across pieces, checked against the whole dataset."""

import argparse


def owned(marks, n, skipped):
    """How many of n cells or points a piece owns: those not marked duplicate or hidden."""
    return n if marks is None else int(((marks.numpy() & skipped) == 0).sum())


def run(arch="cpu"):
    # --8<-- [start:first]
    import numpy as np

    import tack
    import tack.dataset as td

    tack.set_default_device(arch)

    whole = td.wavelet(10)                            # 21^3 points, 8000 cells
    pieces = td.split(whole, 4, ghost_layers=1)       # a PartitionedDataSet

    # Work on one piece takes the piece: loop over them.
    surfaces = []
    for piece in pieces:
        piece.fields["grad"] = td.gradients(piece, "RTData")
        surfaces.append(td.contour(piece, "RTData", 160.0))
    surfaces = td.PartitionedDataSet(surfaces)

    # Reductions take the pieces and count each cell and point once.
    print(td.statistics(pieces, "RTData").n, "points over the pieces,",
          whole.num_points, "in the whole")
    iso = td.contour(whole, "RTData", 160.0)
    assert td.statistics(surfaces, "RTData").n == iso.num_points
    # --8<-- [end:first]
    cells, points = td.Ghosts.SKIPPED_CELLS, td.Ghosts.SKIPPED_POINTS
    assert sum(owned(p.ghosts.cells, p.num_cells, cells) for p in pieces) == whole.num_cells
    assert sum(owned(s.ghosts.cells, s.num_cells, cells) for s in surfaces) == iso.num_cells
    for piece in pieces:
        piece.fields["g"] = td.magnitude(piece, "grad")
    whole.fields["g"] = td.magnitude(whole, td.gradients(whole, "RTData"))
    assert np.isclose(td.statistics(pieces, "g").mean, td.statistics(whole, "g").mean,
                      rtol=1e-5)

    # --8<-- [start:remove]
    finished = td.PartitionedDataSet(td.remove_ghosts(s) for s in surfaces)
    print(sum(s.num_cells for s in finished), "triangles,", iso.num_cells, "in the whole")
    # --8<-- [end:remove]
    assert sum(s.num_cells for s in finished) == iso.num_cells

    # --8<-- [start:no-ghosts]
    bare = td.split(whole, 4)                         # no ghost cells
    cut = td.PartitionedDataSet(td.contour(piece, "RTData", 160.0) for piece in bare)
    try:
        td.statistics(cut, "RTData")
    except NotImplementedError as error:
        print("refused:", str(error).split(":")[0])
    marked = td.generate_global_ids(cut)              # owners found by matching points
    assert td.statistics(marked, "RTData").n == iso.num_points

    surface = td.extract_surface(bare)                # the whole's surface, a piece each
    # --8<-- [end:no-ghosts]
    outside = td.extract_surface(whole)
    assert sum(owned(s.ghosts.cells, s.num_cells, cells) for s in surface) == outside.num_cells

    # --8<-- [start:ghost-cells]
    # Unstructured pieces without ghost cells, as a partitioner or a reader gives them.
    plain = td.PartitionedDataSet(td.tetrahedralize(piece) for piece in bare)
    ghosted = td.ghost_cells(plain)                   # one layer of neighbours' cells
    deeper = td.ghost_cells(plain, layers=2)          # two, for two neighbour filters in a row
    # --8<-- [end:ghost-cells]
    assert all(g.num_cells > p.num_cells for g, p in zip(ghosted, plain))
    assert all(d.num_cells > g.num_cells for d, g in zip(deeper, ghosted))

    # --8<-- [start:update]
    for piece in ghosted:
        piece.fields["c"] = td.values_at_centers(piece, "RTData")         # a cell field
        piece.fields["p"] = td.project(piece, "c", td.H1(piece))          # back to points
    td.update_ghosts(ghosted, "p")                    # the rim's values from their owners
    for piece in ghosted:
        piece.fields["grad_p"] = td.gradients(piece, "p", at="points")    # right where owned
    # --8<-- [end:update]
    tets = td.tetrahedralize(whole)
    tets.fields["c"] = td.values_at_centers(tets, "RTData")
    tets.fields["p"] = td.project(tets, "c", td.H1(tets))
    tets.fields["m"] = td.magnitude(tets, td.gradients(tets, "p", at="points"))
    for piece in ghosted:
        piece.fields["m"] = td.magnitude(piece, "grad_p")
    assert np.isclose(td.statistics(ghosted, "m").mean, td.statistics(tets, "m").mean,
                      rtol=1e-5)

    # --8<-- [start:global-ids]
    numbered = td.generate_global_ids(ghosted)
    ids = numbered[0].global_ids.points               # i64, the same at a shared point
    # --8<-- [end:global-ids]
    every = np.concatenate([p.global_ids.points.numpy() for p in numbered])
    assert np.array_equal(np.unique(every), np.arange(whole.num_points))

    # --8<-- [start:across]
    rng = np.random.default_rng(0)
    points = rng.uniform(-9.0, 9.0, (200, 3))
    probed = td.probe(pieces, points)                 # each point from the piece holding it
    regions = td.connected_components(pieces)         # one number per region, across pieces
    seeds = rng.uniform(-5.0, 5.0, (20, 3))
    paths = td.streamlines(pieces, "grad", seeds, 0.002, 100)   # each piece's stretches
    # --8<-- [end:across]
    assert np.array_equal(probed.fields["RTData"].values.numpy(),
                          td.probe(whole, points).fields["RTData"].values.numpy())
    assert all(int(p.fields["region"].values.numpy().max()) == 0 for p in regions)
    assert len(paths) == len(pieces)

    # --8<-- [start:merge]
    one = td.merge_pieces(pieces)                     # None on the other ranks
    print(type(one.topology).__name__, one.num_cells, "cells")
    # --8<-- [end:merge]
    assert one.num_cells == whole.num_cells and one.num_points == whole.num_points

    # --8<-- [start:devices]
    # Spread the pieces over the GPUs there are, or keep them on the CPU.
    gpus = [tack.device(f"cuda:{i}") for i in range(tack.device_count("cuda"))]
    devices = gpus or [tack.device("cpu")]
    spread = td.PartitionedDataSet(piece.to_device(devices[i % len(devices)])
                                   for i, piece in enumerate(pieces))
    print(td.statistics(spread, "RTData").mean)      # each piece reduced where it is
    # --8<-- [end:devices]

    # --8<-- [start:mpi]
    from tack import communication as tc

    try:
        from mpi4py import MPI
        world = tc.World(MPI.COMM_WORLD)              # under mpiexec: every rank
    except ImportError:
        world = tc.World()                            # this process alone

    every = td.split(whole, 8, ghost_layers=1)        # or each rank reads its own
    mine = td.PartitionedDataSet([every[i] for i in range(world.rank, 8, world.size)], world)
    print(td.statistics(mine, "RTData").mean)        # over all eight pieces, on every rank
    # --8<-- [end:mpi]
    assert np.isclose(td.statistics(mine, "RTData").mean,
                      td.statistics(whole, "RTData").mean, rtol=1e-6)
    print("Filters over pieces: every check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="accepted for validate.py: the results are always checked")
    parser.add_argument("--arch", default="cpu",
                        choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    run(parser.parse_args().arch)

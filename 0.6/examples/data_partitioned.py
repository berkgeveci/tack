# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cut a grid into pieces with ghost cells, filter each piece, and reduce over them."""

import argparse

import numpy as np

import tack
import tack.dataset as td


def owned(marks, n, skipped=td.Ghosts.SKIPPED_CELLS):
    """How many of n cells (or, with SKIPPED_POINTS, points) a piece owns: those not
    marked duplicate or hidden."""
    if marks is None:
        return n
    return int(((marks.numpy() & skipped) == 0).sum())


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:split]
    whole = td.wavelet(10)                                   # 21^3 points, 8000 cells
    pieces = td.split(whole, 4, ghost_layers=1)              # a PartitionedDataSet
    for piece in pieces:
        marks = piece.ghosts
        print(piece.topology.point_dims,
              owned(marks.cells, piece.num_cells), "cells owned of", piece.num_cells,
              owned(marks.points, piece.num_points, td.Ghosts.SKIPPED_POINTS),
              "points owned of", piece.num_points)
    # --8<-- [end:split]
    # --8<-- [start:loop]
    surfaces = []
    for piece in pieces:
        piece.fields["grad"] = td.gradients(piece, "RTData")
        surfaces.append(td.contour(piece, "RTData", 160.0))
    surfaces = td.PartitionedDataSet(surfaces)
    # --8<-- [end:loop]
    # --8<-- [start:reduce]
    grid_stats = td.statistics(pieces, "RTData")             # each point counted once
    surface_stats = td.statistics(surfaces, "RTData")        # each surface point once
    triangles = sum(owned(s.ghosts.cells, s.num_cells) for s in surfaces)
    # --8<-- [end:reduce]
    # --8<-- [start:remove]
    finished = td.PartitionedDataSet(td.remove_ghosts(s) for s in surfaces)
    # --8<-- [end:remove]
    # --8<-- [start:no-ghosts]
    bare = td.split(whole, 4)                                # no ghost cells
    cut = td.PartitionedDataSet(td.contour(piece, "RTData", 160.0) for piece in bare)
    try:
        td.statistics(cut, "RTData")
    except NotImplementedError as error:
        print("refused:", str(error).split(":")[0])
    # --8<-- [end:no-ghosts]
    if check:
        iso = td.contour(whole, "RTData", 160.0)
        assert sum(owned(p.ghosts.cells, p.num_cells) for p in pieces) == whole.num_cells
        assert grid_stats.n == whole.num_points
        assert np.isclose(grid_stats.mean, td.statistics(whole, "RTData").mean, rtol=1e-5)
        assert triangles == iso.num_cells and surface_stats.n == iso.num_points
        assert sum(s.num_cells for s in finished) == iso.num_cells
        assert td.statistics(finished, "RTData").n == iso.num_points
        # With ghost cells, a piece's gradients at its edge are the whole grid's.
        whole.fields["g"] = td.magnitude(whole, td.gradients(whole, "RTData"))
        whole_grad = td.statistics(whole, "g")
        for piece in pieces:
            piece.fields["g"] = td.magnitude(piece, "grad")
        assert np.isclose(td.statistics(pieces, "g").mean, whole_grad.mean, rtol=1e-5)
    print(f"Partitioned data: {len(pieces)} pieces; the isosurface has {triangles} triangles "
          f"and {surface_stats.n} points, as the whole grid's")
    return pieces, surfaces


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)

# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Filter, ghost, average down and resample a block-structured AMR hierarchy."""

import argparse


def run(arch="cpu"):
    # --8<-- [start:pulse]
    import numpy as np

    import tack
    import tack.dataset as td

    tack.set_default_device(arch)

    # VTK's AMR Gaussian pulse: a root image and two boxes refining it by 2.
    amr = td.amr_gaussian_pulse(dimension=2, amplitude=1.0)
    for level, box in amr:
        print(f"level {level}: {box.num_cells} cells")

    # Each region of space counts once, at the finest level that holds it.
    pulse = td.statistics(amr, "Gaussian-Pulse", weights="volume")
    print(f"integral {pulse.sum:.4f} over an area of {pulse.weight}")

    # Resample onto an image: each point from the finest box holding it.
    image = td.uniform_grid((26, 26), origin=(-2.0, -2.0), spacing=(0.1, 0.1))
    resampled = td.probe(amr, image)
    assert resampled.fields["valid"].values.numpy().all()
    # --8<-- [end:pulse]
    refined = td.Ghosts.REFINEDCELL
    finest = sum(float((box.fields["Gaussian-Pulse"].values.numpy()
                        * td.cell_geometry(box).measure.values.numpy())
                       [(box.ghosts.cells.numpy() & refined) == 0].sum()) for _, box in amr)
    assert np.isclose(pulse.sum, finest, rtol=1e-5) and np.isclose(pulse.weight, 6.25)

    # --8<-- [start:boxes]
    def box(lo, cells, spacing):
        """An image box of `cells` cells whose first cell is at index `lo` of its level."""
        corner = tuple(i * spacing for i in lo)
        points = tuple(n + 1 for n in cells)
        grid = td.uniform_grid(points, origin=corner, spacing=(spacing,) * 2)
        data = td.DataSet(td.StructuredTopology(points, offset=lo), grid.geometry.values)
        x = data.positions()
        f = np.exp(-((x[:, 0] - 4) ** 2 + (x[:, 1] - 4) ** 2) / 4)
        data.fields["f"] = td.Field(td.H1(data), tack.array(f.astype(np.float32)))
        return data

    # Two 4 x 8 root boxes, and a level-1 box across their seam (ratio 2).
    amr = td.AMRDataSet([[box((0, 0), (4, 8), 1.0), box((4, 0), (4, 8), 1.0)],
                         [box((4, 4), (8, 8), 0.5)]], ratios=[2])
    # --8<-- [end:boxes]

    # --8<-- [start:filter]
    for level, piece in amr:
        refined = (piece.ghosts.cells.numpy() & td.Ghosts.REFINEDCELL) != 0
        print(f"level {level}: {piece.num_cells} cells, {refined.sum()} refined")

    # The 0.5 contour lies where the fine box refines the roots,
    # so only the fine box gives segments.
    lines = [td.contour(piece, "f", 0.5) for _, piece in amr]
    print([line.num_cells for line in lines], "segments")
    # --8<-- [end:filter]
    assert lines[0].num_cells == lines[1].num_cells == 0 and lines[2].num_cells > 0

    # --8<-- [start:ghosts]
    grown = td.ghost_cells(amr)                  # every box a layer wider
    for level, piece in grown:
        piece.fields["g"] = td.magnitude(piece, td.gradients(piece, "f", at="points"))
    td.average_down(grown, "g")                  # refined points take the fine values
    td.update_ghosts(grown, "g")                 # ghost values from their owners
    # --8<-- [end:ghosts]
    sizes = [piece.num_cells for _, piece in grown]
    print("grown boxes:", sizes, "cells")
    assert sizes[2] > 64
    pieces = [piece for _, piece in grown]
    fine = {tuple(np.round(p, 6)): g for p, g in
            zip(pieces[2].positions(), pieces[2].fields["g"].values.numpy())}
    for coarse in pieces[:2]:
        marked = (coarse.ghosts.points.numpy() & td.Ghosts.REFINEDPOINT) != 0
        g = coarse.fields["g"].values.numpy()
        for p, value in zip(coarse.positions()[marked], g[marked]):
            assert value == fine[tuple(np.round(p, 6))]

    # --8<-- [start:probe]
    image = td.uniform_grid((65, 65), origin=(0.0, 0.0), spacing=(0.125, 0.125))
    resampled = td.probe(amr, image)             # each point from the finest level
    f = resampled.fields["f"].values.numpy()
    # --8<-- [end:probe]
    x = image.positions()
    fine = np.abs(x[:, :2] - 4).max(axis=1) < 2          # inside the level-1 box
    fine_box = next(piece for level, piece in amr if level == 1)
    from_fine = td.probe(fine_box, image).fields["f"].values.numpy()
    assert resampled.fields["valid"].values.numpy().all()
    assert np.array_equal(f[fine], from_fine[fine])
    print("Adaptive data: every check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="accepted for validate.py: the results are always checked")
    parser.add_argument("--arch", default="cpu",
                        choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    run(parser.parse_args().arch)

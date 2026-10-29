# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Fields of high order and a curved geometry on hexahedra; contour and clip them on sub-cells."""

import argparse

import numpy as np

import tack
import tack.dataset as td


# --8<-- [start:functions]
@tack.func
def blob(x):
    # A smooth field: its level sets are curved surfaces inside every cell.
    return x.x * x.x * x.x - x.y * x.z + 0.5 * x.y * x.y


@tack.func
def bend(x):
    # A bend of the mesh: its cells' faces curve.
    return [x.x + 0.1 * x.y * x.y, x.y + 0.1 * x.x * x.z, x.z]
# --8<-- [end:functions]


def run(arch, check):
    tack.set_default_device(arch)
    # --8<-- [start:spaces]
    data = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25))
    space = td.H1(data, order=4)                  # Gauss-Lobatto nodes by default
    data.fields["u"] = td.interpolate(data, blob, space)
    print(space, space.size, "values on", data.num_cells, "cells")
    # --8<-- [end:spaces]
    # A cubic of position is in the space: probing anywhere gives it back.
    rng = np.random.default_rng(0)
    points = rng.uniform(0.0, 1.0, (100, 3))
    probed = td.probe(data, points).fields["u"].values.numpy()
    exact = points[:, 0] ** 3 - points[:, 1] * points[:, 2] + 0.5 * points[:, 1] ** 2
    # --8<-- [start:contour]
    coarse = td.contour(data, "u", 0.2, subdivision=1)      # the cells' corners
    fine = td.contour(data, "u", 0.2)                       # 4 steps, the field's order
    within = td.contour(data, "u", 0.2, tolerance=1e-3)    # as many steps as that takes
    print(coarse.num_cells, fine.num_cells, within.num_cells, "triangles")
    # --8<-- [end:contour]
    # --8<-- [start:curved]
    curved = td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25))
    curved.geometry = td.interpolate(curved, bend, td.H1(curved, order=3), dtype=tack.vec3())
    cut = td.slice_plane(curved, (0.5, 0.5, 0.5), (0.0, 0.0, 1.0))
    linear = td.subdivide(curved, 3)                        # 27 hexahedra a cell
    print(cut.num_cells, "triangles;", linear.num_cells, "linear cells")
    # --8<-- [end:curved]
    # --8<-- [start:clip]
    whole, part = td.clip(data, "u", 0.2).partitions         # a PartitionedDataSet
    print(whole.num_cells, "whole cells at order", whole.fields["u"].space.order, "and",
          part.num_cells, "linear pieces")
    # --8<-- [end:clip]
    # --8<-- [start:layout]
    dg = td.L2(data, order=3, basis="gauss_legendre")             # "auto": by device
    across = td.L2(data, order=3, basis="gauss_legendre", layout="cells")
    p, q = (td.interpolate(data, blob, s) for s in (dg, across))
    print(dg.layout, across.layout, dg.blocks(p.values.numpy()).shape)
    # --8<-- [end:layout]
    if check:
        assert np.abs(probed - exact).max() < 1e-5               # f32
        assert across.layout == "cells"
        assert dg.layout == ("blocks" if arch == "cpu" else "cells")
        assert np.array_equal(dg.blocks(p.values.numpy()), across.blocks(q.values.numpy()))
        assert space.size == 17 ** 3                        # (4 * 4 + 1)^3 nodes
        assert coarse.num_cells < fine.num_cells < within.num_cells
        x = within.positions()
        values = x[:, 0] ** 3 - x[:, 1] * x[:, 2] + 0.5 * x[:, 1] ** 2
        assert np.abs(values - 0.2).max() <= 1e-3
        assert np.allclose(cut.positions()[:, 2], 0.5, atol=1e-6)
        assert linear.num_cells == 27 * curved.num_cells
        assert linear.num_points == td.H1(curved, order=3).size
        assert whole.num_cells and part.num_cells
        assert whole.fields["u"].space == td.H1(whole, order=4)
        assert whole.fields["u"].values.numpy().min() >= 0.2
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)

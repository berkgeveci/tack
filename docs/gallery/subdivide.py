# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Subdivide: high-order and curved cells as the linear sub-cells the cutting filters
cut and VTK can draw, 4 a side, the cells about a corner left out so the inside
shows: each sub-cell's edges drawn, the field as color.

From the checkout (writes docs/assets/gallery/subdivide-<kind>.jpg):
uv run python docs/gallery/subdivide.py [high-order curved curved-tetrahedra curved-wedges]
"""

import _scene
import numpy as np

import tack.dataset as td

KINDS = ("high-order", "curved", "curved-tetrahedra", "curved-wedges")


def subdivide(data):
    # --8<-- [start:subdivide]
    # Each cell's lattice of 4 steps a side: hexahedra into 64 hexahedra,
    # tetrahedra into Freudenthal's 64 tetrahedra, wedges into layers of them.
    # A lattice point shared by cells is one point, computed once, from the
    # entity it lies on, so every piece holding it gets the same bits.
    linear = td.subdivide(data, 4)
    # --8<-- [end:subdivide]
    return linear


def main():
    args = _scene.arguments(__doc__, KINDS)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        centers = td.cell_centers(data).values.numpy()
        far = np.linalg.norm(centers - (1.0, 0.0, 1.0), axis=1) > 0.6
        part = subdivide(td.extract_cells(data, np.flatnonzero(far)))
        _scene.draw(args.output / f"subdivide-{kind}.jpg",
                    [(td.extract_surface(part), "u", (-3.0, 3.0)),
                     (td.extract_edges(part), None, None,
                      {"color": (0.2, 0.2, 0.25), "opacity": 0.35, "ambient": 1.0,
                       "diffuse": 0.0})])


if __name__ == "__main__":
    main()

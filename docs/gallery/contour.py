# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Contour: Schwarz's P surface and two levels either side of it, on each kind of
cell, over the cells' outline.

From the checkout (writes docs/assets/gallery/contour-<kind>.jpg):
uv run python docs/gallery/contour.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td

LEVELS = (-1.0, 0.0, 1.0)


def contour(data):
    # --8<-- [start:contour]
    # Every level in one pass; each triangle's level in the cell field
    # "contour_value". High-order and curved cells are cut on linear sub-cells,
    # crossings placed on the field itself (exact=True).
    surfaces = td.contour(data, "u", LEVELS, subdivision=_scene.level(data))
    # --8<-- [end:contour]
    return surfaces


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        surfaces = _scene.normals(contour(data))
        _scene.draw(args.output / f"contour-{kind}.jpg",
                    [(surfaces, "contour_value", (LEVELS[0], LEVELS[-1]))],
                    outline=_scene.outline(data))


if __name__ == "__main__":
    main()

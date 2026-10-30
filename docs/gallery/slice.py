# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Slice: three planes through each kind of cell, colored by the field they cut,
over the cells' outline.

From the checkout (writes docs/assets/gallery/slice-<kind>.png):
uv run python docs/gallery/slice.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td
from tack.dataset import implicit

PLANES = ((0.5, 0.5, 0.15), (1.0, 0.0, 0.0)), ((0.5, 0.75, 0.5), (0.0, 1.0, 0.0)), \
    ((0.5, 0.5, 0.5), (0.3, -0.4, 1.0))


def slice_(data):
    # --8<-- [start:slice]
    # Each plane an implicit function; the field carried onto the cut. A curved
    # geometry or a high-order field is cut on linear sub-cells.
    steps = _scene.level(data)                # 6, or None on linear cells
    cuts = [td.slice(data, implicit.Plane(origin, normal), fields=["u"], subdivision=steps)
            for origin, normal in PLANES]
    # --8<-- [end:slice]
    return cuts


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        parts = [(cut, "u", (-3.0, 3.0)) for cut in slice_(data)]
        _scene.draw(args.output / f"slice-{kind}.png", parts, outline=_scene.outline(data),
                    lit=False)


if __name__ == "__main__":
    main()

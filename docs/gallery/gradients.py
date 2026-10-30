# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Gradients: the P surface (the field's zero) colored by the field's gradient's
length, on each kind of cell, over the cells' outline. A linear field's gradient
is averaged to the points; a high-order one's is taken at the nodes of an L2 space
of its order, keeping its detail.

From the checkout (writes docs/assets/gallery/gradients-<kind>.jpg):
uv run python docs/gallery/gradients.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td


def gradient(data):
    # --8<-- [start:gradients]
    u = data.fields["u"]
    if u.space.order > 1:
        # Exact at the nodes of L2 of the field's order: discontinuous, one
        # value per node of each cell.
        at = td.L2(data, order=u.space.order)
    else:
        # Each cell's gradient at its corners, averaged over the cells around
        # each point (VTK's and Viskores' point gradients).
        at = "points"
    data.fields["slope"] = td.magnitude(data, td.gradients(data, "u", at=at))
    surface = td.contour(data, "u", 0.0, fields=["slope"], subdivision=_scene.level(data))
    # --8<-- [end:gradients]
    return surface


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        surface = _scene.normals(gradient(data))
        _scene.draw(args.output / f"gradients-{kind}.jpg", [(surface, "slope", (6.0, 11.0))],
                    outline=_scene.outline(data))


if __name__ == "__main__":
    main()

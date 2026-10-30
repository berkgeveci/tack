# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Probe: each kind of cell sampled at the points of a tilted image plane wider than
the cube, the points no cell holds dropped, colored by the field, over the cells'
outline. On the curved cells the plane's edge follows the curved boundary.

From the checkout (writes docs/assets/gallery/probe-<kind>.jpg):
uv run python docs/gallery/probe.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene
import numpy as np

import tack.dataset as td


def plane(n=160):
    """An n x n image, 1.6 wide, through the cube's centre, turned 35 degrees about
    x and 20 about z: transform keeps it an image."""
    image = td.uniform_grid((n, n), origin=(-0.8, -0.8), spacing=(1.6 / (n - 1),) * 2)
    a, b = np.radians(35.0), np.radians(20.0)
    about_x = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    about_z = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return td.transform(image, np.c_[about_z @ about_x, (0.5, 0.5, 0.5)])


def probe(data):
    # --8<-- [start:probe]
    # Each point's cell found through the cell locator (curved cells inverted by
    # Newton's method from their own geometry), the field evaluated there;
    # "valid" is 0 where no cell holds the point.
    sampled = td.probe(data, plane(), fields=["u"])
    held = td.threshold(sampled, "valid", 1, 1)
    # --8<-- [end:probe]
    return held


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        _scene.draw(args.output / f"probe-{kind}.jpg", [(probe(data), "u", (-3.0, 3.0))],
                    outline=_scene.outline(data), lit=False)


if __name__ == "__main__":
    main()

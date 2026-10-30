# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Glyphs: the swirl's velocity as arrows on a lattice across the cube's middle,
sampled by probe, over the slice there colored by speed.

From the checkout (writes docs/assets/gallery/glyphs-<kind>.jpg):
uv run python docs/gallery/glyphs.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td
from tack.dataset import implicit


def lattice(n=12):
    """An n x n image across the cube at z = 0.5, inset from its sides."""
    return td.uniform_grid((n, n), origin=(0.05, 0.05, 0.5), spacing=(0.9 / (n - 1),) * 2)


def arrows(data):
    # --8<-- [start:glyphs]
    # The velocity at the lattice's points (the points no cell holds dropped),
    # an arrow at each, 0.1 long at unit speed, and the slice beneath.
    sampled = td.probe(data, lattice(), fields=["v"])
    held = td.threshold(sampled, "valid", 1, 1)
    held.fields["speed"] = td.magnitude(held, "v")
    shafts = td.glyphs(held, "v", scale=0.1)
    plane = td.slice(data, implicit.Plane((0.5, 0.5, 0.5), (0.0, 0.0, 1.0)), fields=["v"],
                     subdivision=_scene.level(data))
    plane.fields["speed"] = td.magnitude(plane, "v")
    # --8<-- [end:glyphs]
    return shafts, plane


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.with_velocity(_scene.dataset(kind))
        shafts, plane = arrows(data)
        _scene.draw(args.output / f"glyphs-{kind}.jpg",
                    [(shafts, None, None, {"color": (0.15, 0.15, 0.2)}),
                     (plane, "speed", (0.0, 1.5), {"opacity": 0.75})],
                    outline=_scene.outline(data), view=(30.0, 45.0))


if __name__ == "__main__":
    main()

# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Streamlines: paths through a swirl from seeds across the cube, as tubes colored
by the time taken, on each kind of cell, over the cells' outline.

From the checkout (writes docs/assets/gallery/streamlines-<kind>.jpg):
uv run python docs/gallery/streamlines.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene
import numpy as np

import tack.dataset as td

#: Seeds on a 5 x 5 lattice across the cube's middle, slightly off it.
SEEDS = np.array([[0.5 + 0.01 * (i - j), 0.1 + 0.2 * i, 0.1 + 0.2 * j]
                  for i in range(5) for j in range(5)])


def streamlines(data):
    # --8<-- [start:streamlines]
    # RK45 with steps measured along the path (as VTK's vtkStreamTracer): each
    # step's error kept under the tolerance; a path ends where it leaves the
    # mesh. The velocity is evaluated in its own space on each kind of cell,
    # curved ones inverted from their geometry.
    paths = td.streamlines(data, "v", SEEDS, 0.01, 300, integrator="rk45",
                           step_unit="length", tolerance=1e-5, max_length=1.4)
    tubes = td.tubes(paths, 0.006, sides=10)
    # --8<-- [end:streamlines]
    return tubes


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.with_velocity(_scene.dataset(kind))
        _scene.draw(args.output / f"streamlines-{kind}.jpg",
                    [(streamlines(data), "time", (0.0, 3.0))], outline=_scene.outline(data))


if __name__ == "__main__":
    main()

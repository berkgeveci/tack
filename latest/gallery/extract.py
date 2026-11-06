# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Contour, clip and threshold of one ball, as "Extract parts of a dataset" compares
them: the distance from the centre of a cube, r, and the three answers about
r <= 0.5. Each is cut open by the plane y = 0, so the sphere shows as a shell, the
clipped ball as solid with a smooth surface, and the whole cells as solid and
blocky; every cell's edges are drawn.

From the checkout (writes docs/assets/gallery/extract-<sphere|ball|cells>.jpg):
uv run python docs/gallery/extract.py [--arch metal]
"""

import _scene
import numpy as np

import tack
import tack.dataset as td


def ball_data():
    data = td.uniform_grid((21, 21, 21), origin=(-1, -1, -1), spacing=(0.1, 0.1, 0.1))
    data.fields["r"] = td.Field(td.H1(data), tack.array(np.linalg.norm(data.positions(), axis=1)))
    return data


def extracted(data):
    # --8<-- [start:extract]
    sphere = td.contour(data, "r", 0.5)                # the level set: triangles
    ball = td.clip(data, "r", 0.5, inside=True)        # cut cells: the part below 0.5
    cells = td.threshold(data, "r", 0.0, 0.5)          # whole cells, every corner in range
    # --8<-- [end:extract]
    return {"sphere": sphere, "ball": ball, "cells": cells}


def main():
    args = _scene.arguments(__doc__, kinds=())
    back = td.Plane(origin=(0, 0, 0), normal=(0, 1, 0))      # keeps y >= 0, open to the view
    for name, part in extracted(ball_data()).items():
        half = td.clip(part, back)
        if name == "sphere":
            drawn = _scene.normals(half)
        else:
            drawn = td.extract_surface(half)
        _scene.draw(args.output / f"extract-{name}.jpg", [(drawn, "r", (0.0, 0.5))],
                    outline=td.extract_edges(half), view=(35.0, 55.0), size=(480, 420))


if __name__ == "__main__":
    main()

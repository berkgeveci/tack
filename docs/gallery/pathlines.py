# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pathlines: particles carried through an unsteady swirl -- its second part's weight
rising and falling over the steps -- each one's path as a tube colored by time, on
each kind of cell, over the cells' outline.

From the checkout (writes docs/assets/gallery/pathlines-<kind>.jpg):
uv run python docs/gallery/pathlines.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene
import numpy as np

import tack
import tack.dataset as td

#: Seeds on a 4 x 4 lattice across the cube's lower middle.
SEEDS = np.array([[0.2 + 0.2 * i, 0.2 + 0.2 * j, 0.3] for i in range(4) for j in range(4)])
STEPS = 60
DT = 0.04


def step(data, base, turn, k):
    """The mesh at step ``k``: velocity ``base + w(t) turn``, w from 0 to 1 and back."""
    t = k * DT
    w = 0.5 - 0.5 * np.cos(2 * np.pi * k / STEPS)
    values = base + w * turn
    v = tack.empty((len(values),), tack.vec3())
    tack.copyto(v, values.astype(np.float32))
    out = td.DataSet(data.topology, data.positions(), time=t)
    out.geometry = data.geometry                     # a curved one, as it is
    out.fields["v"] = td.Field(data.fields["u"].space, v)
    return out


def pathlines(data):
    space = data.fields["u"].space
    base = td.interpolate(data, _scene.swirl_base, space, dtype=tack.vec3()).values.numpy()
    turn = td.interpolate(data, _scene.swirl_turn, space, dtype=tack.vec3()).values.numpy()
    # --8<-- [start:pathlines]
    # A temporal filter: each update takes a step's dataset and moves every
    # particle from its time to the step's, the velocity linear in time
    # between the steps, in RK4 substeps of at most step_size.
    paths = td.Pathlines("v", seeds=SEEDS, step_size=0.025)
    for k in range(STEPS + 1):
        lines = paths.update(step(data, base, turn, k))
    tubes = td.tubes(lines, 0.008, sides=10)
    # --8<-- [end:pathlines]
    return tubes


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        _scene.draw(args.output / f"pathlines-{kind}.jpg",
                    [(pathlines(data), "time", (0.0, STEPS * DT))], outline=_scene.outline(data))


if __name__ == "__main__":
    main()

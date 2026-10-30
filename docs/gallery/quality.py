# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cell quality: each cell's scaled Jacobian (1 ideal, 0 degenerate, negative
inverted), on meshes with their inner points jittered and on curved cells, the
cells near a corner taken out so the inside shows.

From the checkout (writes docs/assets/gallery/quality-<kind>.jpg):
uv run python docs/gallery/quality.py [polyhedra linear mixed curved] [--arch metal]
"""

import _scene
import numpy as np

import tack.dataset as td

KINDS = ("polyhedra", "linear", "mixed", "curved")


def quality(data):
    # --8<-- [start:quality]
    # Verdict's measures generalized: on each shape from its ideal cell, on
    # polyhedra from the tetrahedra fanning them, on a curved geometry at the
    # samples of its own Jacobian.
    data.fields["scaled_jacobian"] = td.cell_quality(data, ["scaled_jacobian"])[
        "scaled_jacobian"]
    # --8<-- [end:quality]
    return data


def cut_open(data):
    """The cells whose centres lie outside a ball about the cube's near top corner,
    whole (the inside shows as the cells' faces, not a cut)."""
    centers = td.cell_centers(data).values.numpy()
    far = np.linalg.norm(centers - (1.0, 0.0, 1.0), axis=1) > 0.6
    return td.extract_cells(data, np.flatnonzero(far))


def main():
    args = _scene.arguments(__doc__, KINDS)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        if kind == "curved":
            # The straight hexahedra jittered, then twisted.
            straight = td.uniform_grid((5, 5, 5), spacing=(1 / 4,) * 3)
            data = _scene.twisted(_scene.jittered(td.extract_cells(straight,
                                                                   np.arange(64))))
        else:
            data = _scene.jittered(data)
        part = cut_open(quality(data))
        _scene.draw(args.output / f"quality-{kind}.jpg",
                    [(_scene.boundary(part), "scaled_jacobian", (1.0, -0.2))],
                    outline=_scene.outline(part))


if __name__ == "__main__":
    main()

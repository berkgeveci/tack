# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Pieces: each kind of cell in eight pieces, one an octant of the cube, each given a
layer of its neighbours' cells as ghosts, and drawn pulled apart: a piece's own
cells in its color, its ghost cells faint.

From the checkout (writes docs/assets/gallery/pieces-<kind>.jpg):
uv run python docs/gallery/pieces.py [image linear mixed] [--arch metal]
"""

import _scene
import numpy as np

import tack.dataset as td
from tack.dataset import Ghosts

#: A color for each piece (Tableau's ten, the first eight).
PALETTE = ((0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
           (0.58, 0.4, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.5, 0.5, 0.5))

# Not polyhedra, nor curved or high-order cells: ghost_cells takes unstructured
# and structured pieces, not polyhedral ones yet, and carries positions and
# values at the points.
KINDS = ("image", "linear", "mixed")


def octants(data):
    """``data``'s cells by the octant of the cube their centres are in, as pieces: an
    image split into eight blocks (``split``, structured pieces), any other by
    extracting each octant's cells."""
    if isinstance(data.topology, td.StructuredTopology):
        return td.split(data, (2, 2, 2))
    centers = td.cell_centers(data).values.numpy()
    octant = ((centers > 0.5) * (1, 2, 4)).sum(axis=1)
    return td.PartitionedDataSet([td.extract_cells(data, np.flatnonzero(octant == k))
                                  for k in range(8)])


def pieces(data):
    # --8<-- [start:pieces]
    # Pieces neighbour where they share points; each gains the cells of the
    # others that use a point it shares (marked DUPLICATECELL), and owns each
    # shared point once. ghost_cells keeps the plans that update_ghosts reruns.
    ghosted = td.ghost_cells(octants(data))
    # --8<-- [end:pieces]
    return ghosted


def main():
    args = _scene.arguments(__doc__, KINDS)
    for kind in args.kinds:
        if kind == "image":
            data = td.uniform_grid((9, 9, 9), spacing=(1 / 8,) * 3)
            data.fields["u"] = td.interpolate(data, _scene.schwarz, td.H1(data))
        else:
            data = _scene.dataset(kind)
        parts = []
        for k, piece in enumerate(pieces(data)):
            # Pulled out from the cube's centre toward the octant's corner.
            corner = np.array([(k >> axis) & 1 for axis in range(3)]) - 0.5
            moved = td.transform(piece, np.c_[np.eye(3), 0.35 * corner])
            marks = moved.ghosts.cells.numpy() & Ghosts.DUPLICATECELL
            own = td.extract_cells(moved, np.flatnonzero(marks == 0))
            ghosts = td.extract_cells(moved, np.flatnonzero(marks != 0))
            parts.append((_scene.boundary(own), None, None, {"color": PALETTE[k]}))
            if ghosts.num_cells:
                parts.append((_scene.boundary(ghosts), None, None, {"opacity": 0.25}))
        _scene.draw(args.output / f"pieces-{kind}.jpg", parts)


if __name__ == "__main__":
    main()

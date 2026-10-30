# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Extract surface and feature edges: the cube with a ball taken out (as the clip
entries make it), its boundary, and its edges where the boundary turns by more
than 30 degrees, on each kind of cell.

From the checkout (writes docs/assets/gallery/surfaces-<kind>.jpg):
uv run python docs/gallery/surfaces.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene
import clip

import tack.dataset as td
from tack.dataset import PartitionedDataSet


def surfaces(kept):
    # --8<-- [start:surfaces]
    # A high-order or curved part comes subdivided into linear cells first: the
    # surface of its sub-cells, a polyhedral one's its polygons. The feature
    # edges are VTK's: boundary, sharp (two faces' normals 30 degrees apart or
    # more) and non-manifold edges, as lines with their kind.
    surface = td.extract_surface(_scene.linear(kept))
    edges = td.feature_edges(surface, angle=30.0)
    # --8<-- [end:surfaces]
    return surface, edges


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        kept = clip.clip(_scene.dataset(kind))
        if isinstance(kept, PartitionedDataSet):
            whole, part = kept.partitions
            kept = td.merge_pieces(PartitionedDataSet([_scene.linear(whole), part]))
        surface, edges = surfaces(kept)
        _scene.draw(args.output / f"surfaces-{kind}.jpg",
                    [(surface, None, None),
                     (edges, None, None, {"color": (0.75, 0.1, 0.1), "line_width": 3.0,
                                          "ambient": 1.0, "diffuse": 0.0})])


if __name__ == "__main__":
    main()

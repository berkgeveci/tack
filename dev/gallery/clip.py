# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Clip: each kind of cell with a ball about a corner of the cube taken out, the
kept part's surface colored by the field, over the cells' outline.

From the checkout (writes docs/assets/gallery/clip-<kind>.jpg):
uv run python docs/gallery/clip.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td
from tack.dataset import PartitionedDataSet, implicit


def clip(data):
    # --8<-- [start:clip]
    # The part outside a ball about the cube's near top corner. On high-order or
    # curved cells the cells wholly kept stay as they are and the cut ones are
    # cut on linear sub-cells: a PartitionedDataSet of the two.
    ball = implicit.Sphere(center=(1.0, 0.0, 1.0), radius=0.6)
    kept = td.clip(data, ball, subdivision=_scene.level(data))
    # --8<-- [end:clip]
    return kept


def surface(kept):
    """The kept part's boundary, linear: a PartitionedDataSet's whole cells
    subdivided and joined to the cut part (the seam's points are the same), so
    the faces between them are not drawn."""
    if isinstance(kept, PartitionedDataSet):
        whole, part = kept.partitions
        kept = td.merge_pieces(PartitionedDataSet([_scene.linear(whole), part]))
    return _scene.boundary(kept)


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        _scene.draw(args.output / f"clip-{kind}.jpg", [(surface(clip(data)), "u", (-3.0, 3.0))],
                    outline=_scene.outline(data))


if __name__ == "__main__":
    main()

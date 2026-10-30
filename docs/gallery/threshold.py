# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Threshold: the cells where the field is at least 1 -- the eight blobs about the
cube's corners and the middles of its faces -- on each kind of cell, over the
cells' outline.

From the checkout (writes docs/assets/gallery/threshold-<kind>.jpg):
uv run python docs/gallery/threshold.py [polyhedra linear high-order curved] [--arch metal]
"""

import _scene

import tack.dataset as td
from tack.dataset import PartitionedDataSet


def threshold(data):
    # --8<-- [start:threshold]
    # Linear cells are kept whole when every corner is in range. High-order and
    # curved cells are kept whole when the field's bounds (its Bernstein
    # coefficients) lie in range; the others are cut on linear sub-cells at the
    # range's ends: a PartitionedDataSet of the two.
    kept = td.threshold(data, "u", 1.0, 3.0, subdivision=_scene.level(data))
    # --8<-- [end:threshold]
    return kept


def main():
    args = _scene.arguments(__doc__)
    for kind in args.kinds:
        data = _scene.dataset(kind)
        kept = threshold(data)
        pieces = kept.partitions if isinstance(kept, PartitionedDataSet) else [kept]
        parts = [(_scene.boundary(piece), "u", (-3.0, 3.0)) for piece in pieces
                 if piece.num_cells]
        _scene.draw(args.output / f"threshold-{kind}.jpg", parts, outline=_scene.outline(data))


if __name__ == "__main__":
    main()

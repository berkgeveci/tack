# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Compact positive values using atomic results and a scalar kernel return."""

import argparse

import numpy as np

import tack


# --8<-- [start:compact]
@tack.kernel
def compact_positive(values, out, count, n) -> int:
    for i in range(n):
        if values[i] > 0.0:
            slot = tack.atomic_add(count, 0, 1)
            out[slot] = values[i]
    return count[0]
# --8<-- [end:compact]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    # --8<-- [start:run]
    values = tack.field_like(np.array([-2, 1, -0.5, 3, 0, 2], dtype=np.float32))
    out = tack.field(dtype=tack.f32, shape=values.shape)  # capacity for every input
    count = tack.field(dtype=tack.i32, shape=(1,))
    count.fill(0)
    length = compact_positive(values, out, count, values.size)
    result = out.to_numpy()[:length]
    # Arrival order is unspecified: compare the values after sorting.
    assert length == 3 and isinstance(length, int)
    np.testing.assert_array_equal(np.sort(result), np.array([1, 2, 3], dtype=np.float32))
    # --8<-- [end:run]
    values.fill(0)
    count.fill(0)
    assert compact_positive(values, out, count, values.size) == 0
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
    print("Kernel results: check passed")

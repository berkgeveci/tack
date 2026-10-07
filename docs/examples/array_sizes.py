# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Use arithmetic on a constant and a resolved field dimension for local scratch."""

import argparse

import numpy as np

import tack

# --8<-- [start:scratch]
PAIR = tack.constant(2)


@tack.kernel
def row_energy(data, out):
    for row in range(data.shape[0]):
        scratch = tack.local_array_like(data, PAIR * data.shape[1])
        for column in range(data.shape[1]):
            x = data[row, column]
            scratch[PAIR * column] = x
            scratch[PAIR * column + 1] = x * x
        total = 0.0
        for column in range(PAIR * data.shape[1]):
            total += scratch[column]
        out[row] = total
# --8<-- [end:scratch]


def run(arch="cpu", check=False):
    tack.init(arch=getattr(tack, arch))
    # A changed row width resolves a different scratch capacity.
    for width in (4, 7):
        host = np.arange(3 * width, dtype=np.float32).reshape(3, width)
        data = tack.field_like(host)
        out = tack.field(dtype=tack.f32, shape=(3,))
        row_energy(data, out)
        np.testing.assert_array_equal(out.to_numpy(), (host + host * host).sum(axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)
    print("Array sizes: check passed")

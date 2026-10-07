# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate the tutorial figures from actual CPU computations.

From the checkout:
uv run --with matplotlib python docs/examples/figures.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import contours
import dye
import heat
import isosurface
import mpm
import nbody
import physarum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "assets" / "tutorials")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for module in (heat, dye, physarum, contours, mpm, nbody, isosurface):
        print(f"Generating {module.__name__} on cpu", flush=True)
        result = module.run("cpu")
        module.plot(result, args.output / f"{module.__name__}.png")


if __name__ == "__main__":
    main()

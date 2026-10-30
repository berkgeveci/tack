# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate every gallery picture (docs/assets/gallery/), each script in its own
process; the pictures are checked in.

From the checkout:
uv run python docs/gallery/build.py [--arch cpu]
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = ("contour", "slice", "clip", "threshold", "surfaces", "quality", "gradients", "probe",
           "subdivide", "pieces", "streamlines", "glyphs", "pathlines")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    for name in SCRIPTS:
        subprocess.run([sys.executable, str(here / f"{name}.py"), "--arch", args.arch],
                       check=True)


if __name__ == "__main__":
    main()

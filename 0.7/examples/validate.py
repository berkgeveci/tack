# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the guides' small numerical checks on one explicitly chosen backend."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES = ("heat", "dye", "physarum", "contours", "mpm", "nbody",
            "kernel_results", "array_sizes", "datasets", "data_fields", "data_meshes",
            "data_analysis", "data_mesh_kernels", "data_filters", "data_spaces",
            "data_partitioned", "data_high_order", "data_finite_volume", "comm_ghosts",
            "data3_build", "vis2_amr", "vis2_finite_volume", "vis2_pieces", "vis3_mesh_algorithm")
GUIDES = ("core", "communication", "data", "visualization", "vis-guide",
          "users-guide/tutorials")


def check_quickstarts():
    """Execute the pages' marked examples, the actual Markdown code."""
    docs = Path(__file__).resolve().parents[1]
    checked = 0
    with tempfile.TemporaryDirectory(prefix="tack-guide-check-") as temporary:
        # Mark complete examples beyond a page's opening with an HTML comment:
        # <!-- tack-check: example-name --> immediately before a Python fence.
        # Execute the displayed Markdown so its inputs and assertions cannot
        # drift away from a separate validation script.
        for page in sorted([docs / "index.md", docs / "install.md", docs / "use-cases.md"]
                           + [p for guide in GUIDES for p in (docs / guide).glob("*.md")]):
            blocks = re.findall(
                r"<!-- tack-check: ([a-z0-9-]+) -->\s*```python\n(.*?)```",
                page.read_text(encoding="utf-8"), flags=re.S)
            for name, code in blocks:
                script = Path(temporary) / f"{page.parent.name}-{page.stem}-{name}.py"
                script.write_text(code, encoding="utf-8")
                print(f"Checking Markdown example: {page.name}/{name} on cpu", flush=True)
                subprocess.run([sys.executable, str(script)], check=True, timeout=120,
                               cwd=temporary)
                checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--vtk", action="store_true",
                        help="Also check CPU interop with a DLPack-enabled VTK build")
    args = parser.parse_args()
    for name in EXAMPLES:
        path = Path(__file__).with_name(f"{name}.py")
        print(f"Checking {name} on {args.arch}", flush=True)
        subprocess.run([sys.executable, str(path), "--arch", args.arch, "--check"],
                       check=True, timeout=120)
    print(f"All {len(EXAMPLES)} tutorial checks passed on {args.arch}.")
    if args.arch == "cpu":
        checked = check_quickstarts()
        print(f"All {checked} Markdown examples passed on cpu.")
    if args.vtk:
        print("Checking VTK interop on cpu", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("vtk_interop.py"))],
                       check=True, timeout=120)


if __name__ == "__main__":
    main()

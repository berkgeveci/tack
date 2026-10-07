# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Run the user guide's small numerical checks on one explicitly chosen backend."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

EXAMPLES = ("heat", "dye", "physarum", "contours", "mpm", "nbody", "isosurface",
            "kernel_results", "array_sizes")
QUICKSTARTS = (
    "01-getting-started", "12-execution-model", "14-vectors-and-matrices",
    "09-visualization", "10-rendering", "15-debugging-and-timing", "16-interoperability",
)


def check_quickstarts():
    """Execute the complete opening examples, including the actual Markdown code."""
    guide = Path(__file__).resolve().parents[1] / "users-guide"
    with tempfile.TemporaryDirectory(prefix="tack-guide-check-") as temporary:
        for name in QUICKSTARTS:
            page = guide / f"{name}.md"
            blocks = re.findall(r"```python\n(.*?)```", page.read_text(), flags=re.S)
            if not blocks:
                raise RuntimeError(f"No Python quick start in {page.name}")
            code = blocks[0]
            if "import tack" not in code or "tack.init" not in code:
                raise RuntimeError(f"The first Python example in {page.name} is not standalone")
            if name == "09-visualization":
                code += '\nassert mesh is not None and mesh["total_tris"] > 0\n'
            if name == "01-getting-started":
                code += '\nnp.testing.assert_array_equal(result, np.arange(n, dtype=np.float32) + 2)\n'
            script = Path(temporary) / f"{name}.py"
            script.write_text(code)
            print(f"Checking Markdown quick start: {page.name} on cpu", flush=True)
            subprocess.run([sys.executable, str(script)], check=True, timeout=120)


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
        check_quickstarts()
        print(f"All {len(QUICKSTARTS)} Markdown quick starts passed on cpu.")
    if args.vtk:
        print("Checking VTK interop on cpu", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("vtk_interop.py"))],
                       check=True, timeout=120)


if __name__ == "__main__":
    main()

"""Compare vtk_compare runs of several builds, row by row.

    python benchmarks/vtk_compare/compare.py BASE=DIR [LABEL=DIR ...] [--sizes 1M,5M]

Each DIR holds one or more passes of ``run.py --output``, directly or in
subdirectories (``DIR/cpu8.json`` or ``DIR/*/cpu8.json``): run each build
more than once, alternating builds, so drift lands on all of them. Per
row (config, mesh, size, form, filter) the warm median of each pass is
taken, then the median over passes; every other build is reported as a
ratio to the first. Checks are counted per build.
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


def load(path):
    warm = defaultdict(list)
    checks = []
    files = sorted(glob.glob(os.path.join(path, "cpu*.json")) + glob.glob(os.path.join(path, "*.json"))
                   + glob.glob(os.path.join(path, "*", "*.json")))
    seen = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        j = json.load(open(f))
        config = j["meta"]["config"]
        for r in j["records"]:
            key = (config, r["mesh"], r["size"], r["form"], r["filter"])
            if r["impl"] == "check":
                checks.append(bool(r["ok"]))
            elif r["impl"] == "tack" and r.get("mode") == "warm":
                warm[key].append(r["median"] * 1e3)
    return warm, checks, len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("builds", nargs="+", metavar="LABEL=DIR")
    parser.add_argument("--sizes", default="", help="comma-separated sizes to report")
    parser.add_argument("--rows", action="store_true", help="print every row, not only the summary")
    args = parser.parse_args()
    builds = [b.split("=", 1) for b in args.builds]
    data = {label: load(path) for label, path in builds}
    for label, (_, checks, files) in data.items():
        print(f"{label}: {files} files, checks {sum(checks)}/{len(checks)} ok")
    base_label = builds[0][0]
    base = data[base_label][0]
    sizes = args.sizes.split(",") if args.sizes else sorted({k[2] for k in base})
    configs = sorted({k[0] for k in base}, key=lambda c: (len(c), c))
    for size in sizes:
        for form in sorted({k[3] for k in base}):
            for config in configs:
                keys = sorted(k for k in base if k[0] == config and k[2] == size and k[3] == form
                              and k[4] != "import"
                              and all(k in data[label][0] for label, _ in builds[1:]))
                if not keys:
                    continue
                print(f"\n== {size} {form} {config}: {len(keys)} rows")
                for label, _ in builds[1:]:
                    r = [st.median(data[label][0][k]) / st.median(base[k]) for k in keys]
                    print(f"  {label}/{base_label} median {st.median(r):.3f}, "
                          f"{sum(v < 0.97 for v in r)} faster than 3%, {sum(v > 1.03 for v in r)} slower")
                if args.rows:
                    print(f"  {'mesh filter':<28}" + "".join(f"{label:>10}" for label, _ in builds))
                    for k in keys:
                        print(f"  {k[1]} {k[4]:<24}"
                              + "".join(f"{st.median(data[label][0][k]):10.2f}" for label, _ in builds))


if __name__ == "__main__":
    main()

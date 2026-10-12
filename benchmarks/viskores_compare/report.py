"""Tables of a Tack-versus-Viskores run: one per mesh and size, a row per filter.

    python report.py results-viskores [--size 5M] [--mesh hex]

Columns are median times in milliseconds -- Viskores (through VTK), Tack with
its data on the GPU, and Tack's round trip from and back to host memory, the
path Viskores takes -- and the two ratios to Viskores (below 1: Tack is
faster). A pair whose outputs disagreed is marked and its ratios left out.
The cold columns are each filter's first call in the process.
"""

import argparse
import json
import os


def _load(directory):
    with open(os.path.join(directory, "viskores.json")) as data:
        return json.load(data)


def _ms(seconds):
    if seconds is None:
        return "-"
    ms = 1e3 * seconds
    return f"{ms:.3g}" if ms < 100 else f"{ms:.0f}"


def _ratio(a, b):
    return f"{a / b:.2f}" if a is not None and b else "-"


def report(directory, sizes=None, meshes=None):
    run = _load(directory)
    meta, records = run["meta"], run["records"]
    print(f"Tack {meta['tack']} ({meta['tack backend']}) vs Viskores in VTK {meta['vtk']}, "
          f"device {meta['viskores device']}"
          + ("; resident: both sides' data on the GPU" if meta.get("resident") else ""))
    for name, why in meta.get("left out by the preflight", {}).items():
        print(f"left out: {name}: {why}")
    print(f"{meta['gpu']}; {meta['processor']}; {meta['date']}\n")
    timed, checks, notes, cold = {}, {}, {}, {}
    order = []
    for r in records:
        key = (r["mesh"], r["size"], r["filter"])
        if (r["mesh"], r["size"]) not in order:
            order.append((r["mesh"], r["size"]))
        if r["impl"] == "check":
            checks.setdefault(key, {})[r["against"]] = r["ok"]
        elif r.get("mode") == "cold":
            cold[(r["filter"], r["impl"])] = r["median"]
        elif r.get("mode") == "fallback":
            notes[key] = "Viskores fell back"
        elif "median" in r:
            timed[(*key, r["impl"], r["mode"])] = r["median"]
    filters = list(dict.fromkeys(r["filter"] for r in records))
    second = "fresh" if meta.get("resident") else "round trip"
    head = (f"{'filter':<16}{'viskores':>10}{'tack':>10}{second:>12}"
            f"{'tack/visk':>11}{second[:5] + '/visk':>11}  checks")
    for mesh, size in order:
        if (sizes and size not in sizes) or (meshes and mesh not in meshes):
            continue
        cells = next(r["cells"] for r in records if r["mesh"] == mesh and r["size"] == size)
        print(f"{mesh} {size} ({cells:,} cells), ms")
        print(head)
        for name in filters:
            key = (mesh, size, name)
            if key not in checks:
                continue
            v = timed.get((*key, "viskores", "warm"))
            t = timed.get((*key, "tack", "warm"))
            rt = timed.get((*key, "tack", "round trip")) or timed.get((*key, "tack", "fresh"))
            ok = checks[key].get("viskores", False) and key not in notes
            marks = ["tack=visk" if checks[key].get("viskores") else "TACK DIFFERS"]
            if "vtk" in checks[key]:
                marks.append("visk=vtk" if checks[key]["vtk"] else "visk differs from vtk")
            if key in notes:
                marks.append(notes[key])
            print(f"{name:<16}{_ms(v):>10}{_ms(t):>10}{_ms(rt):>12}"
                  f"{_ratio(t, v) if ok else '-':>11}{_ratio(rt, v) if ok else '-':>11}  "
                  + ", ".join(marks))
        print()
    print("cold (first call in the process, ms)")
    print(f"{'filter':<16}{'viskores':>10}{'tack':>10}")
    for name in filters:
        print(f"{name:<16}{_ms(cold.get((name, 'viskores'))):>10}"
              f"{_ms(cold.get((name, 'tack'))):>10}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--size", action="append", help="only this size (repeatable)")
    parser.add_argument("--mesh", action="append", help="only this mesh (repeatable)")
    args = parser.parse_args()
    report(args.directory, args.size, args.mesh)


if __name__ == "__main__":
    main()

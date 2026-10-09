"""Print the comparison from a vtk_compare results directory.

Reads every ``<config>.json`` that ``run.py`` wrote there: ``cpu1``, the
``cpuN`` with the most threads, and any GPU runs. For each mesh, size, form
and filter it prints the median times in ms and the ratio Tack / VTK --
below 1, Tack is faster -- at one thread, at N threads (with Tack's
``fresh`` time: nothing derived beforehand), and for each GPU against VTK on
N threads, plus the outputs check. ``--size`` and ``--mesh`` narrow it.

Usage:
  python benchmarks/vtk_compare/report.py DIR [--size 5M] [--mesh hex]
"""

import argparse
import glob
import json
import os
import statistics


def _load(directory):
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path) as f:
            data = json.load(f)
        runs[data["meta"]["config"]] = data
    return runs


def _table(records):
    """``{(mesh, size, form, filter): {"impl/mode": ms}}`` and the failed checks."""
    table, failed = {}, set()
    for r in records:
        key = (r["mesh"], r["size"], r["form"], r["filter"])
        if r["impl"] == "check":
            if not r["ok"]:
                failed.add(key)
            continue
        table.setdefault(key, {})[f'{r["impl"]}/{r["mode"]}'] = r["median"] * 1e3
    return table, failed


def _fmt(x):
    return "       -" if x is None else (f"{x:8.1f}" if x >= 10 else f"{x:8.2f}")


def _ratio(a, b):
    return "    -" if a is None or not b else f"{a / b:5.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("directory")
    parser.add_argument("--size", help="10k, 100k, 1M, 5M or 112k (squareBend)")
    parser.add_argument("--mesh", help="hex, tet or squareBend")
    args = parser.parse_args()
    runs = _load(args.directory)
    cpus = sorted((int(c[3:]), c) for c in runs if c.startswith("cpu"))
    if not cpus:
        raise SystemExit(f"no cpu<N>.json in {args.directory}")
    one = runs.get("cpu1")
    many = runs[cpus[-1][1]] if cpus[-1][0] > 1 else None
    gpus = [c for c in runs if not c.startswith("cpu")]
    n = cpus[-1][0]

    for config, data in runs.items():
        m = data["meta"]
        extra = f", GPU {m['gpu']}" if m.get("gpu") else ""
        print(f"{config}: {m['processor']}{extra}; Tack {m['tack']} on {m['tack backend']}, "
              f"VTK {m['vtk']} ({m['vtk smp']}, {m['vtk threads']} threads), {m['date']}")

    tables = {c: _table(d["records"]) for c, d in runs.items()}
    failed = set().union(*(f for _, f in tables.values()))
    if failed:
        print("\noutputs differ (timings not comparable):",
              ", ".join(" ".join(k) for k in sorted(failed)))

    keys = sorted({k for t, _ in tables.values() for k in t},
                  key=lambda k: (k[0], {"10k": 0, "100k": 1, "1M": 2, "5M": 3}.get(k[1], 4),
                                 k[2], k[3]))
    header = (f"{'mesh':10s} {'size':5s} {'form':10s} {'filter':14s} |"
              f" {'VTK1':>8s} {'Tack1':>8s} {'r':>5s} |"
              f" {f'VTK{n}':>8s} {f'Tack{n}':>8s} {'r':>5s} {'fresh':>8s} {'r':>5s}")
    for gpu in gpus:
        header += f" | {gpu:>8s} {'r':>5s}"
    print("\nmedian ms; r = Tack / VTK (below 1: Tack faster); GPUs against VTK on "
          f"{n} threads\n" + header)
    speedups = {"vtk": [], "tack": []}
    for key in keys:
        if (args.size and key[1] != args.size) or (args.mesh and key[0] != args.mesh):
            continue
        get = lambda c, col: tables[c][0].get(key, {}).get(col) if c in tables else None
        v1, t1 = get("cpu1", "vtk/warm"), get("cpu1", "tack/warm")
        cn = cpus[-1][1] if many else None
        vn, tn, fn = get(cn, "vtk/warm"), get(cn, "tack/warm"), get(cn, "tack/fresh")
        line = (f"{key[0]:10s} {key[1]:5s} {key[2]:10s} {key[3]:14s} |"
                f" {_fmt(v1)} {_fmt(t1)} {_ratio(t1, v1)} |"
                f" {_fmt(vn)} {_fmt(tn)} {_ratio(tn, vn)} {_fmt(fn)} {_ratio(fn, vn)}")
        for gpu in gpus:
            g = get(gpu, "tack/warm")
            line += f" | {_fmt(g)} {_ratio(g, vn)}"
        print(line)
        if key[3] != "import" and one and many:
            for impl, a, b in (("vtk", v1, vn), ("tack", t1, tn)):
                if a and b:
                    speedups[impl].append(a / b)
    if speedups["vtk"] and speedups["tack"]:
        print(f"\nspeedup from 1 to {n} threads, median: VTK "
              f"{statistics.median(speedups['vtk']):.1f}x, Tack "
              f"{statistics.median(speedups['tack']):.1f}x")


if __name__ == "__main__":
    main()

"""Hexahedra against the same hexahedra as polyhedra, through the algorithms that
run on polyhedra by their sub-tetrahedra: gradients (Green-Gauss), cell
centers, values at centers, the cell locator, probe, advect and streamlines.

The mesh is an unstructured grid of hexahedra, its points moved a little
(a hexahedron's faces are then not planar, and Newton takes more than one step),
and ``as_polyhedra`` of it. Each algorithm runs once cold (compiling, deriving
faces, cell points, links), then ``--repeats`` times warm, hexahedra and
polyhedra alternating so both see the same load; medians and minimums of the
warm runs. Checks that both paths agree on what they must: probe's valid points
and a linear field's values, the centers' and gradients' of a linear field.

    uv run --no-sync python benchmarks/polyhedra_compare.py --arch cpu
    uv run --no-sync python benchmarks/polyhedra_compare.py --arch metal   # outside the sandbox
"""

import argparse
import json
import os
import platform
import statistics
import time

import numpy as np

import tack
import tack.dataset as td


def hexahedra(n, jitter=0.15, seed=0):
    """An n^3 unstructured grid of hexahedra on [0, 1]^3, points moved by up to
    ``jitter`` of the spacing (boundary points only along the boundary)."""
    m = n + 1
    k, j, i = np.meshgrid(np.arange(m), np.arange(m), np.arange(m), indexing="ij")
    x = np.c_[i.ravel(), j.ravel(), k.ravel()].astype(np.float64) / n
    rng = np.random.default_rng(seed)
    move = rng.uniform(-jitter, jitter, x.shape) / n
    move[(x <= 0) | (x >= 1)] = 0.0
    x += move
    ci, cj, ck = (a.ravel() for a in np.meshgrid(np.arange(n), np.arange(n), np.arange(n),
                                                 indexing="ij"))
    base = ck + m * (cj + m * ci)      # cell (ck fastest) at its low corner, point x fastest
    corners = np.array([0, 1, 1 + m, m, m * m, m * m + 1, m * m + m + 1, m * m + m])
    connectivity = (base[:, None] + corners[None, :]).ravel()
    count = n ** 3
    topology = td.UnstructuredTopology(np.full(count, 12, np.uint8),
                                       np.arange(count + 1, dtype=np.int64) * 8, connectivity)
    return td.DataSet(topology, x.astype(np.float32))


def add_fields(data):
    x = data.positions().astype(np.float64)
    s = np.sin(3 * x[:, 0]) * x[:, 1] + x[:, 2] ** 2
    lin = 2 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2]
    v = np.c_[-(x[:, 1] - 0.5), x[:, 0] - 0.5, 0.2 * np.ones(len(x))]
    data.fields["s"] = td.Field(td.H1(data), tack.array(s.astype(np.float32)))
    data.fields["lin"] = td.Field(td.H1(data), tack.array(lin.astype(np.float32)))
    data.fields["v"] = td.Field(td.H1(data), tack.array(v.astype(np.float32), tack.vec3()))


def timed(fn):
    tack.sync()
    start = time.perf_counter()
    out = fn()
    tack.sync()
    return time.perf_counter() - start, out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", default="cpu")
    parser.add_argument("--cells", type=int, default=64, help="cells along each axis")
    parser.add_argument("--probes", type=int, default=200_000)
    parser.add_argument("--seeds", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", help="write the results as JSON here")
    args = parser.parse_args()
    tack.set_default_device(args.arch)
    load = os.getloadavg()

    hexes = hexahedra(args.cells)
    seconds, poly = timed(lambda: td.as_polyhedra(hexes))
    add_fields(hexes)
    add_fields(poly)
    rng = np.random.default_rng(1)
    probes = rng.uniform(0.0, 1.0, (args.probes, 3))
    seeds = 0.5 + rng.uniform(-0.3, 0.3, (args.seeds, 3)) * [1, 1, 0.5]
    lines = seeds[: max(args.seeds // 10, 1)]
    locators = {}

    def locator(d):
        return locators[id(d)]

    algorithms = {
        "gradients (cells)": lambda d: td.gradients(d, "s"),
        "gradients (points)": lambda d: td.gradients(d, "s", at="points"),
        "flow_quantities (with gradients of v)":
            lambda d: td.flow_quantities(d, td.gradients(d, "v")),
        "cell_centers": lambda d: td.algorithms.cell_centers(d),
        "values_at_centers": lambda d: td.algorithms.values_at_centers(d, "s"),
        "CellLocator (build)": lambda d: td.CellLocator(d),
        f"probe ({args.probes} points)": lambda d: td.probe(d, probes, fields=["s", "lin"],
                                                            locator=locator(d)),
        f"advect ({args.seeds} seeds, 50 steps)":
            lambda d: td.advect(d, "v", seeds, 0.01, 50, locator=locator(d)),
        f"streamlines ({len(lines)} seeds, 100 steps)":
            lambda d: td.streamlines(d, "v", lines, 0.01, 100, locator=locator(d)),
    }
    for d in (hexes, poly):
        locators[id(d)] = td.CellLocator(d)

    results = {}
    outputs = {}
    for name, fn in algorithms.items():
        cold = {}
        warm = {"hexahedra": [], "polyhedra": []}
        for label, d in (("hexahedra", hexes), ("polyhedra", poly)):
            cold[label], outputs[(name, label)] = timed(lambda d=d: fn(d))
        for _ in range(args.repeats):
            for label, d in (("hexahedra", hexes), ("polyhedra", poly)):
                warm[label].append(timed(lambda d=d: fn(d))[0])
        results[name] = {label: {"cold": cold[label], "median": statistics.median(warm[label]),
                                 "min": min(warm[label])} for label in warm}
        print(f"{name:45s} hex {results[name]['hexahedra']['median'] * 1e3:9.2f} ms   "
              f"poly {results[name]['polyhedra']['median'] * 1e3:9.2f} ms", flush=True)

    # What both paths must agree on.
    checks = {}
    p = next(k for k in algorithms if k.startswith("probe"))
    a, b = outputs[(p, "hexahedra")], outputs[(p, "polyhedra")]
    va, vb = a.fields["valid"].values.numpy() == 1, b.fields["valid"].values.numpy() == 1
    lin = 2 * probes[:, 0] - probes[:, 1] + 0.5 * probes[:, 2]
    checks["probe: same valid points"] = float((va == vb).mean())
    checks["probe: max |lin - exact|, hexahedra"] = float(
        np.abs(a.fields["lin"].values.numpy()[va] - lin[va]).max())
    checks["probe: max |lin - exact|, polyhedra"] = float(
        np.abs(b.fields["lin"].values.numpy()[vb] - lin[vb]).max())
    g = [td.gradients(d, "lin").values.numpy() for d in (hexes, poly)]
    checks["gradients of lin: max error, hexahedra"] = float(np.abs(g[0] - [2, -1, 0.5]).max())
    checks["gradients of lin: max error, polyhedra"] = float(np.abs(g[1] - [2, -1, 0.5]).max())
    adv = next(k for k in algorithms if k.startswith("advect"))
    checks["advect: max distance between the paths' ends"] = float(np.linalg.norm(
        outputs[(adv, "hexahedra")].positions() - outputs[(adv, "polyhedra")].positions(),
        axis=1).max())

    meta = {"arch": args.arch, "cells": args.cells ** 3, "points": hexes.num_points,
            "faces": poly.topology.num_faces, "probes": args.probes, "seeds": args.seeds,
            "repeats": args.repeats, "as_polyhedra_seconds": seconds,
            "load_before": load, "load_after": os.getloadavg(),
            "machine": platform.platform(), "commit": os.popen("git rev-parse --short HEAD").read().strip()}
    print()
    print(f"{args.arch}: {args.cells}^3 = {args.cells ** 3} cells, {hexes.num_points} points; "
          f"as_polyhedra {seconds:.2f} s; load {load[0]:.1f} -> {meta['load_after'][0]:.1f}")
    print()
    print("| algorithm | hexahedra (ms) | polyhedra (ms) | polyhedra / hexahedra | cold hex / poly (s) |")
    print("|---|---:|---:|---:|---:|")
    for name, r in results.items():
        h, q = r["hexahedra"], r["polyhedra"]
        print(f"| {name} | {h['median'] * 1e3:.2f} | {q['median'] * 1e3:.2f} | "
              f"{q['median'] / h['median']:.2f} | {h['cold']:.2f} / {q['cold']:.2f} |")
    print()
    for name, value in checks.items():
        print(f"- {name}: {value:.3g}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"meta": meta, "results": results, "checks": checks}, f, indent=2)


if __name__ == "__main__":
    main()

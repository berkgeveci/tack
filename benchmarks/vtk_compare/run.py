"""Compare Tack's dataset filters with VTK's, on the same meshes and the same threads.

Each configuration runs in its own process, because both systems fix their
threading (and Tack its backend) at start-up:

  cpu1    VTK Sequential,            Tack CPU, 1 thread
  cpu8    VTK STDThread, 8 threads,  Tack CPU, 8 threads
  metal   Tack Metal; VTK runs only to check Tack's outputs (compare with cpu8)

For every mesh, size, form (shape-based or polyhedral) and filter pair:

1. both sides run once and their outputs are checked against each other
   (``pairs.check``); a pair that disagrees is reported, and its timings are
   marked as not comparable;
2. Tack's first call in the process is timed as ``cold`` (it compiles);
3. VTK is timed as ``Modified(); Update()``, repeated;
4. Tack is timed -- through ``tack.sync()``, since Metal queues launches -- two ways: ``warm`` -- the same dataset each run, so what it
   derived on earlier runs (faces, edges, launch groups) is kept, as in a
   pipeline -- and ``fresh`` -- a new topology each run, built from the same
   arrays outside the timing, holding only what VTK's grid holds.

Each measurement repeats for ``--budget`` seconds (at least 3 and at most 30
runs) and records every time; reports use the median. Converting a VTK grid
to Tack, and shape cells to polyhedra, are timed as ``import``.

Usage:
  python benchmarks/vtk_compare/run.py --config cpu1|cpu8|metal|all
         [--sizes 10k,100k,1M,5M] [--meshes hex,tet,squareBend]
         [--filters contour,slice,...] [--budget 1.0] [--output DIR]
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SQUARE_BEND = os.path.expanduser(
    "~/Data/VTK/Data/OpenFOAM/squareBend/squareBend.foam")
CONFIGS = {"cpu1": ("cpu", 1), "cpu8": ("cpu", 8), "metal": ("metal", None)}


def _measure(run, setup=None, budget=1.0, least=3, most=30):
    """Times of ``run(setup())`` repeated for about ``budget`` seconds; the setup
    and the release of each result are outside the timing."""
    times = []
    began = time.perf_counter()
    while len(times) < least or (time.perf_counter() - began < budget and len(times) < most):
        argument = setup() if setup else None
        gc.collect()
        start = time.perf_counter()
        result = run(argument)
        times.append(time.perf_counter() - start)
        del result
    return times


def _record(records, **fields):
    times = fields.get("times")
    if times:
        fields["median"] = float(np.median(times))
    records.append(fields)
    label = " ".join(str(fields[k]) for k in ("mesh", "form", "size", "filter", "impl", "mode")
                     if k in fields)
    if times:
        print(f"  {label}: {1e3 * fields['median']:.3f} ms ({len(times)} runs)", flush=True)
    else:
        print(f"  {label}: {fields.get('note', '')}", flush=True)


def _meta(config):
    from vtkmodules.vtkCommonCore import vtkSMPTools, vtkVersion

    import tack

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "packages"], cwd=HERE,
                           capture_output=True, text=True).stdout.strip()
    smp = vtkSMPTools()
    return {"config": config, "date": time.strftime("%Y-%m-%d %H:%M"),
            "machine": platform.platform(), "processor": _processor(),
            "python": platform.python_version(), "numpy": np.__version__,
            "vtk": vtkVersion.GetVTKVersionFull(), "vtk smp": smp.GetBackend(),
            "vtk threads": smp.GetEstimatedNumberOfThreads(),
            "tack": sha + ("+changes" if dirty else ""), "tack backend": tack.current_arch()
            if hasattr(tack, "current_arch") else CONFIGS[config][0],
            "tack threads": CONFIGS[config][1]}


def _processor():
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return platform.processor()


def _start(config):
    """Fix both systems' threading and Tack's backend for this process."""
    from vtkmodules.vtkCommonCore import vtkSMPTools

    arch, threads = CONFIGS[config]
    smp = vtkSMPTools()
    if threads == 1:
        smp.SetBackend("Sequential")
    else:
        smp.SetBackend("STDThread")
        smp.Initialize(threads or 8)
    import tack

    if arch == "cpu":
        tack.init(arch=tack.cpu, num_threads=threads)
    else:
        tack.init(arch=getattr(tack, arch))


def _meshes(names, sizes):
    import meshes as m

    for name in names:
        if name == "squareBend":
            if os.path.exists(SQUARE_BEND):
                yield "112k", m.square_bend(SQUARE_BEND)
            else:
                print(f"skipping squareBend: {SQUARE_BEND} is not here")
            continue
        build = {"hex": m.hexahedra, "tet": m.tetrahedra}[name]
        for size in sizes:
            yield size, build(m.SIZES[size])


def run_config(config, sizes, mesh_names, filters, budget):
    _start(config)
    import meshes as m
    import pairs as p

    import tack
    from tack.interop.vtk import vtk_to_dataset

    time_vtk = config != "metal"
    records, seen = [], set()
    for size, mesh in _meshes(mesh_names, sizes):
        print(f"\n{mesh.name} {size}: {mesh.num_cells} cells, {mesh.num_points} points",
              flush=True)
        common = {"mesh": mesh.name, "size": size, "cells": mesh.num_cells}
        grids = {"shape": m.vtk_shape_grid(mesh), "polyhedral": m.vtk_polyhedral_grid(mesh)}
        shape = m.TackMesh(mesh, "shape")
        # What it costs to bring VTK's grid into Tack, and shape cells to polyhedra.
        _record(records, **common, form="shape", filter="import", impl="tack", mode="warm",
                times=_measure(lambda _: vtk_to_dataset(grids["shape"]), budget=budget))
        _record(records, **common, form="polyhedral", filter="import", impl="tack",
                mode="warm",
                times=_measure(lambda _: (m.TackMesh(mesh, "polyhedral"), tack.sync()),
                               budget=budget))
        tack_meshes = {"shape": shape, "polyhedral": m.TackMesh(mesh, "polyhedral")}
        for form in ("shape", "polyhedral"):
            for pair in p.PAIRS:
                if form not in pair.forms or (filters and pair.name not in filters):
                    continue
                key = {**common, "form": form, "filter": pair.name}
                grid = m.vtk_subset(grids[form], pair.point_fields, pair.cell_fields)
                source = tack_meshes[form]
                fields = pair.point_fields + pair.cell_fields
                vtk_filter = pair.vtk(grid, mesh, form)
                vtk_filter.Update()
                # Tack's first call in this process compiles: timed as cold.
                data = source.dataset(fields)

                def run_tack(d, pair=pair):
                    # Metal queues launches: the timing ends when they have run.
                    result = pair.tack(d, mesh)
                    tack.sync()
                    return result
                start = time.perf_counter()
                result = run_tack(data)
                first = time.perf_counter() - start
                if (form, pair.name) not in seen:
                    seen.add((form, pair.name))
                    _record(records, **key, impl="tack", mode="cold", times=[first])
                try:
                    ok, details = p.check(pair, vtk_filter.GetOutput(), result)
                except Exception as error:
                    ok, details = False, {"error": f"{type(error).__name__}: {error}"}
                del result
                if not ok:
                    print(f"  ** {form} {pair.name}: outputs differ: {details}", flush=True)
                records.append({**key, "impl": "check", "ok": ok, "details": details})
                if time_vtk:
                    for impl, build in (("vtk", pair.vtk), *pair.others.items()):
                        f = vtk_filter if impl == "vtk" else build(grid, mesh, form)

                        def update(_, f=f):
                            f.Modified()
                            f.Update()
                        _record(records, **key, impl=impl, mode="warm", comparable=ok,
                                times=_measure(update, budget=budget))
                _record(records, **key, impl="tack", mode="warm", comparable=ok,
                        times=_measure(lambda _: run_tack(data), budget=budget))
                _record(records, **key, impl="tack", mode="fresh", comparable=ok,
                        times=_measure(run_tack,
                                       setup=lambda: source.dataset(fields, fresh=True),
                                       budget=budget))
                del vtk_filter, grid, data
        del grids, tack_meshes, shape
        gc.collect()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="all", choices=[*CONFIGS, "all"])
    parser.add_argument("--sizes", default="10k,100k,1M,5M")
    parser.add_argument("--meshes", default="hex,tet,squareBend")
    parser.add_argument("--filters", default="", help="comma-separated pair names")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="seconds of repeated runs per measurement")
    parser.add_argument("--output", default=".", help="directory for <config>.json")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.config == "all":
        for config in CONFIGS:
            command = [sys.executable, __file__, "--config", config, "--sizes", args.sizes,
                       "--meshes", args.meshes, "--filters", args.filters,
                       "--budget", str(args.budget), "--output", args.output]
            subprocess.run(command, check=True)
        return
    sys.path.insert(0, HERE)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    records = run_config(args.config, args.sizes.split(","), args.meshes.split(","), filters,
                         args.budget)
    path = os.path.join(args.output, f"{args.config}.json")
    with open(path, "w") as out:
        json.dump({"meta": _meta(args.config), "records": records}, out, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""Compare Tack's dataset filters with VTK's, on the same meshes and the same threads.

Each configuration runs in its own process, because both systems fix their
threading (and Tack its backend) at start-up:

  cpu1    VTK Sequential,            Tack CPU, 1 thread
  cpuN    VTK STDThread, N threads,  Tack CPU, N threads (any N)
  cuda, hip, level_zero, metal
          Tack on that GPU; VTK runs only to check Tack's outputs, on
          --vtk-threads threads -- compare with the cpuN run of as many

``--config`` takes a comma-separated list, each run in a process of its own;
the default is cpu1,cpu8,metal on macOS and cpu1,cpu<logical cores>,cuda
elsewhere. ``report.py`` prints the comparison from the output directory.

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
  python benchmarks/vtk_compare/run.py [--config cpu1,cpu16,cuda]
         [--sizes 10k,100k,1M,5M] [--meshes hex,tet,squareBend]
         [--filters contour,slice,...] [--budget 1.0] [--output DIR]
         [--square-bend PATH/squareBend.foam] [--vtk-threads N]

squareBend is OpenFOAM's tutorial case, as in VTK's test data
(Data/OpenFOAM/squareBend); give its .foam file with --square-bend or
TACK_SQUARE_BEND, or leave it out of --meshes.
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
SQUARE_BEND = os.environ.get("TACK_SQUARE_BEND") or os.path.expanduser(
    "~/Data/VTK/Data/OpenFOAM/squareBend/squareBend.foam")
GPUS = ("cuda", "hip", "level_zero", "metal")


def _config(name):
    """``(arch, threads)`` of a configuration name: cpu<N>, or a GPU arch."""
    if name.startswith("cpu") and name[3:].isdigit() and int(name[3:]) > 0:
        return "cpu", int(name[3:])
    if name in GPUS:
        return name, None
    raise ValueError(f"unknown configuration {name!r}: cpu<N> or one of {', '.join(GPUS)}")


def _default_configs():
    if sys.platform == "darwin":
        return "cpu1,cpu8,metal"
    return f"cpu1,cpu{os.cpu_count()},cuda"


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


def _meta(config, vtk_threads):
    from vtkmodules.vtkCommonCore import vtkSMPTools, vtkVersion


    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "packages"], cwd=HERE,
                           capture_output=True, text=True).stdout.strip()
    smp = vtkSMPTools()
    arch, threads = _config(config)
    from tack.runtime.dispatch import get_backend

    return {"config": config, "date": time.strftime("%Y-%m-%d %H:%M"),
            "machine": platform.platform(), "processor": _processor(),
            "gpu": _gpu() if arch != "cpu" else None,
            "python": platform.python_version(), "numpy": np.__version__,
            "vtk": vtkVersion.GetVTKVersionFull(), "vtk smp": smp.GetBackend(),
            "vtk threads": smp.GetEstimatedNumberOfThreads(),
            "tack": sha + ("+changes" if dirty else ""), "tack backend": get_backend().label,
            "tack threads": threads, "vtk threads for checks": vtk_threads}


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _processor():
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    try:
        with open("/proc/cpuinfo") as info:
            for line in info:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _gpu():
    return (_run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
            or _run(["rocm-smi", "--showproductname"]) or "")


def _start(config, vtk_threads):
    """Fix both systems' threading and Tack's backend for this process."""
    from vtkmodules.vtkCommonCore import vtkSMPTools

    arch, threads = _config(config)
    smp = vtkSMPTools()
    if threads == 1:
        smp.SetBackend("Sequential")
    else:
        smp.SetBackend("STDThread")
        smp.Initialize(threads or vtk_threads)
    import tack

    if arch == "cpu":
        tack.init(arch=tack.cpu, num_threads=threads)
    else:
        tack.init(arch=getattr(tack, arch))


def _meshes(names, sizes, square_bend):
    import meshes as m

    for name in names:
        if name == "squareBend":
            if os.path.exists(square_bend):
                yield "112k", m.square_bend(square_bend)
            else:
                print(f"skipping squareBend: {square_bend} is not here (--square-bend)")
            continue
        build = {"hex": m.hexahedra, "tet": m.tetrahedra}[name]
        for size in sizes:
            yield size, build(m.SIZES[size])


def run_config(config, sizes, mesh_names, filters, budget, vtk_threads,
               square_bend=SQUARE_BEND):
    _start(config, vtk_threads)
    import meshes as m
    import pairs as p

    import tack
    from tack.interop.vtk import vtk_to_dataset

    time_vtk = _config(config)[0] == "cpu"
    records, seen = [], set()
    for size, mesh in _meshes(mesh_names, sizes, square_bend):
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
    parser.add_argument("--config", default=_default_configs(),
                        help="comma-separated: cpu<N> and GPU archs (cuda, hip, level_zero, "
                             f"metal); default {_default_configs()}")
    parser.add_argument("--sizes", default="10k,100k,1M,5M")
    parser.add_argument("--meshes", default="hex,tet,squareBend")
    parser.add_argument("--filters", default="", help="comma-separated pair names")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="seconds of repeated runs per measurement")
    parser.add_argument("--output", default=".", help="directory for <config>.json")
    parser.add_argument("--square-bend", default=SQUARE_BEND,
                        help="squareBend.foam (or set TACK_SQUARE_BEND)")
    parser.add_argument("--vtk-threads", type=int, default=os.cpu_count(),
                        help="VTK's threads when it only checks a GPU run's outputs")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    for config in configs:
        _config(config)                       # refuse a bad name before anything runs
    if len(configs) > 1:
        for config in configs:
            command = [sys.executable, __file__, "--config", config, "--sizes", args.sizes,
                       "--meshes", args.meshes, "--filters", args.filters,
                       "--budget", str(args.budget), "--output", args.output,
                       "--square-bend", args.square_bend,
                       "--vtk-threads", str(args.vtk_threads)]
            subprocess.run(command, check=True)
        return
    sys.path.insert(0, HERE)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    records = run_config(configs[0], args.sizes.split(","), args.meshes.split(","), filters,
                         args.budget, args.vtk_threads, args.square_bend)
    path = os.path.join(args.output, f"{configs[0]}.json")
    with open(path, "w") as out:
        json.dump({"meta": _meta(configs[0], args.vtk_threads), "records": records}, out,
                  indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

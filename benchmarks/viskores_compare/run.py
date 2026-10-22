"""Tack on CUDA against Viskores on CUDA, reached through VTK's vtkm... filters.

Needs a VTK built with Viskores on CUDA (``VTK_USE_CUDA``); the PyPI wheel
has no Viskores. Point ``PYTHONPATH`` and ``LD_LIBRARY_PATH`` at its build,
leave any other ``vtk`` out of the environment, and run, e.g.:

    uv run --no-sync python benchmarks/viskores_compare/run.py --output results-viskores
    uv run --no-sync python benchmarks/viskores_compare/report.py results-viskores

The meshes are the VTK comparison's hexahedra and tetrahedra (``vtk_compare/
meshes.py``), shape-based only: Viskores has no polyhedra. For every mesh and
filter pair:

1. Viskores runs once (its first filter of each kind in the process is
   recorded as cold) and Tack runs once (cold, compiling). A Viskores filter
   that throws falls back to VTK's CPU filter with a warning; that is caught,
   recorded, and the pair is not timed.
2. Tack's output is checked against Viskores', and Viskores' against VTK's CPU
   filter where VTK has one. Only checked pairs are comparable.
3. Timed, warm, each repeated for ``--budget`` seconds:
   - ``viskores``: ``Modified(); Update()``. Each call converts VTK's host
     arrays to Viskores, runs on the GPU and converts the result back to VTK.
   - ``tack``: the filter on a dataset already on the GPU, the result left there
     (ended by ``tack.sync()``).
   - ``tack round trip``: Viskores' path -- VTK's grid copied to the GPU
     (``vtk_to_dataset``), the filter, and its result copied back to the host
     (``dataset_to_vtk``, or the values for a filter that makes an array).

``--resident`` compares the two with their data on the GPU, on a VTK that has
DLPack and ``vtkmDeviceAdapter.SetForceVtkmDataArrayOutput`` (the
``cuda-wheel-gpu-output`` branch). VTK's grid is then a view of Tack's own
device arrays through DLPack -- the same int32 connectivity and offsets, the
same ``p`` and ``c`` -- so Viskores copies nothing to the GPU, and with the flag
its output stays there too. Tack's ``warm`` and Viskores' ``warm`` are then
the same work, but for one thing: VTK hands Viskores a new dataset on every
Update, so Viskores derives point links and faces on each call, where Tack's
``warm`` reuses those it cached on the topology. ``fresh`` times Tack on a new
topology over the same device arrays each call, deriving them too. The
round trip and the checks against VTK's CPU filters (made
by the default mode) are left out; Viskores' output is read for the checks
through ``vtkmConvertToVTKArrays``, outside the timing. A Viskores filter that
fails falls back to VTK's CPU filter, which reads device memory and crashes,
so each pair first runs once at 10k in a child process (``--preflight``) and
a pair that does not survive it is left out and named in the results.
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
sys.path.insert(0, os.path.join(HERE, os.pardir, "vtk_compare"))   # meshes.py
sys.path.insert(0, HERE)

FALLBACK = "Falling back"


def _measure(run, budget=1.0, least=3, most=30, setup=None):
    """Times of ``run()``, or of ``run(setup())`` with the setup outside the timing."""
    times = []
    began = time.perf_counter()
    while len(times) < least or (time.perf_counter() - began < budget and len(times) < most):
        argument = setup() if setup else None
        gc.collect()
        start = time.perf_counter()
        result = run(argument) if setup else run()
        times.append(time.perf_counter() - start)
        del result
    return times


def _record(records, **fields):
    times = fields.get("times")
    if times:
        fields["median"] = float(np.median(times))
    records.append(fields)
    label = " ".join(str(fields[k]) for k in ("mesh", "size", "filter", "impl", "mode")
                     if k in fields)
    if times:
        print(f"  {label}: {1e3 * fields['median']:.3f} ms ({len(times)} runs)", flush=True)
    else:
        print(f"  {label}: {fields.get('note', '')}", flush=True)


class _Warnings:
    """VTK warnings and errors raised by one filter: how a Viskores fallback shows."""

    def __init__(self, algorithm):
        self.messages = []
        for event in ("WarningEvent", "ErrorEvent"):
            algorithm.AddObserver(event, self._seen)

    def _seen(self, caller, event):
        self.messages.append(event)


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _meta(vtk_threads):
    import vtkmodules
    from vtkmodules.vtkCommonCore import vtkSMPTools, vtkVersion

    from tack.runtime.dispatch import get_backend

    sha = _run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"])
    dirty = _run(["git", "-C", HERE, "status", "--porcelain", "--", "packages"])
    processor = platform.processor()
    try:
        with open("/proc/cpuinfo") as info:
            processor = next(line.split(":", 1)[1].strip() for line in info
                             if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"date": time.strftime("%Y-%m-%d %H:%M"), "machine": platform.platform(),
            "processor": processor,
            "gpu": _run(["nvidia-smi", "--query-gpu=name,driver_version",
                         "--format=csv,noheader"]),
            "python": platform.python_version(), "numpy": np.__version__,
            "vtk": vtkVersion.GetVTKVersionFull(), "vtk path": os.path.dirname(vtkmodules.__file__),
            "viskores device": os.environ.get("VISKORES_DEVICE"),
            "vtk smp (reference checks)": vtkSMPTools().backend,
            "vtk threads (reference checks)": vtk_threads,
            "tack": sha + ("+changes" if dirty else ""), "tack backend": get_backend().label}


def _start(vtk_threads, resident):
    # Viskores reads its device when VTK first initializes it, at the first filter.
    os.environ.setdefault("VISKORES_DEVICE", "Cuda")
    import vtkmodules.vtkAcceleratorsVTKmFilters  # noqa: F401 -- refuse a VTK without Viskores
    from vtkmodules.vtkCommonCore import vtkSMPTools

    if resident:
        from vtkmodules.vtkAcceleratorsVTKmCore import vtkmDeviceAdapter

        if not hasattr(vtkmDeviceAdapter, "SetForceVtkmDataArrayOutput"):
            sys.exit("--resident needs a VTK with vtkmDeviceAdapter.SetForceVtkmDataArrayOutput")
        vtkmDeviceAdapter.SetForceVtkmDataArrayOutput(True)

    smp = vtkSMPTools()
    smp.SetBackend("STDThread")
    smp.Initialize(vtk_threads)
    import tack

    tack.set_default_device("cuda")


def _host_output(pair, result):
    """What Viskores hands back, from Tack's result: a VTK dataset, or the values."""
    from tack.interop.vtk import dataset_to_vtk

    if pair.output == "array":
        return pair.tack_values(result)
    return dataset_to_vtk(result)


def _device_grid(source, mesh, positions, point_fields, cell_fields):
    """A vtkUnstructuredGrid viewing Tack's device arrays through DLPack: nothing
    copied, Tack's int32 ids. ``positions`` is an (n, 3) field: a vector field
    exports as one flat component."""
    from vtkmodules.util.dlpack_support import dlpack_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid

    grid = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetData(dlpack_to_vtk(positions, name="points"))
    grid.SetPoints(points)
    cells = vtkCellArray()
    cells.SetData(dlpack_to_vtk(source.topology.offsets),
                  dlpack_to_vtk(source.topology.connectivity))
    grid.SetCells(mesh.cell_type, cells)
    for data, names in ((grid.GetPointData(), point_fields), (grid.GetCellData(), cell_fields)):
        for name in names:
            data.AddArray(dlpack_to_vtk(source.values[name][1], name=name))
        if names:
            data.SetActiveScalars(names[0])
    return grid


def _to_host(output):
    """Viskores' device-resident output as plain VTK arrays, for the checks."""
    import vtkmodules.vtkAcceleratorsVTKmFilters as vf

    convert = vf.vtkmConvertToVTKArrays()
    convert.SetInputData(output)
    convert.Update()
    return convert.GetOutput()


def run(sizes, mesh_names, filters, budget, vtk_threads, resident=False):
    _start(vtk_threads, resident)
    import meshes as m
    import pairs as p

    import tack
    from tack.interop.vtk import vtk_to_dataset

    records, seen_tack, seen_viskores = [], set(), set()
    for name in mesh_names:
        build = {"hex": m.hexahedra, "tet": m.tetrahedra}[name]
        for size in sizes:
            mesh = build(m.SIZES[size])
            print(f"\n{mesh.name} {size}: {mesh.num_cells} cells, {mesh.num_points} points",
                  flush=True)
            common = {"mesh": mesh.name, "size": size, "cells": mesh.num_cells}
            full = None if resident else m.vtk_shape_grid(mesh)
            source = m.TackMesh(mesh, "shape")
            if resident:
                positions = tack.empty((mesh.num_points, 3))
                tack.copyto(positions, mesh.points.astype(np.float32))
                tack.sync()
            for pair in p.PAIRS:
                if filters and pair.name not in filters:
                    continue
                key = {**common, "filter": pair.name}
                if resident:
                    grid = _device_grid(source, mesh, positions, pair.point_fields,
                                        pair.cell_fields)
                else:
                    grid = m.vtk_subset(full, pair.point_fields, pair.cell_fields)
                fields = pair.point_fields + pair.cell_fields

                viskores = pair.viskores(grid, mesh)
                warnings = _Warnings(viskores)
                start = time.perf_counter()
                viskores.Update()
                first = time.perf_counter() - start
                if pair.name not in seen_viskores:
                    seen_viskores.add(pair.name)
                    _record(records, **key, impl="viskores", mode="cold", times=[first])
                if warnings.messages:
                    note = "warned (falls back to VTK on the CPU?): " + \
                        "; ".join(warnings.messages)[:300]
                    _record(records, **key, impl="viskores", mode="fallback", note=note)

                data = source.dataset(fields)

                def run_tack(pair=pair, data=data):
                    result = pair.tack(data, mesh)
                    tack.sync()
                    return result

                start = time.perf_counter()
                result = run_tack()
                first = time.perf_counter() - start
                if pair.name not in seen_tack:
                    seen_tack.add(pair.name)
                    _record(records, **key, impl="tack", mode="cold", times=[first])
                try:
                    if resident:
                        # Read back from a second filter on its own view of the same
                        # arrays: vtkmConvertToVTKArrays converts in place the points
                        # and cells an output shares with its input, which would leave
                        # the timed filter's input on the host.
                        separate = pair.viskores(
                            _device_grid(source, mesh, positions, pair.point_fields,
                                         pair.cell_fields), mesh)
                        separate.Update()
                        output = _to_host(separate.GetOutput())
                        del separate
                    else:
                        output = viskores.GetOutput()
                    ok, details = p.check(pair, output, result)
                except Exception as error:
                    ok, details = False, {"error": f"{type(error).__name__}: {error}"}
                del result
                if not ok:
                    print(f"  ** {pair.name}: Tack and Viskores differ: {details}", flush=True)
                records.append({**key, "impl": "check", "against": "viskores", "ok": ok,
                                "details": details})
                if pair.vtk is not None and not resident:
                    reference = pair.vtk(grid, mesh)
                    reference.Update()
                    try:
                        ref_ok, ref_details = p.check_reference(pair, viskores.GetOutput(),
                                                                reference.GetOutput())
                    except Exception as error:
                        ref_ok, ref_details = False, {"error": f"{type(error).__name__}: {error}"}
                    if not ref_ok:
                        print(f"  .. {pair.name}: Viskores and VTK's CPU filter differ: "
                              f"{ref_details}", flush=True)
                    records.append({**key, "impl": "check", "against": "vtk", "ok": ref_ok,
                                    "details": ref_details})
                    del reference
                comparable = ok and not warnings.messages

                def update(f=viskores):
                    f.Modified()
                    f.Update()
                    return
                _record(records, **key, impl="viskores", mode="warm", comparable=comparable,
                        times=_measure(update, budget=budget))
                if warnings.messages:          # a fallback in the timed runs, too
                    records[-1]["comparable"] = False
                _record(records, **key, impl="tack", mode="warm", comparable=comparable,
                        times=_measure(run_tack, budget=budget))
                if resident:
                    # VTK gives Viskores a new dataset on every Update, so it derives
                    # point links and faces each time; "fresh" gives Tack a new
                    # topology over the same device arrays each time, likewise.
                    _record(records, **key, impl="tack", mode="fresh", comparable=comparable,
                            times=_measure(lambda d: run_tack(data=d), budget=budget,
                                           setup=lambda f=fields: source.dataset(f, fresh=True)))

                def round_trip(pair=pair, grid=grid):
                    result = pair.tack(vtk_to_dataset(grid), mesh)
                    return _host_output(pair, result)
                if not resident:
                    _record(records, **key, impl="tack", mode="round trip",
                            comparable=comparable, times=_measure(round_trip, budget=budget))
                del viskores, grid, data
                gc.collect()
            del full, source, mesh
            if resident:
                del positions
            gc.collect()
    return records


def _preflight(args, filters):
    """The pairs that survive one run at 10k on the device, each in its own
    process, and why each other one did not."""
    import tempfile

    import meshes as m  # noqa: F401 -- the same imports the child makes
    import pairs as p

    names = filters or [pair.name for pair in p.PAIRS]
    kept, dropped = [], {}
    for name in names:
        with tempfile.TemporaryDirectory() as scratch:
            command = [sys.executable, __file__, "--resident", "--preflight", "--sizes", "10k",
                       "--meshes", args.meshes, "--filters", name, "--budget", "0.01",
                       "--output", scratch]
            done = subprocess.run(command, capture_output=True, text=True)
            path = os.path.join(scratch, "viskores.json")
            if done.returncode != 0 or not os.path.exists(path):
                tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
                dropped[name] = f"exit {done.returncode}: " + " | ".join(tail)[-300:]
                continue
            with open(path) as result:
                fallbacks = [r["note"] for r in json.load(result)["records"]
                             if r.get("mode") == "fallback"]
            if fallbacks:
                dropped[name] = fallbacks[0]
            else:
                kept.append(name)
        print(f"preflight {name}: {'ok' if name in kept else dropped[name]}", flush=True)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sizes", default="10k,100k,1M,5M")
    parser.add_argument("--meshes", default="hex,tet")
    parser.add_argument("--filters", default="", help="comma-separated pair names")
    parser.add_argument("--budget", type=float, default=1.0,
                        help="seconds of repeated runs per measurement")
    parser.add_argument("--output", default=".", help="directory for viskores.json")
    parser.add_argument("--vtk-threads", type=int, default=os.cpu_count(),
                        help="VTK's threads for the CPU reference filters (checks only)")
    parser.add_argument("--resident", action="store_true",
                        help="both sides' data on the GPU (needs the cuda-wheel-gpu-output VTK)")
    parser.add_argument("--preflight", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    dropped = {}
    if args.resident and not args.preflight:
        filters, dropped = _preflight(args, filters)
        if not filters:
            sys.exit("no pair survived the preflight")
    records = run(args.sizes.split(","), args.meshes.split(","), filters, args.budget,
                  args.vtk_threads, resident=args.resident)
    meta = _meta(args.vtk_threads)
    meta["resident"] = args.resident
    if dropped:
        meta["left out by the preflight"] = dropped
    path = os.path.join(args.output, "viskores.json")
    with open(path, "w") as out:
        json.dump({"meta": meta, "records": records}, out, indent=1)
    print(f"\nwrote {path}", flush=True)
    if args.resident:
        # Viskores' teardown frees into a CUDA context that is already gone and
        # aborts; everything is written by now.
        os._exit(0)


if __name__ == "__main__":
    main()

"""Peak memory of each filter, Tack's and VTK's, on the meshes of ``run.py``.

Each (mesh, size, filter, system) runs in a process of its own: it builds the
mesh and that system's input (Tack's fresh dataset, holding only what VTK's
grid holds; VTK's grid with the filter's arrays), runs the filter once on a
small mesh of the same kind so compiling is not counted, then samples the
process's memory every 2 ms through one call on the large one. It reports

- ``input``: the input's memory, by its own accounting (Tack's arrays' bytes,
  VTK's ``GetActualMemorySize``);
- ``peak``: the most the process held during the call beyond what it held
  before it (temporaries and output);
- ``kept``: what it holds after the call while the output is alive (the
  output, and whatever the filter derived and kept on the input);
- the call's time.

The memory measured is what the allocator has handed out and not taken back
(``malloc_zone_statistics`` on macOS, ``mallinfo2`` on Linux): every live
NumPy, Tack (CPU) and VTK allocation, and none of what it caches after a
free. GPU memory is not counted.

Usage:
  python benchmarks/vtk_compare/memory.py [--sizes 1M,20M] [--meshes hex,tet]
         [--filters contour,...] [--threads 8] [--output memory.json]
"""

import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _in_use_reader():
    """A function giving the bytes the process's allocator has handed out and not
    taken back, now: live memory, not what the allocator caches after a free
    (a process's footprint counts that too, so a filter that reuses freed
    memory showed no growth)."""
    if sys.platform == "darwin":
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")

        class Statistics(ctypes.Structure):          # malloc_statistics_t
            _fields_ = [("blocks_in_use", ctypes.c_uint), ("size_in_use", ctypes.c_size_t),
                        ("max_size_in_use", ctypes.c_size_t),
                        ("size_allocated", ctypes.c_size_t)]

        libc.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(Statistics)]

        def read():
            stats = Statistics()
            libc.malloc_zone_statistics(None, ctypes.byref(stats))     # every zone
            return int(stats.size_in_use)
        return read
    libc = ctypes.CDLL("libc.so.6")

    class Info(ctypes.Structure):                    # struct mallinfo2
        _fields_ = [(n, ctypes.c_size_t) for n in (
            "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
            "uordblks", "fordblks", "keepcost")]

    libc.mallinfo2.restype = Info

    def read():
        info = libc.mallinfo2()
        return int(info.uordblks + info.hblkhd)      # in the heap, and mapped alone
    return read


def _release_free_memory():
    """Return the allocator's free pages to the system, so the baseline holds only
    what is in use: otherwise a filter that reuses freed memory shows no growth."""
    gc.collect()
    try:
        if sys.platform == "darwin":
            libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
            libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.malloc_zone_pressure_relief(None, 0)
        else:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class _Sampler:
    """The highest memory seen while running, sampled every ``interval`` seconds."""

    def __init__(self, read, interval=0.002):
        self.read, self.interval, self.peak = read, interval, 0
        self._stop = threading.Event()

    def __enter__(self):
        self.peak = self.read()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self.read())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.read())


def _tack_bytes(data):
    """The bytes of a dataset's topology arrays, positions and fields."""
    from tack.lang.array import Array

    seen, total = set(), 0

    def add(a):
        nonlocal total
        if isinstance(a, Array) and id(a) not in seen:
            seen.add(id(a))
            total += a.nbytes
    for value in vars(data.topology).values():
        add(value)
    add(data.geometry.values)
    for f in data.fields.values():
        add(f.values)
    return total


def measure_one(mesh_name, size, filter_name, system, threads):
    """One measurement, in this process."""
    sys.path.insert(0, HERE)
    import meshes as m
    import pairs as p

    import tack

    pair = next(q for q in p.PAIRS if q.name == filter_name)
    build = {"hex": m.hexahedra, "tet": m.tetrahedra}[mesh_name]
    fields = pair.point_fields + pair.cell_fields
    read = _in_use_reader()
    if system == "vtk":
        from vtkmodules.vtkCommonCore import vtkSMPTools

        smp = vtkSMPTools()
        smp.SetBackend("STDThread" if threads > 1 else "Sequential")
        if threads > 1:
            smp.Initialize(threads)

        def prepare(mesh):
            grid = m.vtk_subset(m.vtk_shape_grid(mesh), pair.point_fields, pair.cell_fields)
            return grid, grid.GetActualMemorySize() * 1024

        def run(grid, mesh):
            f = pair.vtk(grid, mesh, "shape")
            f.Update()
            return f, f.output
    else:
        tack.set_default_device("cpu", num_threads=threads)

        def prepare(mesh):
            data = m.TackMesh(mesh, "shape").dataset(fields, fresh=True)
            tack.sync()
            return data, _tack_bytes(data)

        def run(data, mesh):
            result = pair.tack(data, mesh)
            tack.sync()
            return result, None

    small = build(10_000)
    first, _ = prepare(small)
    run(first, small)                         # compiles, outside the measurement
    del first, small
    mesh = build(m.SIZES[size])
    source, input_bytes = prepare(mesh)
    _release_free_memory()
    before = read()
    with _Sampler(read) as sampler:
        start = time.perf_counter()
        result = run(source, mesh)
        seconds = time.perf_counter() - start
    _release_free_memory()
    kept = read() - before
    del result
    return {"mesh": mesh_name, "size": size, "cells": mesh.num_cells, "filter": filter_name,
            "system": system, "threads": threads, "input": input_bytes,
            "peak": sampler.peak - before, "kept": kept, "seconds": seconds}


def _child(args):
    """Run one measurement in a new process; its record, or the failure."""
    command = [sys.executable, os.path.abspath(__file__), "--one", *args]
    done = subprocess.run(command, capture_output=True, text=True)
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return {"error": f"exit {done.returncode}: {' | '.join(tail)}"}
    return json.loads(lines[-1])


def _gb(n):
    return f"{n / 2**20:8.0f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", nargs=5, metavar=("MESH", "SIZE", "FILTER", "SYSTEM", "THREADS"),
                        help=argparse.SUPPRESS)
    parser.add_argument("--sizes", default="1M,20M")
    parser.add_argument("--meshes", default="hex,tet")
    parser.add_argument("--filters", default="")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.one:
        mesh, size, name, system, threads = args.one
        print(json.dumps(measure_one(mesh, size, name, system, int(threads))), flush=True)
        return
    sys.path.insert(0, HERE)
    import pairs as p

    names = [q.name for q in p.PAIRS if not args.filters or q.name in args.filters.split(",")]
    records = []
    print(f"{'mesh':>4} {'size':>5} {'filter':<15} {'system':<5} {'input MB':>8} "
          f"{'peak MB':>8} {'kept MB':>8} {'s':>7}  peak/input", flush=True)
    for mesh in args.meshes.split(","):
        for size in args.sizes.split(","):
            for name in names:
                for system in ("vtk", "tack"):
                    r = _child([mesh, size, name, system, str(args.threads)])
                    r.update(mesh=mesh, size=size, filter=name, system=system)
                    records.append(r)
                    if "error" in r:
                        print(f"{mesh:>4} {size:>5} {name:<15} {system:<5} {r['error']}",
                              flush=True)
                        continue
                    print(f"{mesh:>4} {size:>5} {name:<15} {system:<5} {_gb(r['input'])}  "
                          f"{_gb(r['peak'])}  {_gb(r['kept'])} {r['seconds']:7.3f}  "
                          f"{r['peak'] / max(r['input'], 1):5.2f}", flush=True)
                    if args.output:
                        with open(args.output, "w") as f:
                            json.dump(records, f, indent=1)


if __name__ == "__main__":
    main()

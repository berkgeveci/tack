"""Per-launch and per-allocation trace of the vtk_compare filters on the CPU.

    python benchmarks/alloc_trace.py REPO THREADS SIZE [filters...]

Hex mesh, shape form, the fifth warm call of each filter. REPO is a
checkout whose ``benchmarks/vtk_compare`` supplies the meshes and the
filter pairs; the tack that runs is whatever this interpreter imports,
so run it with the venv of the build being traced. SIZE is one of
vtk_compare's sizes or 20M / 50M. Patches the CPU backend's dispatch,
``allocate_array``, ``resolve_variant``, ``tack.zeros`` and
``Field.fill`` to report, per filter:

  per launch      kernel, elements, wall ms, fanned out, minor faults, the
                  gap since the previous launch returned
  per allocation  bytes, dtype, time in allocate_array, the fill zeros or
                  Field.fill added, and the first kernel that bound the
                  field: written over its whole range or not, an atomic
                  target, or only read

The header line of each filter gives the totals: kernels, the time
between launches, allocation and fill, page faults.
"""
import collections
import os
import resource
import sys
import time

repo, threads, size = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, f"{repo}/benchmarks/vtk_compare")
want = sys.argv[4:] or ["contour", "slice", "threshold", "external faces",
                        "point to cell", "cell to point", "cell volume"]
os.environ["TACK_CPU_THREADS"] = str(threads)
import meshes as m
import numpy as np
import pairs as p

import tack
import tack.lang.array as fieldmod
import tack.runtime.cpu as cpu

tack.set_default_device("cpu", num_threads=threads)

SIZES = dict(m.SIZES, **{"20M": 20_000_000, "50M": 50_000_000})

state = {"on": False, "fanned": False, "variant": None, "eargs": None,
         "in_zeros": False, "in_fill": False}
launches = []          # (name, loop_end, ns, fanned, faults, gap_ns)
allocs = {}            # address -> record
alloc_order = []
seen_kernels = {}
last_end = [0]

_resolve = cpu.resolve_variant
_dispatch = cpu.CPUBackend._dispatch
_par = cpu.CPUBackend._parallel_execute
_alloc = cpu.CPUBackend.allocate_array
_zeros = fieldmod.zeros
_fill = fieldmod.Field.fill


def resolve_variant(backend, kernel, args, kwargs, **kw):
    out = _resolve(backend, kernel, args, kwargs, **kw)
    if state["on"]:
        state["variant"], state["eargs"] = out
    return out


def dispatch(self, compiled, kernel_args, loop_end, extents=()):
    if not state["on"]:
        return _dispatch(self, compiled, kernel_args, loop_end, extents)
    state["fanned"] = False
    name = getattr(compiled, "_func_name", "?")
    variant, eargs = state["variant"], state["eargs"]
    state["variant"] = state["eargs"] = None
    if variant is not None:
        written = {i for i, _ in variant.written_arrays}
        atomic = {i for i, _ in variant.atomic_targets}
        for i, a in enumerate(eargs):
            if not isinstance(a, fieldmod.Field):
                continue
            rec = allocs.get(a._buffer.address)
            if rec is None or rec["first"] is not None:
                continue
            elems = int(np.prod(a.shape))
            if i in written and i in atomic:
                role = "written+atomic"
            elif i in written:
                role = "written=loop" if loop_end == elems else "written<>loop"
            elif i in atomic:
                role = "atomic"
            else:
                role = "read"
            rec["first"] = (name, role, loop_end)
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    t0 = time.perf_counter_ns()
    gap = t0 - last_end[0]
    try:
        return _dispatch(self, compiled, kernel_args, loop_end, extents)
    finally:
        t1 = time.perf_counter_ns()
        f1 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        launches.append((name, loop_end, t1 - t0, state["fanned"], f1 - f0, gap))
        seen_kernels[name] = compiled
        last_end[0] = t1


def par(self, compiled, prefix, start, end, **kw):
    state["fanned"] = True
    return _par(self, compiled, prefix, start, end, **kw)


def allocate_array(self, dtype, shape, exportable=False):
    t0 = time.perf_counter_ns()
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    buf = _alloc(self, dtype, shape, exportable)
    if state["on"]:
        ns = time.perf_counter_ns() - t0
        faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0
        rec = {"bytes": buf.nbytes, "dtype": dtype.name, "shape": tuple(shape),
               "alloc_ns": ns, "faults": faults, "fill_ns": 0, "fills": 0,
               "via": "zeros" if state["in_zeros"] else "field", "first": None}
        allocs[buf.address] = rec
        alloc_order.append(rec)
        state["last_alloc"] = rec
    return buf


def zeros(*a, **kw):
    state["in_zeros"] = True
    try:
        return _zeros(*a, **kw)
    finally:
        state["in_zeros"] = False


def fill(self, value):
    t0 = time.perf_counter_ns()
    try:
        return _fill(self, value)
    finally:
        if state["on"]:
            rec = allocs.get(self._buffer.address)
            if rec is not None:
                rec["fill_ns"] += time.perf_counter_ns() - t0
                rec["fills"] += 1


cpu.resolve_variant = resolve_variant
cpu.CPUBackend._dispatch = dispatch
cpu.CPUBackend._parallel_execute = par
cpu.CPUBackend.allocate_array = allocate_array
fieldmod.zeros = zeros
tack.zeros = zeros
fieldmod.Field.fill = fill

mesh = m.hexahedra(SIZES[size])
source = m.TackMesh(mesh, "shape")
b = tack.runtime.dispatch.get_backend()
print(f"# tack {tack.__file__}; pool {getattr(b, 'pool_kind', 'executor')}; "
      f"numa {cpu._numa_available}; "
      f"MALLOC_MMAP_THRESHOLD_={os.environ.get('MALLOC_MMAP_THRESHOLD_')} "
      f"MALLOC_TRIM_THRESHOLD_={os.environ.get('MALLOC_TRIM_THRESHOLD_')}; "
      f"load {os.getloadavg()[0]:.2f}")
for pair in p.PAIRS:
    if pair.name not in want or "shape" not in pair.forms:
        continue
    data = source.dataset(pair.point_fields + pair.cell_fields)
    for _ in range(4):
        r = pair.tack(data, mesh); tack.sync(); del r
    launches.clear(); allocs.clear(); alloc_order.clear(); state["on"] = True
    f0 = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    t0 = time.perf_counter_ns(); last_end[0] = t0
    r = pair.tack(data, mesh); tack.sync()
    total = time.perf_counter_ns() - t0
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt - f0
    state["on"] = False
    del r
    kern = sum(t[2] for t in launches)
    a_ns = sum(a["alloc_ns"] for a in alloc_order)
    z_ns = sum(a["fill_ns"] for a in alloc_order)
    a_mb = sum(a["bytes"] for a in alloc_order) / 1e6
    a_faults = sum(a["faults"] for a in alloc_order)
    print(f"\n== {pair.name}: {threads} threads, {size}: total {total/1e6:.2f} ms = kernels "
          f"{kern/1e6:.2f} + between {(total-kern)/1e6:.2f}; {len(launches)} launches, "
          f"{sum(t[3] for t in launches)} fanned; alloc {a_ns/1e6:.2f} ms + fill "
          f"{z_ns/1e6:.2f} ms for {len(alloc_order)} fields, {a_mb:.1f} MB; "
          f"faults {faults} ({a_faults} in alloc)")
    agg = collections.OrderedDict()
    for name, n, ns, fanned, flt, gap in launches:
        a = agg.setdefault(name, [0, 0, 0, 0, 0, 0])
        a[0] += 1; a[1] += n; a[2] += ns; a[3] += fanned; a[4] += flt; a[5] += gap
    print(f"  {'kernel':<28}{'calls':>6}{'elems':>11}{'ms':>8}{'fan':>4}{'faults':>7}{'gap ms':>8}")
    for name, (c, n, ns, f, flt, gap) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print(f"  {name[:28]:<28}{c:>6}{n:>11}{ns/1e6:>8.2f}{f:>4}{flt:>7}{gap/1e6:>8.2f}")
    print(f"  {'allocation':<48}{'n':>3}{'MB':>8}{'alloc ms':>9}{'fill ms':>8}{'faults':>7}  first use")
    byk = collections.OrderedDict()
    for a in alloc_order:
        first = a["first"] or ("-", "unbound", 0)
        key = (a["via"], a["dtype"], a["bytes"] // max(1, int(np.prod(a["shape"]) or 1)),
               first[1], first[0], a["bytes"] >= 1 << 20)
        g = byk.setdefault(key, [0, 0, 0, 0, 0, 0, []])
        g[0] += 1; g[1] += a["bytes"]; g[2] += a["alloc_ns"]; g[3] += a["fill_ns"]
        g[4] += a["faults"]; g[5] += a["fills"]; g[6].append(int(np.prod(a["shape"])))
    for (via, dt, isz, role, kname, big), (n, byt, ans, fns, flt, fills, sizes) in sorted(
            byk.items(), key=lambda kv: -(kv[1][2] + kv[1][3])):
        el = f"{max(sizes):,}" if n == 1 else f"{min(sizes):,}..{max(sizes):,}"
        label = f"{via} {dt} x{el}"
        print(f"  {label:<48}{n:>3}{byt/1e6:>8.1f}{ans/1e6:>9.2f}{fns/1e6:>8.2f}{flt:>7}  "
              f"{role} by {kname}" + (f" ({fills} fills)" if fills > 1 else ""))
sys.stdout.flush()

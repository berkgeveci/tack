"""Tack against NVIDIA's warp-simdata, on the operators both have.

The meshes and fields are the VTK comparison's (``vtk_compare/meshes.py``:
hexahedra and tetrahedra filling [0, 1]^3, ``p`` on the points, ``c`` on the
cells), as are the contour value, the slice plane and its normal
(``vtk_compare/pairs.py``). For each mesh and operator:

1. both run once (the first call, compiling as it must);
2. their outputs are checked against each other: area, cell and point counts
   for surfaces; values for arrays; for probe, values where both found a cell,
   and which points each found (Tack's tolerance is 0.1% of a cell's diagonal,
   as ``vtkProbeFilter``'s; warp-simdata's is a world-space 1e-6, so Tack may
   take points just outside the mesh, and only those);
3. both are timed, warm (the same dataset each call, so each keeps what it
   caches on it) and fresh (a new dataset each call), medians of repeats.

Operators: contour, slice, external faces, cell centers, point-to-cell and
cell-to-point averaging, and probe (100k points in the box grown by 5%).

Runs on the CPU or CUDA (``--device``). On the CPU, Warp runs kernels on the
calling thread, one after another; Tack uses its thread pool, so CPU ratios
compare more than the libraries' designs. warp-simdata is not a dependency of
Tack: install it beside Tack, e.g.

    uv venv /tmp/warp && uv pip install -p /tmp/warp "warp-lang==1.18.0" "vtk>=9.7.1" \\
        git+https://github.com/NVIDIA/warp-simdata -e packages/tack-core[cpu] -e packages/tack-vis
    /tmp/warp/bin/python benchmarks/warp_compare/run.py --device cpu --sizes 10k 100k 1M

(add ``packages/tack-core[cuda]`` for ``--device cuda``). Results of the first
runs (yavin's CPU, 2026-10-23; mustafar-linux's RTX 4060 Ti, 2026-10-23) are
in the vault's "Comparison - Warp and warp-simdata".
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir, "vtk_compare"))   # meshes.py, pairs.py

import meshes
import pairs

PROBE_COUNT = 100_000


def setup(device):
    """Import both libraries on ``device`` ("cpu" or "cuda")."""
    import warp as wp

    wp.config.quiet = True
    wp.init()
    import tack

    tack.set_default_device(device)
    return wp, ("cpu" if device == "cpu" else "cuda:0")


def warp_dataset(wp, dev, mesh):
    from warp_simdata import AssociationType, Field
    from warp_simdata.data_models.vtk import unstructured_grid, vtk_shapes

    m, k = mesh.cells.shape
    kind = (vtk_shapes.VTKCellTypes.VTK_HEXAHEDRON if mesh.cell_type == meshes.VTK_HEXAHEDRON
            else vtk_shapes.VTKCellTypes.VTK_TETRA)
    ds = unstructured_grid.create_dataset(
        points=wp.array(mesh.points.astype(np.float32), dtype=wp.vec3f, device=dev),
        cell_types=wp.full(m, kind, dtype=wp.uint8, device=dev),
        cell_offsets=wp.array(np.arange(0, m * k + 1, k, dtype=np.int32), dtype=wp.int32,
                              device=dev),
        cell_connectivity=wp.array(mesh.cells.ravel().astype(np.int32), dtype=wp.int32,
                                   device=dev))
    ds.add_field("p", Field.from_array(wp.array(mesh.p, dtype=wp.float32, device=dev),
                                       AssociationType.NODE))
    ds.add_field("c", Field.from_array(wp.array(mesh.c, dtype=wp.float32, device=dev),
                                       AssociationType.ELEMENT))
    return ds


def _get(out, name):
    return out.get_field(name).get_data().numpy()


def operators(wp, dev, mesh, query):
    """(name, warp-simdata's call, Tack's call, kind of output, Tack's fields)."""
    from warp_simdata.data_models.custom import point_cloud
    from warp_simdata.operators import (
        centroid,
        element_faces,
        element_field,
        iso_surface,
        node_field,
        probe,
    )
    from warp_simdata.operators import slice as wslice

    import tack
    import tack.dataset as td

    origin, normal = tuple(pairs._center(mesh)), tuple(pairs.NORMAL)
    wq = point_cloud.create_dataset(wp.array(query, dtype=wp.vec3f, device=dev))
    tq = tack.empty((len(query),), tack.vec(3, tack.f32))
    tack.copyto(tq, query)

    def host(values):
        return np.asarray(tack.arrays.to_host(values))

    return [
        ("contour", lambda d: iso_surface.compute(d, "p", pairs.ISOVALUE),
         lambda d: td.contour(d, "p", pairs.ISOVALUE), "surface", ("p",)),
        ("slice", lambda d: wslice.compute(d, origin, normal),
         lambda d: td.slice_plane(d, origin, normal), "surface", ("p",)),
        ("external faces", lambda d: element_faces.compute(d, external_only=True),
         td.extract_surface, "surface", ()),
        ("cell centers", lambda d: _get(centroid.compute(d), "element_centers"),
         lambda d: host(td.cell_centers(d).values), "values", ()),
        ("point to cell",
         lambda d: _get(element_field.compute(d, "p", output_field_name="pc"), "pc"),
         lambda d: host(td.project(d, "p", td.Constant(d)).values), "values", ("p",)),
        ("cell to point",
         lambda d: _get(node_field.compute(d, "c", output_field_name="cp"), "cp"),
         lambda d: host(td.project(d, "c", td.H1(d)).values), "values", ("c",)),
        ("probe", lambda d: probe.compute(d, "p", wq, output_mask_field_name="valid"),
         lambda d: td.probe(d, tq, fields=["p"]), "probe", ("p",)),
    ]


def _area(points, idx, counts):
    total, start = 0.0, np.concatenate([[0], np.cumsum(counts)[:-1]])
    for n in np.unique(counts):
        first = start[counts == n]
        for j in range(1, n - 1):
            a, b, c = points[idx[first]], points[idx[first + j]], points[idx[first + j + 1]]
            total += 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    return total


def check(kind, w, t, query):
    """(agree, what was compared)."""
    import tack

    if kind == "surface":
        h = w.handle
        wp_, wi, wc = h.points.numpy(), h.face_vertex_indices.numpy(), h.face_vertex_counts.numpy()
        off = np.asarray(tack.arrays.to_host(t.topology.offsets)).astype(np.int64)
        tp = t.positions()
        ti = np.asarray(tack.arrays.to_host(t.topology.connectivity)).astype(np.int64)
        tc = np.diff(off)
        wa, ta = _area(wp_, wi, wc), _area(tp, ti, tc)
        used = len(np.unique(ti))
        ok = abs(wa - ta) <= 1e-4 * max(wa, 1e-9) and len(wc) == len(tc)
        return ok, (f"cells {len(wc)} / {len(tc)}, used points {len(np.unique(wi))} / {used}, "
                    f"area {wa:.6f} / {ta:.6f}")
    if kind == "values":
        w = np.asarray(w, np.float64).reshape(len(w), -1)
        t = np.asarray(t, np.float64).reshape(len(t), -1)
        err = np.abs(w - t).max() if w.shape == t.shape else np.inf
        return bool(err <= 1e-4), f"n {len(w)} / {len(t)}, max |diff| {err:.2e}"
    wv, wm = _get(w, "probed_values"), _get(w, "valid").astype(bool)
    tv = np.asarray(tack.arrays.to_host(t.fields["p"].values))
    tm = np.asarray(tack.arrays.to_host(t.fields["valid"].values)).astype(bool)
    both = wm & tm
    err = float(np.abs(wv[both] - tv[both]).max()) if both.any() else 0.0
    outside = np.maximum(np.maximum(-query, query - 1.0), 0).max(axis=1)
    extra = tm & ~wm
    ok = not (wm & ~tm).any() and bool((outside[extra] > 0).all()) and err <= 1e-4
    return ok, (f"found {wm.sum()} / {tm.sum()} (Tack's extra {extra.sum()} all outside, by at "
                f"most {outside[extra].max() if extra.any() else 0:.1e}), max |diff| {err:.2e}")


def timed(fn, budget, sync, min_runs=3, max_runs=50):
    times = []
    start = time.perf_counter()
    while len(times) < min_runs or (time.perf_counter() - start < budget
                                    and len(times) < max_runs):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--sizes", nargs="+", default=["10k", "100k", "1M"],
                    choices=sorted(meshes.SIZES))
    ap.add_argument("--kinds", nargs="+", default=["hex", "tet"], choices=("hex", "tet"))
    ap.add_argument("--budget", type=float, default=1.5, help="seconds per timing")
    ap.add_argument("--output", help="write the results as JSON here")
    a = ap.parse_args()
    wp, dev = setup(a.device)
    import tack

    def sync():
        wp.synchronize()
        tack.sync()

    rng = np.random.default_rng(7)
    query = rng.uniform(-0.05, 1.05, (PROBE_COUNT, 3)).astype(np.float32)
    rows = []
    for kind in a.kinds:
        for size in a.sizes:
            mesh = (meshes.hexahedra if kind == "hex" else meshes.tetrahedra)(meshes.SIZES[size])
            tm = meshes.TackMesh(mesh, "shape")
            wds = warp_dataset(wp, dev, mesh)
            for name, wfn, tfn, out_kind, fields in operators(wp, dev, mesh, query):
                tds = tm.dataset(fields)
                t0 = time.perf_counter()
                wout = wfn(wds)
                sync()
                w_first = time.perf_counter() - t0
                t0 = time.perf_counter()
                tout = tfn(tds)
                sync()
                t_first = time.perf_counter() - t0
                ok, detail = check(out_kind, wout, tout, query)
                row = {"kind": kind, "size": size, "cells": mesh.num_cells, "op": name,
                       "agree": ok, "detail": detail, "warp_first": w_first,
                       "tack_first": t_first,
                       "warp_warm": timed(lambda: wfn(wds), a.budget, sync),
                       "tack_warm": timed(lambda: tfn(tds), a.budget, sync),
                       "warp_fresh": timed(lambda: wfn(warp_dataset(wp, dev, mesh)), a.budget,
                                           sync, max_runs=10),
                       "tack_fresh": timed(lambda: tfn(tm.dataset(fields, fresh=True)),
                                           a.budget, sync, max_runs=10)}
                rows.append(row)
                print(f"{kind:3} {size:>4} {name:15} {'ok ' if ok else 'BAD'} "
                      f"warm {row['tack_warm'] * 1e3:8.2f} / {row['warp_warm'] * 1e3:8.2f} ms "
                      f"({row['tack_warm'] / row['warp_warm']:5.2f}x)  "
                      f"fresh {row['tack_fresh'] * 1e3:8.2f} / {row['warp_fresh'] * 1e3:8.2f} ms "
                      f"({row['tack_fresh'] / row['warp_fresh']:5.2f}x)  | {detail}", flush=True)
    print(f"{sum(r['agree'] for r in rows)}/{len(rows)} agree")
    if a.output:
        with open(a.output, "w") as f:
            json.dump({"device": a.device, "warp": wp.__version__, "cpus": os.cpu_count(),
                       "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()

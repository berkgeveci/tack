# Copyright (c) 2026 Kitware, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Five filters written with public names only, each checked against the built-in."""

import argparse

import numpy as np

import tack
import tack.dataset as td
from tack import communication as tc
from tack.algorithms import exclusive_scan, histogram
from tack.arrays import ConstantArray, CountingArray, materialize, size_of
from tack.dataset import units


# --8<-- [start:field]
@tack.kernel
def _elevations(positions, lx, ly, lz, dx, dy, dz, length2, low, span, out):
    for i in range(out.shape[0]):
        p = positions[i]
        s = ((p[0] - lx) * dx + (p[1] - ly) * dy + (p[2] - lz) * dz) / length2
        out[i] = low + min(max(s, 0.0), 1.0) * span


def elevation(data, low_point, high_point, range):
    """Each point's position along the line from low_point to high_point,
    scaled to range: td.point_elevation."""
    # The two points are world positions; the kernel sees positions relative
    # to the dataset's origin. Convert on the host, in f64.
    low = np.asarray(low_point, float) - np.asarray(data.origin)
    direction = np.asarray(high_point, float) - np.asarray(data.origin) - low
    n = size_of(data.geometry.values)               # an implicit array has no .shape
    out = tack.empty((n,), data.dtype, device=data.device)
    _elevations(materialize(data.geometry.values), *map(float, low), *map(float, direction),
                float(direction @ direction), float(range[0]), float(range[1] - range[0]),
                out)
    return td.Field(data.geometry.space, out)
# --8<-- [end:field]


# --8<-- [start:subset]
@tack.kernel
def _all_corners_inside(cells, u, lower, upper, keep):
    for c in cells:
        inside = 0
        for j in range(cells.NUM_POINTS):
            v = u.corner_value(c, j)
            if lower <= v and v <= upper:
                inside += 1
        keep[cells.entity_id(c)] = 1 if inside == cells.NUM_POINTS else 0


def threshold(data, name, lower, upper):
    """The cells whose corners all lie in [lower, upper]: td.threshold."""
    field = data.fields[name]
    lower = units.magnitude(lower, field.unit, "threshold's bounds")   # "300 K" or 300.0
    upper = units.magnitude(upper, field.unit, "threshold's bounds")
    keep = tack.empty((data.num_cells,), tack.i32, device=data.device)
    td.for_each(_all_corners_inside, data, "cells", field, lower, upper, keep)
    return td.extract_cells(data, td.Set.from_flags(data, "cells", keep))
# --8<-- [end:subset]


# --8<-- [start:geometry]
@tack.kernel
def _shrink_corners(positions, offsets, factor, n_cells):
    for c in range(n_cells):
        first = offsets[c]
        end = offsets[c + 1]
        center = positions[first]
        for k in range(first + 1, end):
            center += positions[k]
        center = center / (end - first)
        for k in range(first, end):
            positions[k] = center + factor * (positions[k] - center)


def shrink(data, factor):
    """Each cell shrunk toward its corners' mean: td.shrink."""
    l2 = td.L2(data)                                # each cell its own copy of its corners
    corners = td.project(data, data.geometry, l2)
    if data.num_cells:
        _shrink_corners(corners.values, corners.space.offsets, float(factor), data.num_cells)
    fields = {}
    for name, f in data.fields.items():
        if isinstance(f.space, td.Values) and f.space.on == "points":
            f = td.Field(td.H1(data), f.values, f.unit)
        fields[name] = td.project(data, f, l2) if isinstance(f.space, td.H1) else f
    # The same cells and points as the input: keep its marks, ids, origin,
    # unit and time.
    return td.keep_marks(data, td.DataSet(data.topology, corners, fields=fields,
                                          sets=dict(data.sets)))
# --8<-- [end:geometry]


# --8<-- [start:faces]
@tack.kernel
def _cell_jump(faces, values, out):
    for f in faces:
        inside = values[faces.side_cell(f, 0)]
        out[faces.entity_id(f)] = (values[faces.side_cell(f, 1)] - inside
                                   if faces.num_sides(f) == 2 else inside * 0.0)


def jump(data, name):
    """A cell field's difference across each face: td.jump."""
    field = data.fields[name]
    values = materialize(field.values)
    out = tack.empty((data.topology.faces().num_faces,), values.dtype, device=data.device)
    td.for_each(_cell_jump, data, "faces", values, out)
    # A difference of degrees Celsius is in kelvin; unknown stays unknown.
    return td.Field(td.Values(data, "faces"), out, units.delta(field.unit))
# --8<-- [end:faces]


# --8<-- [start:cells]
_TRIANGLES = {int(td.shapes.TRIANGLE): [0, 1, 2], int(td.shapes.QUAD): [0, 1, 2, 0, 2, 3],
              int(td.shapes.PIXEL): [0, 1, 3, 0, 3, 2]}


@tack.kernel
def _piece_counts(cells, pieces, counts):
    for c in cells:
        counts[cells.entity_id(c)] = pieces


@tack.kernel
def _split(cells, table, pieces, starts, connectivity, sources):
    for c in cells:
        e = cells.entity_id(c)
        at = starts[e]
        for t in range(pieces):
            for v in range(3):
                connectivity[(at + t) * 3 + v] = cells.point_id(c, table[t * 3 + v])
            sources[at + t] = e


def triangulate(data):
    """Each polygon of a surface as triangles: td.triangulate."""
    data = td.visible(data)                         # hidden cells are never processed
    n, idt, device = data.num_cells, data.id_dtype, data.device
    groups = [g for g in data.topology.groups() if g.count and int(g.shape.ID) in _TRIANGLES]
    # Count each cell's triangles, scan the counts into starts, fill.
    counts = tack.zeros((n,), tack.i32, device=device)
    for g in groups:
        _piece_counts(g.view(), len(_TRIANGLES[int(g.shape.ID)]) // 3, counts)
    starts = tack.empty((n,), idt, device=device)
    count = exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((3 * count,), idt, device=device)
    sources = tack.empty((count,), idt, device=device)
    for g in groups:
        table = _TRIANGLES[int(g.shape.ID)]
        _split(g.view(), tack.array(np.array(table, np.int32), device=device),
               len(table) // 3, starts, connectivity, sources)
    out = td.UnstructuredTopology(ConstantArray(int(td.shapes.TRIANGLE), count, dtype=tack.u8,
                                                device=device),
                                  CountingArray(count + 1, start=0, step=3, dtype=idt),
                                  connectivity, num_points=data.num_points)
    # The input's points; each triangle a piece of the cell `sources` names.
    return td.carry(data, out, points=td.Same(), cells=td.Pieces(sources))
# --8<-- [end:cells]


# --8<-- [start:pieces]
def entropy(pieces, name, bins=10):
    """The entropy of a point field's histogram over every piece: td.entropy."""
    blocks = tc.Blocks([p.device for p in pieces], pieces.world)
    # Owned values only: a point a ghost layer copies is counted once.
    values = {g: td.owned_values(p, name).numpy() for g, p in zip(blocks.local, pieces)}
    lo = blocks.all_reduce({g: v.min() for g, v in values.items()}, op="min")
    hi = blocks.all_reduce({g: v.max() for g, v in values.items()}, op="max")
    counts = {}
    for (g, v), piece in zip(values.items(), pieces):
        c, _ = histogram(tack.array(v, device=piece.device), bins=bins, range=(lo, hi))
        counts[g] = np.asarray(c.numpy() if hasattr(c, "numpy") else c, float)
    p = blocks.all_reduce(counts)                    # summed in piece order, on every rank
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())
# --8<-- [end:pieces]


def _host(values):
    return None if values is None else np.asarray(materialize(values).numpy())


def _same_arrays(a, b):
    a, b = _host(a), _host(b)
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def _same_field(a, b):
    return type(a.space) is type(b.space) and a.unit == b.unit \
        and _same_arrays(a.values, b.values)


def _same_dataset(a, b):
    """Every part: topology, geometry, fields, sets, marks, ids and frame."""
    ta, tb = a.topology, b.topology
    parts = [type(ta) is type(tb), a.num_points == b.num_points,
             _same_field(a.geometry, b.geometry), sorted(a.fields) == sorted(b.fields),
             sorted(a.sets) == sorted(b.sets),
             (a.origin, a.unit, a.time, a.time_unit) == (b.origin, b.unit, b.time, b.time_unit)]
    parts += [_same_arrays(getattr(ta, k), getattr(tb, k))
              for k in ("types", "offsets", "connectivity") if hasattr(ta, k)]
    parts += [_same_field(a.fields[k], b.fields[k]) for k in a.fields if k in b.fields]
    parts += [_same_arrays(a.sets[k].ids, b.sets[k].ids) for k in a.sets if k in b.sets]
    for part in ("cells", "points"):
        parts += [_same_arrays(x and getattr(x, part), y and getattr(y, part))
                  for x, y in ((a.ghosts, b.ghosts), (a.global_ids, b.global_ids))]
    return all(parts)


def run(arch="cpu", check=False):
    tack.set_default_device(arch)
    # --8<-- [start:data]
    w = td.wavelet(6)
    data = td.DataSet(w.topology, w.geometry, fields=dict(w.fields),
                      origin=(1.0e6, 2.0e6, 0.0), unit="m", time=3.0, time_unit="s")
    data.fields["cell_value"] = td.values_at_centers(data, "RTData")
    data.sets["some"] = td.Set(data, "cells", [0, 5, 9, 100])
    pieces = td.generate_global_ids(td.split(data, 4, ghost_layers=1))
    # --8<-- [end:data]
    low, high = (1.0e6, 2.0e6, -5.0), (1.0e6 + 3, 2.0e6, 5.0)
    results = {}
    for label, d in (("image", data), ("tetrahedra", td.tetrahedralize(data)),
                     ("ghosted piece", td.tetrahedralize(pieces[2]))):
        surface = td.extract_surface(d)
        results[label] = {
            "elevation": _same_field(elevation(d, low, high, (2.0, 4.0)),
                                     td.point_elevation(d, low, high, (2.0, 4.0))),
            "threshold": _same_dataset(threshold(d, "RTData", 150.0, 250.0),
                                       td.threshold(d, "RTData", 150.0, 250.0)),
            "shrink": _same_dataset(shrink(d, 0.6), td.shrink(d, 0.6)),
            "jump": _same_field(jump(d, "cell_value"), td.jump(d, "cell_value")),
            "triangulate": _same_dataset(triangulate(surface), td.triangulate(surface)),
        }
    results["pieces"] = {"entropy": entropy(pieces, "RTData") == td.entropy(pieces, "RTData")}
    if check:
        failed = [f"{label}: {name}" for label, r in results.items()
                  for name, same in r.items() if not same]
        assert not failed, failed
        kelvin = td.DataSet(data.topology, data.geometry, fields={
            "T": td.Field(data.fields["RTData"].space, data.fields["RTData"].values, "K")})
        assert _same_dataset(threshold(kelvin, "T", "-123.15 degC", (250.0, "K")),
                             td.threshold(kelvin, "T", "-123.15 degC", (250.0, "K")))
    agree = sum(sum(r.values()) for r in results.values())
    total = sum(len(r) for r in results.values())
    print(f"Filters: {agree} of {total} rewrites give the built-in's result, every part")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="cpu", choices=["cpu", "metal", "cuda", "hip", "level_zero"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(args.arch, args.check)

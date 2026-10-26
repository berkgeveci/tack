"""Built-in filters rewritten with public names only, held to the built-ins.

Tack's filters are meant to be extended with the machinery they are written
with: views and ``for_each``, ``carry`` and its maps, ghost marks, units,
``tack.communication``. Each rewrite here imports nothing private (the first
test checks) and must give the built-in's result, every part of it --
topology, geometry, fields and their units, sets, ghost marks, global ids,
origin, unit, time -- on an image with a far origin, a unit and a time, its
tetrahedra, and pieces with ghost marks and global ids. A built-in that comes
to need something users cannot reach fails here.

The vault's "Extensibility audit - findings" has the audit these come from;
three of its gaps were silent errors, each with a test below: a dataset built
by hand lost the record of lost ghost marks (``keep_marks``), a difference of
degrees Celsius kept the unit (``units.delta``), and a hidden cell was
processed (``visible``).
"""

import ast
import pathlib

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack import communication as tc
from tack.algorithms import histogram
from tack.arrays import ConstantArray, CountingArray, materialize, size_of
from tack.dataset import units

pytestmark = pytest.mark.quick      # the inner-loop set; see CLAUDE.md, Tests


def test_this_module_uses_public_names_only():
    tree = ast.parse(pathlib.Path(__file__).read_text())
    private = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(part.startswith("_") for part in node.module.split(".")):
                private.append(node.module)
            private += [a.name for a in node.names if a.name.startswith("_")]
        elif isinstance(node, ast.Import):
            private += [a.name for a in node.names
                        if any(p.startswith("_") for p in a.name.split("."))]
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                and not node.attr.startswith("__"):
            private.append(node.attr)
    assert private == []


# ── The datasets, and a comparison of every part ────────────────────

def _cases():
    w = td.wavelet(6)
    base = td.DataSet(w.topology, w.geometry, fields=dict(w.fields),
                      origin=(1.0e6, 2.0e6, 0.0), unit="m", time=3.0, time_unit="s")
    base.fields["source"] = td.cell_ids(base)
    base.fields["cell_value"] = td.values_at_centers(base, "RTData")
    base.sets["some"] = td.Set(base, "cells", [0, 5, 9, 100])
    pieces = td.generate_global_ids(td.split(base, 4, ghost_layers=1))
    return {"image": base, "tets": td.tetrahedralize(base), "piece": pieces[1],
            "ghosted tets": td.tetrahedralize(pieces[2])}


CASES = _cases()


@pytest.fixture(params=list(CASES))
def data(request):
    return CASES[request.param]


def _host(values):
    return None if values is None else np.asarray(materialize(values).numpy())


def _same_arrays(a, b):
    a, b = _host(a), _host(b)
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def _same_field(a, b):
    return (type(a.space) is type(b.space) and a.unit == b.unit
            and _same_arrays(a.values, b.values))


_TOPOLOGY = ("types", "offsets", "connectivity", "dims", "offset")


def _differences(a, b):
    out = []
    if type(a.topology) is not type(b.topology):
        return ["topology kind"]
    for name in _TOPOLOGY:
        if hasattr(a.topology, name):
            x, y = getattr(a.topology, name), getattr(b.topology, name)
            if isinstance(x, (tuple, int)):
                if x != y:
                    out.append(f"topology.{name}")
            elif not _same_arrays(x, y):
                out.append(f"topology.{name}")
    if a.num_points != b.num_points or not _same_field(a.geometry, b.geometry):
        out.append("geometry")
    if sorted(a.fields) != sorted(b.fields):
        out.append(f"fields {sorted(a.fields)} != {sorted(b.fields)}")
    out += [f"field {n}" for n in set(a.fields) & set(b.fields)
            if not _same_field(a.fields[n], b.fields[n])]
    if sorted(a.sets) != sorted(b.sets):
        out.append("sets")
    out += [f"set {n}" for n in set(a.sets) & set(b.sets)
            if not _same_arrays(a.sets[n].ids, b.sets[n].ids)]
    for part in ("cells", "points"):
        for what, x, y in (("ghosts", a.ghosts, b.ghosts), ("global_ids", a.global_ids,
                                                             b.global_ids)):
            if not _same_arrays(x and getattr(x, part), y and getattr(y, part)):
                out.append(f"{what}.{part}")
    out += [attr for attr in ("origin", "unit", "time", "time_unit")
            if getattr(a, attr) != getattr(b, attr)]
    return out


# ── A field from the geometry: point_elevation ──────────────────────

@tack.kernel
def _elevations(positions, lx, ly, lz, dx, dy, dz, length2, low, span, out):
    for i in range(out.shape[0]):
        p = positions[i]
        s = ((p[0] - lx) * dx + (p[1] - ly) * dy + (p[2] - lz) * dz) / length2
        out[i] = low + min(max(s, 0.0), 1.0) * span


def elevation(data, low_point, high_point, range):
    low = np.asarray(low_point, float) - np.asarray(data.origin)       # world to relative
    direction = np.asarray(high_point, float) - np.asarray(data.origin) - low
    n = size_of(data.geometry.values)
    out = tack.empty((n,), data.dtype, device=data.device)
    _elevations(materialize(data.geometry.values), *map(float, low), *map(float, direction),
                float(direction @ direction), float(range[0]), float(range[1] - range[0]),
                out)
    return td.Field(data.geometry.space, out)


def test_point_elevation(data):
    args = {"low_point": (1.0e6, 2.0e6, -5.0), "high_point": (1.0e6 + 3, 2.0e6, 5.0),
            "range": (2.0, 4.0)}
    assert _same_field(elevation(data, **args), td.point_elevation(data, **args))


# ── A subset of cells: threshold ────────────────────────────────────

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
    field = data.fields[name]
    lower = units.magnitude(lower, field.unit, "threshold's bounds")
    upper = units.magnitude(upper, field.unit, "threshold's bounds")
    keep = tack.empty((data.num_cells,), tack.i32, device=data.device)
    td.for_each(_all_corners_inside, data, "cells", field, lower, upper, keep)
    return td.extract_cells(data, td.Set.from_flags(data, "cells", keep))


def test_threshold(data):
    assert _differences(threshold(data, "RTData", 150.0, 250.0),
                        td.threshold(data, "RTData", 150.0, 250.0)) == []


def test_threshold_takes_quantities_in_the_fields_unit():
    data = CASES["image"]
    kelvin = td.DataSet(data.topology, data.geometry, fields={
        "T": td.Field(data.fields["RTData"].space, data.fields["RTData"].values, "K")})
    bounds = ("-123.15 degC", (250.0, "K"))
    assert _differences(threshold(kelvin, "T", *bounds), td.threshold(kelvin, "T", *bounds)) \
        == []


# ── A new geometry: shrink ──────────────────────────────────────────

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
    l2 = td.L2(data)
    corners = td.project(data, data.geometry, l2)
    if data.num_cells:
        _shrink_corners(corners.values, corners.space.offsets, float(factor), data.num_cells)
    fields = {}
    for name, f in data.fields.items():
        if isinstance(f.space, td.Values) and f.space.on == "points":
            f = td.Field(td.H1(data), f.values, f.unit)
        fields[name] = td.project(data, f, l2) if isinstance(f.space, td.H1) else f
    # The same cells and points: the input's marks, ids, origin, unit and time.
    return td.keep_marks(data, td.DataSet(data.topology, corners, fields=fields,
                                          sets=dict(data.sets)))


def test_shrink(data):
    assert _differences(shrink(data, 0.6), td.shrink(data, 0.6)) == []


def test_a_dataset_built_by_hand_keeps_the_record_of_lost_marks():
    """Contours of pieces without ghost cells cannot place their points' marks;
    a reduction over them must refuse, after a user's filter too."""
    pieces = td.generate_global_ids(td.split(CASES["image"], 4))
    cut = [td.contour(p, "RTData", 180.0) for p in pieces]
    shrunk = td.PartitionedDataSet([shrink(c, 0.5) for c in cut])
    with pytest.raises(NotImplementedError, match="ghost marks"):
        td.statistics(shrunk, "RTData")


# ── Faces and cells: jump of cell data, net_flux ────────────────────

@tack.kernel
def _cell_jump(faces, values, out):
    for f in faces:
        inside = values[faces.side_cell(f, 0)]
        out[faces.entity_id(f)] = (values[faces.side_cell(f, 1)] - inside
                                   if faces.num_sides(f) == 2 else inside * 0.0)


@tack.kernel
def _outward_sums(cells, flux, out):
    for c in cells:
        total = flux[0] * 0.0
        for f in range(cells.num_faces(c)):
            sign = 1.0 if cells.face_side(c, f) == 0 else -1.0
            total += sign * flux[cells.face_id(c, f)]
        out[cells.entity_id(c)] = total


def jump(data, name):
    field = data.fields[name]
    values = materialize(field.values)
    out = tack.empty((data.topology.faces().num_faces,), values.dtype, device=data.device)
    td.for_each(_cell_jump, data, "faces", values, out)
    return td.Field(td.Values(data, "faces"), out, units.delta(field.unit))


def net_flux(data, flux):
    values = materialize(flux.values)
    out = tack.zeros((data.num_cells,), values.dtype, device=data.device)
    for group in data.launch_groups("cells", []):
        if group.count and group.shape.DIMENSION == data.topology.dimension:
            _outward_sums(data.domain_view("cells", group), values, out)
    return td.Field(td.Values(data, "cells"), out, flux.unit)


def test_jump_and_net_flux(data):
    assert _same_field(jump(data, "cell_value"), td.jump(data, "cell_value"))
    flux = td.jump(data, "cell_value")
    assert _same_field(net_flux(data, flux), td.net_flux(data, flux))


def test_a_difference_of_degrees_celsius_is_in_kelvin():
    data = CASES["tets"]
    data = td.DataSet(data.topology, data.geometry, fields={"T": td.Field(
        data.fields["cell_value"].space, data.fields["cell_value"].values, "degC")})
    assert jump(data, "T").unit == td.jump(data, "T").unit == "K"


# ── New cells: triangulate ──────────────────────────────────────────

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
    data = td.visible(data)                     # hidden cells are never processed
    n, idt, device = data.num_cells, data.id_dtype, data.device
    groups = [g for g in data.topology.groups() if g.count and int(g.shape.ID) in _TRIANGLES]
    counts = tack.zeros((n,), tack.i32, device=device)
    for g in groups:
        _piece_counts(g.view(), len(_TRIANGLES[int(g.shape.ID)]) // 3, counts)
    starts = tack.empty((n,), idt, device=device)
    count = tack.algorithms.exclusive_scan(counts, starts, n) if n else 0
    connectivity = tack.empty((3 * count,), idt, device=device)
    sources = tack.empty((count,), idt, device=device)
    for g in groups:
        table = _TRIANGLES[int(g.shape.ID)]
        _split(g.view(), tack.array(np.array(table, np.int32), device=device),
               len(table) // 3, starts, connectivity, sources)
    out = td.UnstructuredTopology(ConstantArray(int(td.shapes.TRIANGLE), count, dtype=tack.u8),
                                  CountingArray(count + 1, start=0, step=3, dtype=idt),
                                  connectivity, num_points=data.num_points)
    return td.carry(data, out, points=td.Same(), cells=td.Pieces(sources))


def test_triangulate(data):
    surface = td.extract_surface(data)
    assert _differences(triangulate(surface), td.triangulate(surface)) == []


def test_hidden_cells_are_not_triangulated():
    surface = td.extract_surface(td.wavelet(3))
    marks = np.zeros(surface.num_cells, np.uint8)
    marks[0] = td.Ghosts.HIDDENCELL
    surface.ghosts = td.Ghosts(cells=tack.array(marks))
    assert _differences(triangulate(surface), td.triangulate(surface)) == []


# ── Over pieces: entropy ────────────────────────────────────────────

def entropy(pieces, name, bins=10):
    blocks = tc.Blocks([p.device for p in pieces], pieces.world)
    values = {g: td.owned_values(p, name).numpy() for g, p in zip(blocks.local, pieces)}
    lo = blocks.all_reduce({g: v.min() for g, v in values.items()}, op="min")
    hi = blocks.all_reduce({g: v.max() for g, v in values.items()}, op="max")
    counts = {}
    for (g, v), piece in zip(values.items(), pieces):
        c, _ = histogram(tack.array(v, device=piece.device), bins=bins, range=(lo, hi))
        counts[g] = np.asarray(c.numpy() if hasattr(c, "numpy") else c, float)
    p = blocks.all_reduce(counts)
    p = p[p > 0] / p.sum()
    return 0.0 - float((p * np.log2(p)).sum())


@pytest.mark.parametrize("tetrahedra", [False, True])
def test_entropy_over_pieces(tetrahedra):
    pieces = td.generate_global_ids(td.split(CASES["image"], 4, ghost_layers=1))
    if tetrahedra:
        pieces = td.PartitionedDataSet([td.tetrahedralize(p) for p in pieces])
    assert entropy(pieces, "RTData") == td.entropy(pieces, "RTData")

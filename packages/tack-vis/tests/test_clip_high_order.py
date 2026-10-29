"""``clip`` and ``threshold`` of curved cells and high-order fields: the cells the
field's Bernstein bound keeps entirely stay as they are, and the rest are cut on
linear sub-cells, as ``contour`` cuts them; the result is
``PartitionedDataSet([whole, part])``."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.topology import uniform_cells


@pytest.fixture(autouse=True)
def _f64(f64_backend):
    """Compared in f64, on every backend with it."""


@tack.func
def _square(x):
    return x[0] * x[0] + 0.5 * x[1] * x[2]


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _wedges(hexes):
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    topology = td.UnstructuredTopology(types, offsets,
                                       tack.array(np.array(conn, np.int32).reshape(-1)),
                                       num_points=hexes.num_points)
    return td.DataSet(topology, hexes.positions())


def _mesh(cells, curved=True):
    """Four cells a side over the unit cube (or a 2D square), the geometry bent at
    order 2 (or straight), ``u`` a quadratic at order 2 and ``c`` a cell field."""
    two = cells in ("quads", "triangles")
    grid = td.uniform_grid((5, 5) if two else (5, 5, 5), spacing=(0.25,) * (2 if two else 3),
                           dtype=tack.f64)
    hexes = td.extract_cells(grid, np.arange(grid.num_cells))
    data = {"hexahedra": lambda: hexes, "quads": lambda: hexes,
            "tetrahedra": lambda: td.tetrahedralize(hexes),
            "triangles": lambda: td.triangulate(hexes),
            "wedges": lambda: _wedges(hexes)}[cells]()
    data = td.DataSet(data.topology, data.geometry)
    if curved:
        data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                       dtype=tack.vec3(tack.f64))
    data.fields["u"] = td.interpolate(data, _square, td.H1(data, order=2), dtype=tack.f64)
    data.fields["c"] = td.Field(td.Constant(data),
                                tack.array(np.arange(data.num_cells, dtype=np.float64)))
    return data


def _cells(data):
    """Each cell as its type and the mean of its points: a set to compare by."""
    p = data.positions()
    t = data.topology
    types = tack.arrays.to_host(t.types)
    offsets = tack.arrays.to_host(t.offsets)
    conn = tack.arrays.to_host(t.connectivity)
    return {(int(types[i]), *np.round(p[conn[offsets[i]:offsets[i + 1]]].mean(0), 9))
            for i in range(len(types))}


def _split(out):
    assert isinstance(out, td.PartitionedDataSet)
    whole, part = out.partitions
    return whole, part


def _check_against_subdivided(whole, part, full, level=2):
    """``whole``'s cells subdivided and ``part``'s are ``full``'s, the filter of the
    whole dataset subdivided: only the cells the bound cannot decide are cut."""
    assert _cells(td.subdivide(whole, level)) | _cells(part) == _cells(full)
    assert td.subdivide(whole, level).num_cells + part.num_cells == full.num_cells


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges", "quads", "triangles"])
@pytest.mark.parametrize("inside", [False, True])
def test_clip_keeps_whole_cells_and_cuts_the_rest_on_sub_cells(cells, inside):
    """Whole cells keep their order-2 geometry and field, and their cell field;
    the others are the clip of their sub-cells, as clipping the whole subdivided
    dataset gives them."""
    data = _mesh(cells)
    whole, part = _split(td.clip(data, "u", 0.35, inside=inside, exact=False))
    assert whole.num_cells and part.num_cells
    assert whole.geometry.space.order == 2 and whole.fields["u"].space.order == 2
    assert part.geometry.space.order == 1
    values = whole.fields["u"].values.numpy()
    assert (values <= 0.35).all() if inside else (values >= 0.35).all()
    _check_against_subdivided(whole, part, td.clip(td.subdivide(data, 2), "u", 0.35,
                                                   inside=inside))
    # At the field's roots (the default): the same cells, the cut through the
    # contour's vertices, to the bit (the same sub-edges, the same roots).
    exact = _split(td.clip(data, "u", 0.35, inside=inside))[1]
    assert exact.num_cells == part.num_cells
    surface = td.contour(data, "u", 0.35)
    assert {tuple(x) for x in surface.positions()} <= {tuple(x) for x in exact.positions()}
    kept = set(whole.fields["c"].values.numpy()) | set(part.fields["c"].values.numpy())
    assert kept <= set(range(data.num_cells))


@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges"])
@pytest.mark.parametrize("all_points", [True, False])
def test_threshold_keeps_whole_cells_and_cuts_the_rest_on_sub_cells(cells, all_points):
    data = _mesh(cells)
    whole, part = _split(td.threshold(data, "u", 0.15, 0.55, all_points=all_points))
    assert whole.num_cells and part.num_cells
    assert whole.fields["u"].space.order == 2
    values = whole.fields["u"].values.numpy()
    assert ((0.15 <= values) & (values <= 0.55)).all()
    _check_against_subdivided(whole, part, td.threshold(td.subdivide(data, 2), "u", 0.15, 0.55,
                                                        all_points=all_points))


def test_an_order_two_field_on_straight_cells_and_finer_sub_cells():
    """A field above order 1 alone takes the path; ``subdivision`` cuts finer."""
    data = _mesh("hexahedra", curved=False)
    whole, part = _split(td.clip(data, "u", 0.35, subdivision=3, exact=False))
    assert whole.geometry.space.order == 1
    _check_against_subdivided(whole, part, td.clip(td.subdivide(data, 3), "u", 0.35), level=3)


def test_a_tolerance_chooses_the_sub_cells():
    data = _mesh("hexahedra")
    coarse = _split(td.clip(data, "u", 0.35))[1]
    fine = _split(td.clip(data, "u", 0.35, tolerance=1e-4))[1]
    assert fine.num_cells > coarse.num_cells
    with pytest.raises(ValueError, match="not both"):
        td.clip(data, "u", 0.35, subdivision=2, tolerance=1e-3)


def test_an_implicit_function_on_a_curved_geometry():
    """The function's values in the geometry's order-2 space decide the cells, as
    ``slice`` cuts them: whole cells hold it at or above zero at every node."""
    data = _mesh("hexahedra")
    sphere = td.implicit.Sphere((0.6, 0.6, 0.6), 0.45)
    whole, part = _split(td.clip(data, sphere))
    assert whole.num_cells and part.num_cells
    nodes = whole.geometry.values.numpy()
    assert (np.linalg.norm(nodes - 0.6, axis=1) >= 0.45 - 1e-12).all()
    assert "__cut__" not in part.fields
    assert sorted(part.fields) == sorted(whole.fields) == ["c", "u"]


def test_fields_are_chosen_and_linear_input_is_unchanged():
    data = _mesh("hexahedra")
    whole, part = _split(td.clip(data, "u", 0.35, fields=["c"]))
    assert list(whole.fields) == list(part.fields) == ["c"]
    straight = _mesh("hexahedra", curved=False)
    straight.fields["u"] = td.project(straight, straight.fields["u"], td.H1(straight))
    assert isinstance(td.clip(straight, "u", 0.35), td.DataSet)
    assert isinstance(td.threshold(straight, "u", 0.1, 0.5), td.DataSet)


def test_a_cell_field_thresholds_whole_curved_cells():
    """A cell field has one value a cell, which no sub-cell divides: a curved
    dataset thresholded by it is its kept cells, as before."""
    data = _mesh("hexahedra")
    out = td.threshold(data, "c", 10, 40)
    assert isinstance(out, td.DataSet) and out.num_cells == 31
    assert out.geometry.space.order == 2


def _owned_positions(parts):
    """The positions of every partition's points not marked a copy, and how many."""
    owned = []
    for part in parts:
        p = part.positions()
        marks = part.ghosts.points if part.ghosts is not None else None
        if marks is not None:
            p = p[(marks.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        owned.append(p)
    owned = np.concatenate(owned)
    return {tuple(x) for x in owned}, len(owned)


@pytest.mark.parametrize("filter", ["clip", "threshold"])
@pytest.mark.parametrize("cells", ["hexahedra", "tetrahedra", "wedges"])
def test_the_seams_points_are_owned_once(filter, cells):
    """The whole cells' corners on the seam are in both parts; the whole part owns
    them and the cut part's copies are ``DUPLICATEPOINT``, so ``owned_values``
    over the two counts every point once."""
    data = _mesh(cells)
    run = {"clip": lambda d: td.clip(d, "u", 0.35),
           "threshold": lambda d: td.threshold(d, "u", 0.15, 0.55)}[filter]
    whole, part = _split(run(data))
    shared = {tuple(x) for x in whole.positions()} & {tuple(x) for x in part.positions()}
    assert shared
    positions, count = _owned_positions([whole, part])
    assert count == len(positions)
    assert positions == {tuple(x) for x in np.concatenate([whole.positions(),
                                                           part.positions()])}
    assert part.ghosts.cells is None
    assert td.owned_values(part, part.geometry).shape[0] == part.num_points - len(shared)


def test_the_seams_points_are_owned_once_over_pieces():
    """With ghost layers, each piece's parts own their share: over every piece and
    part, the owned points are the whole dataset's, each once."""
    def grid():
        return td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)

    def prepared(d):
        d.geometry = td.interpolate(d, _bend, td.H1(d, order=2), dtype=tack.vec3(tack.f64))
        d.fields["u"] = td.interpolate(d, _square, td.H1(d, order=2), dtype=tack.f64)
        return d

    expected, count = _owned_positions(_split(td.clip(prepared(grid()), "u", 0.35)))
    assert count == len(expected)
    parts = []
    for piece in td.split(grid(), (2, 2, 1), ghost_layers=1):
        parts += list(td.clip(prepared(piece), "u", 0.35).partitions)
    positions, count = _owned_positions(parts)
    assert count == len(positions)
    assert positions == expected

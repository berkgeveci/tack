"""Serendipity (``H1(order=2, basis="serendipity")``): the full order-2 space less its
values at quad faces' centres and cells' insides, which its functions determine from
the corners and edges -- VTK's quad8, hex20 and wedge15, Exodus's and Abaqus's
quadratic cells. A serendipity field holds every quadratic of position on straight
cells, so it is held to the full space there, and to VTK's own quadratic cells."""

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
def _quadratic(x):
    return x[0] * x[0] - 2.0 * x[1] * x[2] + 0.5 * x[1] * x[1] + 0.3 * x[2] + x[0] * x[1]


@tack.func
def _bend(x):
    return [x[0] + 0.1 * x[1] * x[1], x[1] + 0.12 * x[0] * x[0], x[2] + 0.05 * x[0] * x[1]]


def _quadratic_at(p):
    return (p[:, 0] ** 2 - 2 * p[:, 1] * p[:, 2] + 0.5 * p[:, 1] ** 2 + 0.3 * p[:, 2]
            + p[:, 0] * p[:, 1])


_DIMENSION = {"quads": 2, "triangles": 2, "hexahedra": 3, "tetrahedra": 3, "wedges": 3}


def _wedges(hexes):
    conn = []
    for h in hexes.topology.connectivity.numpy().reshape(-1, 8):
        conn += [[h[0], h[1], h[3], h[4], h[5], h[7]], [h[1], h[2], h[3], h[5], h[6], h[7]]]
    types, offsets = uniform_cells(int(shapes.WEDGE), len(conn), tack.i32)
    return td.DataSet(td.UnstructuredTopology(
        types, offsets, tack.array(np.array(conn, np.int32).reshape(-1)),
        num_points=hexes.num_points), hexes.geometry.values)


def _mesh(cells):
    d = _DIMENSION[cells]
    grid = td.uniform_grid((4,) * d, spacing=(1 / 3,) * d, dtype=tack.f64)
    tensor = td.extract_cells(grid, np.arange(grid.num_cells))
    data = {"tetrahedra": lambda: td.tetrahedralize(tensor),
            "triangles": lambda: td.triangulate(tensor),
            "wedges": lambda: _wedges(tensor)}.get(cells, lambda: tensor)()
    return td.DataSet(data.topology, data.geometry)


def _points(cells, n=120):
    p = np.random.default_rng(4).uniform(0.02, 0.98, (n, 3))
    p[:, _DIMENSION[cells]:] = 0.0
    return p


@pytest.mark.parametrize("cells", list(_DIMENSION))
def test_serendipity_holds_the_quadratics(cells):
    """Corners and edges alone (no face centres or insides on quads, hexahedra and
    wedges; the full space on simplices); every quadratic of position reproduced,
    with its gradients; the full space's twin through project and carry."""
    data = _mesh(cells)
    space = td.H1(data, order=2, basis="serendipity")
    full = td.H1(data, order=2)
    edges = data.topology.edges().num_edges
    assert space.size == data.num_points + edges
    assert space.size < full.size if cells in ("quads", "hexahedra", "wedges") \
        else space.size == full.size
    data.fields["s"] = td.interpolate(data, _quadratic, space)
    data.fields["f"] = td.interpolate(data, _quadratic, full)
    points = _points(cells)
    probed = td.probe(data, points)
    np.testing.assert_allclose(probed.fields["s"].values.numpy(), _quadratic_at(points),
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(td.gradients(data, "s").values.numpy(),
                               td.gradients(data, "f").values.numpy(), rtol=0, atol=1e-11)
    np.testing.assert_allclose(td.project(data, "f", space).values.numpy(),
                               data.fields["s"].values.numpy(), rtol=0, atol=1e-13)
    kept = td.extract_cells(data, np.arange(0, data.num_cells, 2))
    assert kept.fields["s"].space is td.H1(kept, order=2, basis="serendipity")
    centers = td.cell_centers(kept).values.numpy()
    np.testing.assert_allclose(td.probe(kept, centers).fields["s"].values.numpy(),
                               _quadratic_at(centers), rtol=0, atol=1e-13)


@pytest.mark.parametrize("cells", ["hexahedra", "wedges", "quads"])
def test_filters_see_the_quadratic(cells):
    """Exact contours of a serendipity field lie on its level, on a straight geometry
    and on one curved in the full order-2 space; clips and subdivisions agree with
    the full space's where both hold the same function (straight cells)."""
    for curved in (False, True):
        data = _mesh(cells)
        if curved:
            data.geometry = td.interpolate(data, _bend, td.H1(data, order=2),
                                           dtype=tack.vec3(tack.f64))
        data.fields["s"] = td.interpolate(data, _quadratic, td.H1(data, order=2,
                                                                  basis="serendipity"))
        surface = td.contour(data, "s", 0.3)
        assert surface.num_cells
        probed = td.probe(data, surface.positions()).fields["s"].values.numpy()
        assert np.abs(probed - 0.3).max() < (1e-9 if curved else 1e-13)      # Newton
        if not curved:
            data.fields["f"] = td.interpolate(data, _quadratic, td.H1(data, order=2))
            for a, b in ((td.clip(data, "f", 0.3).partitions[1],
                          td.clip(data, "s", 0.3).partitions[1]),
                         (td.subdivide(data, 2), td.subdivide(data, 2))):
                assert a.num_cells == b.num_cells
                np.testing.assert_allclose(np.sort(b.positions(), 0), np.sort(a.positions(), 0),
                                           rtol=0, atol=1e-13)


def test_pieces_read_shared_face_centres_alike():
    """A face's centre is summed from its corners and edges smallest first: pieces
    with a ghost layer own the whole's contour vertices, bit for bit."""
    def owned(surface):
        x = surface.positions()
        if surface.ghosts is not None and surface.ghosts.points is not None:
            x = x[(surface.ghosts.points.numpy() & td.Ghosts.SKIPPED_POINTS) == 0]
        return {tuple(p) for p in x}

    def prepared(d):
        d.geometry = td.interpolate(d, _bend, td.H1(d, order=2), dtype=tack.vec3(tack.f64))
        d.fields["s"] = td.interpolate(d, _quadratic, td.H1(d, order=2, basis="serendipity"))
        return d

    def grid():
        return td.uniform_grid((5, 5, 5), spacing=(0.25, 0.25, 0.25), dtype=tack.f64)

    whole = owned(td.contour(prepared(grid()), "s", 0.3))
    mine, count = set(), 0
    for piece in td.split(grid(), (2, 2, 1), ghost_layers=1):
        got = owned(td.contour(prepared(piece), "s", 0.3))
        count += len(got)
        mine |= got
    assert count == len(mine) and mine == whole


def test_a_curved_serendipity_geometry():
    """A geometry in serendipity is the full order-2 space's that holds the same map:
    cell centres and probes agree with that projection's."""
    data = _mesh("hexahedra")
    data.geometry = td.interpolate(data, _bend, td.H1(data, order=2, basis="serendipity"),
                                   dtype=tack.vec3(tack.f64))
    data.fields["s"] = td.interpolate(data, _quadratic, td.H1(data, order=2,
                                                              basis="serendipity"))
    twin = td.DataSet(data.topology, td.project(data, data.geometry, td.H1(data, order=2)))
    twin.fields["s"] = data.fields["s"]
    np.testing.assert_allclose(td.cell_centers(data).values.numpy(),
                               td.cell_centers(twin).values.numpy(), rtol=0, atol=1e-13)
    points = td.cell_centers(data).values.numpy() + 0.03
    np.testing.assert_allclose(td.probe(data, points).fields["s"].values.numpy(),
                               td.probe(twin, points).fields["s"].values.numpy(), rtol=0,
                               atol=1e-10)


def test_refusals():
    data = _mesh("hexahedra")
    with pytest.raises(NotImplementedError, match="of order 2"):
        td.H1(data, order=3, basis="serendipity")
    with pytest.raises(ValueError, match="an H1 basis"):
        td.L2(data, order=2, basis="serendipity")
    lines = td.extract_cells(td.uniform_grid((4,), spacing=(0.3,), dtype=tack.f64),
                             np.arange(3))
    with pytest.raises(NotImplementedError, match="serendipity is on quads"):
        td.interpolate(lines, _quadratic, td.H1(lines, order=2, basis="serendipity"))


# ── VTK's quadratic cells ──────────────────────────────────────────

def _vtk_at(grid, name, pcs):
    vtk = pytest.importorskip("vtkmodules.vtkCommonDataModel")
    from vtkmodules.vtkCommonCore import reference

    cell = vtk.vtkGenericCell()
    xs, us = [], []
    array = grid.point_data[name]
    for c in range(grid.number_of_cells):
        grid.GetCell(c, cell)
        for pc in pcs:
            weights = [0.0] * cell.GetNumberOfPoints()
            x = [0.0, 0.0, 0.0]
            cell.EvaluateLocation(reference(0), list(pc), x, weights)
            xs.append(x)
            us.append(sum(w * array.GetValue(cell.GetPointId(k)) for k, w in enumerate(weights)))
    return np.array(xs), np.array(us)


@pytest.mark.parametrize("vtk_type, basis", [(23, "serendipity"), (25, "serendipity"),
                                             (26, "serendipity"), (22, None), (24, None)])
def test_vtks_quadratic_cells(vtk_type, basis):
    """VTK's own quadratic cells (vtkCellTypeSource), bent: read into serendipity (quad8,
    hex20, wedge15) or the full order-2 space (triangles, tetrahedra), fields and
    geometry; Tack evaluates them as VTK does; written back, VTK evaluates them
    unchanged."""
    pytest.importorskip("vtkmodules.vtkFiltersSources")
    from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset

    solid = vtk_type in (24, 25, 26)
    grid = vtkCellTypeSource(cell_type=vtk_type,
                             blocks_dimensions=(3, 3, 2) if solid else (3, 3, 1))()
    p = np.asarray(grid.points.data, np.float64)
    p = p + 0.04 * np.c_[np.sin(p[:, 1]), np.sin(p[:, 2]), np.sin(p[:, 0])]
    grid.points = vtkPoints(data=p)
    grid.point_data["u"] = vtkDoubleArray(p[:, 0] ** 2 - p[:, 1] * p[:, 2] + 0.3 * p[:, 2])
    pcs = np.array([[0.2, 0.3, 0.25], [0.15, 0.1, 0.6]])
    if not solid:
        pcs[:, 2] = 0.0
    xs, us = _vtk_at(grid, "u", pcs)
    data = vtk_to_dataset(grid, dtype=tack.f64)
    space = td.H1(data, order=2, **({"basis": basis} if basis else {}))
    assert data.fields["u"].space is space and data.geometry.space is space
    probed = td.probe(data, xs)
    assert probed.fields["valid"].values.numpy().all()
    np.testing.assert_allclose(probed.fields["u"].values.numpy(), us, rtol=0, atol=1e-9)
    back = dataset_to_vtk(data, higher_order="quadratic")
    assert back.GetCellType(0) == vtk_type
    xb, ub = _vtk_at(back, "u", pcs)
    np.testing.assert_allclose(xb, xs, rtol=0, atol=1e-13)
    np.testing.assert_allclose(ub, us, rtol=0, atol=1e-13)

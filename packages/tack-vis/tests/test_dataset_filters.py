"""Tests for tack.dataset.filters: contour, slice, threshold, external faces and
edges.

Ported from vis/data-model's filter tests, with the same references: VTK's
vtkContourGrid for every case of every 3D shape and for real meshes,
vtkCutter, vtkThreshold and vtkGeometryFilter; and, without VTK, closed
surfaces on the isovalue, agreement between grid forms, and counts. New
here is what the spaces bring: a DG field contours with its jumps, an L2
geometry slices its own cells, and threshold carries L2 fields (of any
orders per cell) block by block.
"""

import types

import numpy as np
import pytest
from test_shapes import EXPECTED

import tack
import tack.dataset as td
from tack.dataset import algorithms as alg
from tack.dataset import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter

    from tack.interop.vtk import dataset_to_vtk, vtk_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(**{name: getattr(module, name)
                                   for module in (vtkCommonCore, vtkCommonDataModel,
                                                  vtkFiltersCore)
                                   for name in dir(module) if name.startswith("vtk")})
    vtk.vtkGeometryFilter = vtkGeometryFilter

needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

SOLIDS = [sh.Tetra, sh.Voxel, sh.Hexahedron, sh.Wedge, sh.Pyramid]


def _scalars(values, dtype=tack.f32):
    values = np.asarray(values)
    f = tack.empty((len(values),), dtype)
    if len(values):
        tack.copyto(f, values.astype(dtype.numpy_dtype))
    return f


def _grid(n, form, dtype=tack.f32):
    """A grid of n points a side over [-1, 1]^3: rectilinear, or the same cells
    as an explicit hexahedral mesh."""
    axis = np.linspace(-1, 1, n)
    if form == "rectilinear":
        return td.rectilinear_grid(axis, axis, axis, dtype=dtype)
    points = np.array([(x, y, z) for z in axis for y in axis for x in axis])
    rows = []
    for k in range(n - 1):
        for j in range(n - 1):
            for i in range(n - 1):
                b = i + j * n + k * n * n
                rows.append([b, b + 1, b + 1 + n, b + n,
                             b + n * n, b + 1 + n * n, b + 1 + n + n * n, b + n + n * n])
    topology = td.UnstructuredTopology(np.full(len(rows), 12, np.uint8),
                                       8 * np.arange(len(rows) + 1), np.concatenate(rows))
    return td.DataSet(topology, points, dtype=dtype)


def _with_radius(data, dtype=tack.f32):
    data.fields["r"] = td.Field(td.H1(data), _scalars(np.linalg.norm(data.positions(), axis=1),
                                                      dtype))
    return data


def _triangles(surface):
    points = surface.positions()
    return points, surface.topology.connectivity.numpy().reshape(-1, 3)


def _edge_uses(triangles):
    e = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
                                triangles[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=True)[1]


def _oriented(points, triangles, decimals=5):
    """Triangles by coordinates, each rotated to start at its smallest point."""
    out = set()
    for t in triangles:
        c = [tuple(np.round(points[p], decimals) + 0.0) for p in t]
        s = c.index(min(c))
        out.add(tuple(c[s:] + c[:s]))
    return out


# ── Contour ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("form", ["rectilinear", "explicit"])
def test_a_sphere_is_closed_and_on_the_isovalue(backend, form):
    data = _with_radius(_grid(9, form))
    surface = td.contour(data, "r", 0.6)
    points, triangles = _triangles(surface)
    assert len(triangles) > 100
    assert (_edge_uses(triangles) == 2).all()                  # watertight
    np.testing.assert_allclose(surface.fields["r"].values.numpy(), 0.6, atol=1e-5)
    # Linear interpolation puts the points inside the sphere, by little.
    radii = np.linalg.norm(points, axis=1)
    assert (radii <= 0.6 + 1e-5).all() and (radii > 0.55).all()


def test_grid_forms_and_merging_agree(backend):
    rect = td.contour(_with_radius(_grid(7, "rectilinear")), "r", 0.7)
    expl = td.contour(_with_radius(_grid(7, "explicit")), "r", 0.7)
    loose = td.contour(_with_radius(_grid(7, "explicit")), "r", 0.7, merge_points=False)
    a, b, c = (_oriented(*_triangles(s)) for s in (rect, expl, loose))
    assert a == b == c
    assert loose.num_points == 3 * loose.num_cells > expl.num_points


def test_a_dg_field_keeps_its_jumps(backend):
    data = _with_radius(_grid(7, "explicit"))
    continuous = td.contour(data, "r", 0.7)
    dg = alg.project(data, data.fields["r"], td.L2(data))
    same = td.contour(data, dg, 0.7)
    # The same function: the same triangles, but points merged only within cells,
    # so the surface is no longer stitched across cell faces.
    assert _oriented(*_triangles(same)) == _oriented(*_triangles(continuous))
    assert same.num_points > continuous.num_points
    assert (_edge_uses(_triangles(same)[1]) == 1).any()
    # Each cell its own offset: every cell's piece moves, and pieces part.
    offsets = td.L2(data).offsets.numpy()
    values = dg.values.numpy()
    for cell in range(data.num_cells):
        values[offsets[cell]:offsets[cell + 1]] += 0.02 * (cell % 3 - 1)
    shifted = td.contour(data, td.Field(dg.space, _scalars(values)), 0.7)
    assert _oriented(*_triangles(shifted)) != _oriented(*_triangles(continuous))


def test_a_variable_order_field_contours_its_order_one_cells(backend):
    data = _with_radius(_grid(7, "explicit"))
    radius = data.fields["r"].values.numpy()
    orders = (np.arange(data.num_cells) % 2).astype(int)
    space = td.L2(data, order=orders)
    connectivity = data.topology.connectivity.numpy().reshape(-1, 8)
    offsets = space.offsets.numpy()
    values = np.empty(space.size)
    for cell in range(data.num_cells):
        values[offsets[cell]:offsets[cell + 1]] = (radius[connectivity[cell]] if orders[cell]
                                                   else 0.0)
    surface = td.contour(data, td.Field(space, _scalars(values)), 0.7)
    # Order-0 cells are constant: no level set inside them. The rest are the
    # continuous contour's triangles from those cells.
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    full = td.contour(data, "r", 0.7, merge_points=False)
    from_order_one = orders[full.fields["cell"].values.numpy().astype(int)] == 1
    points, triangles = _triangles(full)
    assert _oriented(*_triangles(surface)) == _oriented(points, triangles[from_order_one])


def test_an_order_two_field_is_refused(backend):
    """Its corner values alone are a linear field: a quadratic that rises inside
    every cell from zero at the corners would have no surface at all."""
    data = _grid(3, "explicit")
    space = td.H1(data, order=2)
    values = np.ones(space.size)
    values[:data.num_points] = 0.0
    with pytest.raises(NotImplementedError, match="order-2 H1"):
        td.contour(data, td.Field(space, _scalars(values)), 0.5)


def test_fields_carried_onto_the_surface(backend):
    data = _with_radius(_grid(6, "explicit"))
    positions = data.positions()
    data.fields["x"] = td.Field(td.H1(data), _scalars(positions[:, 0]))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    surface = td.contour(data, "r", 0.7)
    points = surface.positions()
    np.testing.assert_allclose(surface.fields["x"].values.numpy(), points[:, 0], atol=1e-5)
    assert surface.fields["cell"].space is td.Constant(surface)    # it keeps its kind
    # Each triangle lies in the cell it says it came from.
    _, triangles = _triangles(surface)
    cells = surface.fields["cell"].values.numpy().astype(int)
    corners = positions[data.topology.connectivity.numpy().reshape(-1, 8)]
    lo, hi = corners.min(axis=1)[cells], corners.max(axis=1)[cells]
    centroid = points[triangles].mean(axis=1)
    assert ((centroid >= lo - 1e-5) & (centroid <= hi + 1e-5)).all()


def test_contours_that_miss_and_bad_fields(backend):
    data = _with_radius(_grid(5, "explicit"))
    for isovalue in (5.0, -1.0):
        surface = td.contour(data, "r", isovalue)
        assert surface.num_cells == 0 and surface.num_points == 0
    with pytest.raises(TypeError, match="scalar field"):
        td.contour(data, data.geometry, 0.5)
    data.fields["id"] = td.Field(td.H1(data), tack.arange(data.num_points))
    with pytest.raises(TypeError, match="floating-point"):
        td.contour(data, "id", 1.0)
    with pytest.raises(TypeError, match="with a basis"):
        td.contour(data, td.Field(td.Values(data, "faces"),
                                  _scalars(np.zeros(data.topology.faces().num_faces))), 0.5)


def _levels_data(form):
    """A field ``r`` (the distance from the origin) on a mesh of each kind contour has a
    path for."""
    if form in ("2d", "2d-l2"):
        data = td.uniform_grid((21, 17), origin=(-1, -1, 0), spacing=(0.1, 0.125, 1.0))
    else:
        data = _grid(6, "explicit")
        if form == "polyhedra":
            data = td.as_polyhedra(data)
    data = _with_radius(data)
    if form in ("l2", "2d-l2"):
        data.fields["r"] = td.project(data, "r", td.L2(data))
    return data


@pytest.mark.parametrize("form", ["shapes", "l2", "polyhedra", "2d", "2d-l2"])
def test_several_values_are_every_values_contour_in_one_dataset(backend, form):
    """A sequence of values gives every value's contour in one dataset, made in one
    pass over the cells, each output cell's value in ``contour_value``: the
    cells of each value are that value's contour, cell for cell and in measure,
    and no point is shared between values."""
    data = _levels_data(form)
    levels = [0.45, 0.7, 1.05]
    many = td.contour(data, "r", levels)
    singles = [td.contour(data, "r", v) for v in levels]
    assert many.num_cells == sum(s.num_cells for s in singles) > 0
    assert many.num_points == sum(s.num_points for s in singles)
    values = many.fields["contour_value"].values.numpy()
    measure = alg.cell_geometry(many).measure.values.numpy()
    for v, single in zip(levels, singles):
        mine = np.flatnonzero(values == np.float32(v))
        assert len(mine) == single.num_cells
        assert measure[mine].sum() == pytest.approx(
            alg.cell_geometry(single).measure.values.numpy().sum(), rel=1e-5)


def test_one_value_or_a_sequence_and_slices_at_several_offsets(backend):
    data = _with_radius(_grid(5, "explicit"))
    assert "contour_value" not in td.contour(data, "r", 0.5).fields
    one = td.contour(data, "r", [0.5])
    assert one.num_cells == td.contour(data, "r", 0.5).num_cells
    np.testing.assert_array_equal(one.fields["contour_value"].values.numpy(), 0.5)
    with pytest.raises(ValueError, match="at least one value"):
        td.contour(data, "r", [])
    # Parallel planes: signed distances along the unit normal.
    normal = np.array([1.0, 2.0, 2.0]) / 3.0
    offsets = [-0.5, 0.0, 0.25]
    planes = td.slice_plane(data, (0, 0, 0), normal, value=offsets)
    assert planes.num_cells == sum(td.slice_plane(data, (0, 0, 0), normal, value=v).num_cells
                                   for v in offsets)
    level = planes.fields["contour_value"].values.numpy()
    rows = planes.topology.connectivity.numpy().reshape(-1, 3)
    distance = planes.positions() @ normal
    np.testing.assert_allclose(distance[rows[:, 0]], level, atol=1e-5)


def _vtk_contour(data, name, isovalue):
    grid = dataset_to_vtk(data)
    grid.GetPointData().SetActiveScalars(name)
    contour = vtk.vtkContourGrid()
    contour.SetValue(0, isovalue)
    out = contour(grid)
    if not out.number_of_cells:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    offsets = np.asarray(out.polys.offsets_array)
    connectivity = np.asarray(out.polys.connectivity_array)
    return (np.asarray(out.points.data),
            np.array([connectivity[offsets[i]:offsets[i + 1]]
                      for i in range(len(offsets) - 1)]))


def _every_case(shape, rng):
    """One cell per case of ``shape``, apart from each other, with values above and
    below 0.5."""
    ref = np.array(EXPECTED[shape]["points"], float)
    n = len(ref)
    points, values, rows = [], [], []
    for case in range(2 ** n):
        corner = np.array([case % 16, (case // 16) % 16, case // 256], float) * 3.0
        rows.append(list(range(len(points), len(points) + n)))
        points.extend(corner + ref * 2.0)
        values.extend(0.5 + (1 if (case >> j) & 1 else -1) * rng.uniform(0.1, 0.4)
                      for j in range(n))
    return np.array(points), np.array(values), np.array(rows)


def _explicit(points, shape_ids, rows, dtype=tack.f64):
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
    topology = td.UnstructuredTopology(np.asarray(shape_ids, np.uint8), offsets,
                                       np.concatenate(rows), num_points=len(points))
    return td.DataSet(topology, points, dtype=dtype)


@needs_vtk
@pytest.mark.parametrize("shape", SOLIDS, ids=lambda cls: cls.__name__)
def test_every_case_of_every_shape_is_vtks(f64_backend, shape):
    points, values, rows = _every_case(shape, np.random.default_rng(int(shape.ID)))
    data = _explicit(points, [shape.ID] * len(rows), rows)
    data.fields["v"] = td.Field(td.H1(data), _scalars(values, tack.f64))
    ours = _oriented(*_triangles(td.contour(data, "v", 0.5)))
    theirs = _oriented(*_vtk_contour(data, "v", 0.5))
    assert len(ours) == len(theirs) and ours == theirs


@needs_vtk
@pytest.mark.parametrize("mesh", ["delaunay", "mixed"])
def test_meshes_are_vtks(f64_backend, mesh):
    rng = np.random.default_rng(9)
    if mesh == "delaunay":
        cloud = vtk.vtkPolyData(points=vtk.vtkPoints(data=rng.uniform(-1, 1, (300, 3))))
        data = vtk_to_dataset(vtk.vtkDelaunay3D()(cloud), dtype=tack.f64)
    else:
        ids, rows, points = [], [], []
        for shape in rng.choice(SOLIDS, 200):
            ref = np.array(EXPECTED[shape]["points"], float)
            rows.append(list(range(len(points), len(points) + len(ref))))
            points.extend(rng.uniform(-1, 1, 3) + ref * rng.uniform(0.2, 0.5, 3))
            ids.append(shape.ID)
        data = _explicit(np.array(points), ids, rows)
    _with_radius(data, tack.f64)
    for isovalue in (0.4, 0.8):
        ours = _oriented(*_triangles(td.contour(data, "r", isovalue)))
        theirs = _oriented(*_vtk_contour(data, "r", isovalue))
        assert len(ours) == len(theirs) and ours == theirs


# ── Slice ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("form", ["rectilinear", "explicit"])
def test_an_axis_aligned_slice(backend, form):
    data = _grid(5, form)
    surface = td.slice_plane(data, (0.3, 0.0, 0.0), (2.0, 0.0, 0.0))
    points, triangles = _triangles(surface)
    np.testing.assert_allclose(points[:, 0], 0.3, atol=1e-6)
    a, b = points[triangles[:, 1]] - points[triangles[:, 0]], points[triangles[:, 2]] - points[triangles[:, 0]]
    np.testing.assert_allclose(0.5 * np.linalg.norm(np.cross(a, b), axis=1).sum(), 4.0,
                               rtol=1e-5)
    assert (_edge_uses(triangles) <= 2).all()


def test_an_oblique_slice_and_a_miss(backend):
    data = _grid(6, "explicit")
    origin, normal = np.array([0.1, -0.2, 0.3]), np.array([1.0, 0.4, 0.7])
    points, _ = _triangles(td.slice_plane(data, origin, normal))
    np.testing.assert_allclose((points - origin) @ normal, 0, atol=1e-5)
    assert td.slice_plane(data, (5.0, 0, 0), (1.0, 0, 0)).num_cells == 0


def test_a_slice_of_an_l2_geometry_is_its_cells_own(backend):
    """Each cell shrunk about its center, an L2 geometry: the slice cuts each cell
    where its own corners are, and the pieces no longer meet."""
    data = _grid(5, "explicit")
    centers = alg.cell_centers(data).values.numpy()
    connectivity = data.topology.connectivity.numpy().reshape(-1, 8)
    corners = (centers[:, None, :] + 0.8 * (data.positions()[connectivity]
                                            - centers[:, None, :])).reshape(-1, 3)
    geometry = td.Field(td.L2(data), _vectors(corners))
    shrunk = td.DataSet(data.topology, geometry)
    origin, normal = np.array([0.05, 0.0, 0.0]), np.array([1.0, 0.3, 0.2])
    cut = td.slice_plane(shrunk, origin, normal)
    points, triangles = _triangles(cut)
    np.testing.assert_allclose((points - origin) @ normal, 0, atol=1e-5)
    assert (_edge_uses(triangles) == 1).any()                  # pieces part
    whole = td.slice_plane(data, origin, normal)
    assert cut.num_cells > 0 and whole.num_cells > 0


def _vectors(values):
    values = np.asarray(values)
    f = tack.empty((len(values),), tack.vec(3, tack.f32))
    tack.copyto(f, values.astype(np.float32))
    return f


@needs_vtk
def test_slice_is_vtks(f64_backend):
    data = _with_radius(_grid(6, "explicit", tack.f64), tack.f64)
    origin, normal = [0.23, 0.11, -0.17], [1.0, 0.4, 0.7]
    plane = vtk.vtkPlane(origin=origin, normal=normal)
    out = vtk.vtkCutter(cut_function=plane)(dataset_to_vtk(data))
    offsets = np.asarray(out.polys.offsets_array)
    connectivity = np.asarray(out.polys.connectivity_array)
    theirs = _oriented(np.asarray(out.points.data),
                       np.array([connectivity[offsets[i]:offsets[i + 1]]
                                 for i in range(len(offsets) - 1)]))
    ours = _oriented(*_triangles(td.slice_plane(data, origin, normal)))
    assert len(ours) == len(theirs) and ours == theirs


# ── Threshold ───────────────────────────────────────────────────────

def _cells(data):
    """Each cell as (type, its points' coordinates in order), in cell order."""
    points = data.positions()
    t = data.topology
    offsets, connectivity = t.offsets.numpy(), t.connectivity.numpy()
    return [(int(k), tuple(tuple(np.round(points[p], 6)) for p in
                           connectivity[offsets[i]:offsets[i + 1]]))
            for i, k in enumerate(t.types.numpy())]


def _values_data(dtype=tack.f32):
    data = _grid(5, "explicit", dtype)
    centers = alg.cell_centers(data).values.numpy()
    data.fields["c"] = td.Field(td.Constant(data), _scalars(centers[:, 0] + 0.5 * centers[:, 1],
                                                            dtype))
    p = data.positions()
    data.fields["p"] = td.Field(td.H1(data), _scalars(p[:, 0] + p[:, 2], dtype))
    return data


def test_threshold_by_cell_values(backend):
    data = _values_data()
    c = data.fields["c"].values.numpy()
    out = td.threshold(data, "c", -0.2, 0.4)
    kept = np.flatnonzero((c >= -0.2) & (c <= 0.4))
    assert out.num_cells == len(kept)
    np.testing.assert_allclose(out.fields["c"].values.numpy(), c[kept])
    used = np.unique(data.topology.connectivity.numpy().reshape(-1, 8)[kept])
    assert out.num_points == len(used)
    np.testing.assert_allclose(out.positions(), data.positions()[used])
    np.testing.assert_allclose(out.fields["p"].values.numpy(),
                               data.fields["p"].values.numpy()[used])


@pytest.mark.parametrize("all_points", [True, False])
def test_threshold_by_point_values(backend, all_points):
    data = _values_data()
    p = data.fields["p"].values.numpy()
    rows = data.topology.connectivity.numpy().reshape(-1, 8)
    inside = (p[rows] >= -0.5) & (p[rows] <= 0.8)
    expected = inside.all(axis=1) if all_points else inside.any(axis=1)
    out = td.threshold(data, "p", -0.5, 0.8, all_points=all_points)
    assert out.num_cells == expected.sum()


def test_threshold_of_a_rectilinear_grid_and_nothing(backend):
    data = _with_radius(_grid(5, "rectilinear"))
    out = td.threshold(data, "r", 0.0, 0.9)
    assert out.num_cells > 0 and isinstance(out.topology, td.UnstructuredTopology)
    assert set(out.topology.types.numpy()) == {12}
    assert (out.fields["r"].values.numpy() <= 0.9 + 1e-6).all()
    empty = td.threshold(data, "r", 10.0, 11.0)
    assert empty.num_cells == 0 and empty.num_points == 0


def test_threshold_carries_l2_blocks(backend):
    """A DG field of an order per cell and an L2 geometry come through cell by cell,
    each kept cell with its own block and order."""
    data = _values_data()
    n = data.num_cells
    orders = (np.arange(n) % 3 != 0).astype(int)
    space = td.L2(data, order=orders)
    values = np.arange(space.size, dtype=float)
    data.fields["dg"] = td.Field(space, _scalars(values))
    centers = alg.cell_centers(data).values.numpy()
    connectivity = data.topology.connectivity.numpy().reshape(-1, 8)
    corners = (centers[:, None, :] + 0.9 * (data.positions()[connectivity]
                                            - centers[:, None, :])).reshape(-1, 3)
    shrunk = td.DataSet(data.topology, td.Field(td.L2(data), _vectors(corners)),
                        fields=dict(data.fields))
    c = data.fields["c"].values.numpy()
    kept = np.flatnonzero((c >= -0.2) & (c <= 0.4))
    out = td.threshold(shrunk, "c", -0.2, 0.4)
    assert isinstance(out.geometry.space, td.L2) and out.dtype == data.dtype
    np.testing.assert_allclose(out.geometry.values.numpy(),
                               corners.reshape(n, 8, 3)[kept].reshape(-1, 3), atol=1e-6)
    dg = out.fields["dg"]
    offsets = space.offsets.numpy()
    expected = np.concatenate([values[offsets[k]:offsets[k + 1]] for k in kept])
    np.testing.assert_allclose(dg.values.numpy(), expected)
    np.testing.assert_array_equal(dg.space.cell_keys().numpy(), orders[kept])


@needs_vtk
@pytest.mark.parametrize("name, lower, upper", [("c", -0.2, 0.4), ("p", -0.5, 0.8)])
def test_threshold_is_vtks(f64_backend, name, lower, upper):
    data = _values_data(tack.f64)
    threshold = vtk.vtkThreshold(lower_threshold=lower, upper_threshold=upper,
                                 threshold_function=vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.SetInputArrayToProcess(0, 0, 0, 1 if name == "c" else 0, name)
    theirs = vtk_to_dataset(threshold(dataset_to_vtk(data)), dtype=tack.f64)
    ours = td.threshold(data, name, lower, upper)
    assert (ours.num_cells, ours.num_points) == (theirs.num_cells, theirs.num_points)
    assert _cells(ours) == _cells(theirs)


# ── External faces ──────────────────────────────────────────────────

@pytest.mark.parametrize("form", ["rectilinear", "explicit"])
def test_external_faces_of_a_grid(backend, form):
    data = _with_radius(_grid(4, form))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    surface = td.extract_surface(data)
    assert surface.num_cells == 6 * 9
    assert set(surface.topology.types.numpy()) == {9}
    assert surface.fields["r"].space is td.H1(surface)
    # Each face carries its cell's value: the cell whose points it is made of.
    cells = surface.fields["cell"].values.numpy().astype(int)
    faces = data.topology.faces()
    np.testing.assert_array_equal(
        cells, faces.sides.numpy()[td.algorithms.boundary_faces(data).numpy(), 0])
    face_rows = surface.topology.connectivity.numpy().reshape(-1, 4)
    cell_points = [set(alg_cell) for alg_cell in _cell_point_ids(data)]
    assert all(set(row) <= cell_points[c] for row, c in zip(face_rows, cells))
    # And faces out of it.
    centers = alg.cell_centers(data).values.numpy()[cells]
    points = surface.positions()
    rows = surface.topology.connectivity.numpy().reshape(-1, 4)
    normal = np.cross(points[rows[:, 1]] - points[rows[:, 0]], points[rows[:, 2]] - points[rows[:, 0]])
    assert ((points[rows].mean(axis=1) - centers) * normal).sum(axis=1).min() > 0


def _quadratic(x):
    return x[:, 0] ** 2 + x[:, 1] * x[:, 2] - 0.5 * x[:, 2]


def _bend(x):
    return x + 0.1 * np.c_[x[:, 1] ** 2, x[:, 2] ** 2, x[:, 0] ** 2]


@pytest.mark.parametrize("mesh", ["rectilinear", "explicit", "tetrahedra"])
def test_order_two_stays_order_two_on_the_surface(backend, mesh):
    """The surface's nodes are the solid's: a face's corners, edge middles and (a
    quad's) center. An order-2 field and an order-2 geometry are gathered onto
    them, exactly; the surface used to keep only the points' values."""
    data = _grid(4, "rectilinear" if mesh == "rectilinear" else "explicit")
    if mesh == "tetrahedra":
        data = td.tetrahedralize(data)
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy()
    data.fields["u"] = td.Field(space, _scalars(_quadratic(nodes)))
    surface = td.extract_surface(data)
    u = surface.fields["u"]
    assert u.space is td.H1(surface, order=2)
    at = td.project(surface, surface.geometry, u.space).values.numpy()
    np.testing.assert_allclose(u.values.numpy(), _quadratic(at), atol=1e-5)
    # Inside a face too: the quadratic is in the triangle's and the quad's spaces.
    centers = td.cell_centers(surface).values.numpy()
    np.testing.assert_allclose(td.values_at_centers(surface, "u").values.numpy(),
                               _quadratic(centers), atol=1e-5)
    # A curved geometry: the surface's nodes where the solid's are.
    data.geometry = td.Field(space, tack.array(_bend(nodes).astype(np.float32), tack.vec3()))
    curved = td.extract_surface(data, fields=[])
    assert curved.geometry.space is td.H1(curved, order=2)
    np.testing.assert_allclose(curved.geometry.values.numpy(), _bend(at), atol=1e-5)


def test_an_order_two_surface_needs_no_faces_of_its_own(backend):
    """Two hexahedra sharing one edge: four boundary quads meet there, which no face
    of the surface could have. Its order-2 layout has no face values, so it
    derives no faces."""
    points = np.array([(x, y, z) for z in (0, 1) for y in (0, 1, 2) for x in (0, 1, 2)],
                      np.float32)
    pid = lambda i, j, k: i + 3 * (j + 3 * k)
    hexes = [[pid(i, j, 0), pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i, j + 1, 0),
              pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)]
             for i, j in ((0, 0), (1, 1))]
    data = td.DataSet(td.UnstructuredTopology([12, 12], [0, 8, 16], np.concatenate(hexes),
                                              num_points=len(points)), points)
    space = td.H1(data, order=2)
    nodes = td.project(data, data.geometry, space).values.numpy()
    data.fields["u"] = td.Field(space, _scalars(_quadratic(nodes)))
    surface = td.extract_surface(data)
    assert surface.num_cells == 12
    with pytest.raises(ValueError, match="shared by more than two cells"):
        surface.topology.faces()
    at = td.project(surface, surface.geometry, td.H1(surface, order=2)).values.numpy()
    np.testing.assert_allclose(surface.fields["u"].values.numpy(), _quadratic(at), atol=1e-5)


@tack.kernel
def _rows_of(cells, out):
    for c in cells:
        for j in range(cells.NUM_POINTS):
            out[cells.entity_id(c), j] = cells.point_id(c, j)


def _cell_point_ids(data):
    """Each hexahedron's point ids, from its view: explicit or structured alike."""
    out = tack.empty((data.num_cells, 8), tack.i32)
    td.for_each(_rows_of, data, "cells", out)
    return out.numpy()


@needs_vtk
@pytest.mark.parametrize("kind", [10, 12, 13, 14])
def test_external_faces_are_vtks(f64_backend, kind):
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    grid = vtkCellTypeSource(cell_type=kind, blocks_dimensions=(3, 2, 2))()
    out = vtk.vtkGeometryFilter()(grid)
    offsets = np.asarray(out.polys.offsets_array)
    connectivity = np.asarray(out.polys.connectivity_array)
    # vtkGeometryFilter renumbers its points; compare faces by their points' places.
    at = np.round(np.asarray(out.points.data), 6)
    theirs = {frozenset(map(tuple, at[connectivity[offsets[i]:offsets[i + 1]]]))
              for i in range(len(offsets) - 1)}
    surface = td.extract_surface(vtk_to_dataset(grid, dtype=tack.f64))
    t = surface.topology
    o, c = t.offsets.numpy(), t.connectivity.numpy()
    ours_at = np.round(surface.positions(), 6)
    ours = {frozenset(map(tuple, ours_at[c[o[i]:o[i + 1]]])) for i in range(len(o) - 1)}
    assert len(ours) == surface.num_cells == len(theirs) and ours == theirs


# ── Edges ───────────────────────────────────────────────────────────

def _lines(data):
    """A line dataset's point id pairs, ``(num_cells, 2)``."""
    t = data.topology
    assert set(t.types.numpy()) <= {3}
    np.testing.assert_array_equal(t.offsets.numpy(), 2 * np.arange(data.num_cells + 1))
    return t.connectivity.numpy().reshape(-1, 2)


@pytest.mark.parametrize("form", ["rectilinear", "explicit"])
def test_extract_edges_of_a_grid(backend, form):
    data = _with_radius(_grid(4, form))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    data.fields["length"] = td.edge_lengths(data)
    edges = data.topology.edges()
    lines = td.extract_edges(data)
    assert lines.num_cells == edges.num_edges == 3 * 3 * 4 * 4
    assert lines.num_points == data.num_points
    np.testing.assert_array_equal(_lines(lines), edges.rows.numpy())     # low to high
    assert lines.fields["r"].space is td.H1(lines)
    assert "cell" not in lines.fields                  # an edge has several cells
    # Values on edges become the lines' values: each its own length.
    assert lines.fields["length"].space is td.Values(lines, "cells")
    points = lines.positions()
    rows = _lines(lines)
    np.testing.assert_allclose(lines.fields["length"].values.numpy(),
                               np.linalg.norm(points[rows[:, 1]] - points[rows[:, 0]], axis=1),
                               rtol=1e-6)


def test_extract_edges_of_a_set(backend):
    data = _grid(3, "explicit")
    rows = data.topology.edges().rows.numpy()
    data.sets["some"] = td.Set(data, "edges", [5, 2, 30])
    data.sets["more"] = td.Set(data, "edges", [2, 4])
    data.sets["origin"] = td.Set(data, "points", [0])
    data.sets["first"] = td.Set(data, "cells", [0])
    for given in ("some", data.sets["some"], [2, 5, 30], tack.array(np.array([2, 5, 30]))):
        lines = td.extract_edges(data, given)
        np.testing.assert_array_equal(_lines(lines), rows[[2, 5, 30]])
    # Sets go by the same maps: an edge set onto the lines it became, points kept.
    np.testing.assert_array_equal(lines.sets["more"].numpy(), [0])
    assert lines.sets["more"].on == "cells"
    np.testing.assert_array_equal(lines.sets["origin"].numpy(), [0])
    assert "first" not in lines.sets
    assert td.extract_edges(data, []).num_cells == 0
    with pytest.raises(ValueError, match="not a set of faces"):
        td.extract_edges(data, td.boundary_faces(data))
    with pytest.raises(ValueError, match="another topology"):
        td.extract_edges(data, td.Set(_grid(3, "explicit"), "edges", [0]))
    with pytest.raises(IndexError, match="edge ids lie outside"):
        td.extract_edges(data, [len(rows)])
    with pytest.raises(TypeError, match="H1 geometry"):
        td.extract_edges(td.DataSet(data.topology, td.project(data, data.geometry,
                                                              td.L2(data))))


def test_extract_edges_of_polyhedra_and_surfaces(backend):
    data = _grid(3, "explicit")
    lines = _lines(td.extract_edges(data))
    np.testing.assert_array_equal(_lines(td.extract_edges(td.as_polyhedra(data))), lines)
    flat = td.uniform_grid((4, 4, 1))
    assert td.extract_edges(flat).num_cells == 24
    np.testing.assert_array_equal(_lines(td.extract_edges(td.as_polygons(flat))),
                                  _lines(td.extract_edges(flat)))


@needs_vtk
@pytest.mark.parametrize("kind", [10, 12, 13, 14])
def test_extract_edges_is_vtks(f64_backend, kind):
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    grid = vtkCellTypeSource(cell_type=kind, blocks_dimensions=(3, 2, 2))()
    out = vtk.vtkExtractEdges()(grid)
    connectivity = np.asarray(out.lines.connectivity_array).reshape(-1, 2)
    at = np.round(np.asarray(out.points.data), 6)
    theirs = {frozenset(map(tuple, at[row])) for row in connectivity}
    lines = td.extract_edges(vtk_to_dataset(grid, dtype=tack.f64))
    ours_at = np.round(lines.positions(), 6)
    ours = {frozenset(map(tuple, ours_at[row])) for row in _lines(lines)}
    assert len(ours) == lines.num_cells == len(theirs) and ours == theirs

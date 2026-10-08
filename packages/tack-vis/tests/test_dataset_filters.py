"""Tests for tack.data.filters: contour, slice, threshold and external faces.

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
import tack.data as td
from tack.data import algorithms as alg
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel, vtkFiltersCore
    from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
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
    f = tack.field(dtype, shape=(len(values),))
    if len(values):
        f.from_numpy(values.astype(dtype.numpy_dtype))
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
    return points, surface.topology.connectivity.to_numpy().reshape(-1, 3)


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
    np.testing.assert_allclose(surface.fields["r"].values.to_numpy(), 0.6, atol=1e-5)
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
    dg = alg.discontinuous(data, data.fields["r"])
    same = td.contour(data, dg, 0.7)
    # The same function: the same triangles, but points merged only within cells,
    # so the surface is no longer stitched across cell faces.
    assert _oriented(*_triangles(same)) == _oriented(*_triangles(continuous))
    assert same.num_points > continuous.num_points
    assert (_edge_uses(_triangles(same)[1]) == 1).any()
    # Each cell its own offset: every cell's piece moves, and pieces part.
    offsets = td.L2(data).offsets.to_numpy()
    values = dg.values.to_numpy()
    for cell in range(data.num_cells):
        values[offsets[cell]:offsets[cell + 1]] += 0.02 * (cell % 3 - 1)
    shifted = td.contour(data, td.Field(dg.space, _scalars(values)), 0.7)
    assert _oriented(*_triangles(shifted)) != _oriented(*_triangles(continuous))


def test_a_variable_order_field_contours_its_order_one_cells(backend):
    data = _with_radius(_grid(7, "explicit"))
    radius = data.fields["r"].values.to_numpy()
    orders = (np.arange(data.num_cells) % 2).astype(int)
    space = td.L2(data, order=orders)
    connectivity = data.topology.connectivity.to_numpy().reshape(-1, 8)
    offsets = space.offsets.to_numpy()
    values = np.empty(space.size)
    for cell in range(data.num_cells):
        values[offsets[cell]:offsets[cell + 1]] = (radius[connectivity[cell]] if orders[cell]
                                                   else 0.0)
    surface = td.contour(data, td.Field(space, _scalars(values)), 0.7)
    # Order-0 cells are constant: no level set inside them. The rest are the
    # continuous contour's triangles from those cells.
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    full = td.contour(data, "r", 0.7, merge_points=False)
    from_order_one = orders[full.fields["cell"].values.to_numpy().astype(int)] == 1
    points, triangles = _triangles(full)
    assert _oriented(*_triangles(surface)) == _oriented(points, triangles[from_order_one])


def test_fields_carried_onto_the_surface(backend):
    data = _with_radius(_grid(6, "explicit"))
    positions = data.positions()
    data.fields["x"] = td.Field(td.H1(data), _scalars(positions[:, 0]))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    surface = td.contour(data, "r", 0.7)
    points = surface.positions()
    np.testing.assert_allclose(surface.fields["x"].values.to_numpy(), points[:, 0], atol=1e-5)
    assert surface.fields["cell"].space is td.Values(surface, "cells")
    # Each triangle lies in the cell it says it came from.
    _, triangles = _triangles(surface)
    cells = surface.fields["cell"].values.to_numpy().astype(int)
    corners = positions[data.topology.connectivity.to_numpy().reshape(-1, 8)]
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
    data.fields["id"] = td.Field(td.H1(data), tack.arange(data.num_points, tack.i32))
    with pytest.raises(TypeError, match="floating-point"):
        td.contour(data, "id", 1.0)
    with pytest.raises(TypeError, match="with a basis"):
        td.contour(data, td.Field(td.Values(data, "faces"),
                                  _scalars(np.zeros(data.topology.faces().num_faces))), 0.5)


def _vtk_contour(data, name, isovalue):
    grid = dataset_to_vtk(data)
    grid.GetPointData().SetActiveScalars(name)
    contour = vtk.vtkContourGrid()
    contour.SetInputData(grid)
    contour.SetValue(0, isovalue)
    contour.Update()
    out = contour.GetOutput()
    if not out.GetNumberOfCells():
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    polys = out.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    return (vtk_to_numpy(out.GetPoints().GetData()),
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
        cloud_points = vtk.vtkPoints()
        cloud_points.SetData(numpy_to_vtk(rng.uniform(-1, 1, (300, 3)), deep=1))
        cloud = vtk.vtkPolyData()
        cloud.SetPoints(cloud_points)
        delaunay = vtk.vtkDelaunay3D()
        delaunay.SetInputData(cloud)
        delaunay.Update()
        data = vtk_to_dataset(delaunay.GetOutput(), dtype=tack.f64)
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
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    connectivity = data.topology.connectivity.to_numpy().reshape(-1, 8)
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
    f = tack.Vector.field(3, tack.f32, shape=(len(values),))
    f.from_numpy(values.astype(np.float32))
    return f


@needs_vtk
def test_slice_is_vtks(f64_backend):
    data = _with_radius(_grid(6, "explicit", tack.f64), tack.f64)
    origin, normal = [0.23, 0.11, -0.17], [1.0, 0.4, 0.7]
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)
    cutter = vtk.vtkCutter()
    cutter.SetCutFunction(plane)
    cutter.SetInputData(dataset_to_vtk(data))
    cutter.Update()
    out = cutter.GetOutput()
    polys = out.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    theirs = _oriented(vtk_to_numpy(out.GetPoints().GetData()),
                       np.array([connectivity[offsets[i]:offsets[i + 1]]
                                 for i in range(len(offsets) - 1)]))
    ours = _oriented(*_triangles(td.slice_plane(data, origin, normal)))
    assert len(ours) == len(theirs) and ours == theirs


# ── Threshold ───────────────────────────────────────────────────────

def _cells(data):
    """Each cell as (type, its points' coordinates in order), in cell order."""
    points = data.positions()
    t = data.topology
    offsets, connectivity = t.offsets.to_numpy(), t.connectivity.to_numpy()
    return [(int(k), tuple(tuple(np.round(points[p], 6)) for p in
                           connectivity[offsets[i]:offsets[i + 1]]))
            for i, k in enumerate(t.types.to_numpy())]


def _values_data(dtype=tack.f32):
    data = _grid(5, "explicit", dtype)
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    data.fields["c"] = td.Field(td.Constant(data), _scalars(centers[:, 0] + 0.5 * centers[:, 1],
                                                            dtype))
    p = data.positions()
    data.fields["p"] = td.Field(td.H1(data), _scalars(p[:, 0] + p[:, 2], dtype))
    return data


def test_threshold_by_cell_values(backend):
    data = _values_data()
    c = data.fields["c"].values.to_numpy()
    out = td.threshold(data, "c", -0.2, 0.4)
    kept = np.flatnonzero((c >= -0.2) & (c <= 0.4))
    assert out.num_cells == len(kept)
    np.testing.assert_allclose(out.fields["c"].values.to_numpy(), c[kept])
    used = np.unique(data.topology.connectivity.to_numpy().reshape(-1, 8)[kept])
    assert out.num_points == len(used)
    np.testing.assert_allclose(out.positions(), data.positions()[used])
    np.testing.assert_allclose(out.fields["p"].values.to_numpy(),
                               data.fields["p"].values.to_numpy()[used])


@pytest.mark.parametrize("all_points", [True, False])
def test_threshold_by_point_values(backend, all_points):
    data = _values_data()
    p = data.fields["p"].values.to_numpy()
    rows = data.topology.connectivity.to_numpy().reshape(-1, 8)
    inside = (p[rows] >= -0.5) & (p[rows] <= 0.8)
    expected = inside.all(axis=1) if all_points else inside.any(axis=1)
    out = td.threshold(data, "p", -0.5, 0.8, all_points=all_points)
    assert out.num_cells == expected.sum()


def test_threshold_of_a_rectilinear_grid_and_nothing(backend):
    data = _with_radius(_grid(5, "rectilinear"))
    out = td.threshold(data, "r", 0.0, 0.9)
    assert out.num_cells > 0 and isinstance(out.topology, td.UnstructuredTopology)
    assert set(out.topology.types.to_numpy()) == {12}
    assert (out.fields["r"].values.to_numpy() <= 0.9 + 1e-6).all()
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
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)
    connectivity = data.topology.connectivity.to_numpy().reshape(-1, 8)
    corners = (centers[:, None, :] + 0.9 * (data.positions()[connectivity]
                                            - centers[:, None, :])).reshape(-1, 3)
    shrunk = td.DataSet(data.topology, td.Field(td.L2(data), _vectors(corners)),
                        fields={k: v for k, v in data.fields.items() if k != "shape"})
    c = data.fields["c"].values.to_numpy()
    kept = np.flatnonzero((c >= -0.2) & (c <= 0.4))
    out = td.threshold(shrunk, "c", -0.2, 0.4)
    assert isinstance(out.geometry.space, td.L2) and out.dtype == data.dtype
    np.testing.assert_allclose(out.geometry.values.to_numpy(vectors=True),
                               corners.reshape(n, 8, 3)[kept].reshape(-1, 3), atol=1e-6)
    dg = out.fields["dg"]
    offsets = space.offsets.to_numpy()
    expected = np.concatenate([values[offsets[k]:offsets[k + 1]] for k in kept])
    np.testing.assert_allclose(dg.values.to_numpy(), expected)
    np.testing.assert_array_equal(dg.space.cell_keys().to_numpy(), orders[kept])


@needs_vtk
@pytest.mark.parametrize("name, lower, upper", [("c", -0.2, 0.4), ("p", -0.5, 0.8)])
def test_threshold_is_vtks(f64_backend, name, lower, upper):
    data = _values_data(tack.f64)
    threshold = vtk.vtkThreshold()
    threshold.SetInputData(dataset_to_vtk(data))
    threshold.SetInputArrayToProcess(0, 0, 0, 1 if name == "c" else 0, name)
    threshold.SetLowerThreshold(lower)
    threshold.SetUpperThreshold(upper)
    threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    threshold.Update()
    theirs = vtk_to_dataset(threshold.GetOutput(), dtype=tack.f64)
    ours = td.threshold(data, name, lower, upper)
    assert (ours.num_cells, ours.num_points) == (theirs.num_cells, theirs.num_points)
    assert _cells(ours) == _cells(theirs)


# ── External faces ──────────────────────────────────────────────────

@pytest.mark.parametrize("form", ["rectilinear", "explicit"])
def test_external_faces_of_a_grid(backend, form):
    data = _with_radius(_grid(4, form))
    data.fields["cell"] = td.Field(td.Constant(data), _scalars(np.arange(data.num_cells)))
    surface = td.external_faces(data)
    assert surface.num_cells == 6 * 9
    assert set(surface.topology.types.to_numpy()) == {9}
    assert surface.fields["r"].space is td.H1(surface)
    # Each face carries its cell's value: the cell whose points it is made of.
    cells = surface.fields["cell"].values.to_numpy().astype(int)
    faces = data.topology.faces()
    np.testing.assert_array_equal(
        cells, faces.sides.to_numpy(vectors=True)[data.sets["boundary"].to_numpy(), 0])
    face_rows = surface.topology.connectivity.to_numpy().reshape(-1, 4)
    cell_points = [set(alg_cell) for alg_cell in _cell_point_ids(data)]
    assert all(set(row) <= cell_points[c] for row, c in zip(face_rows, cells))
    # And faces out of it.
    centers = alg.cell_centers(data).values.to_numpy(vectors=True)[cells]
    points = surface.positions()
    rows = surface.topology.connectivity.to_numpy().reshape(-1, 4)
    normal = np.cross(points[rows[:, 1]] - points[rows[:, 0]], points[rows[:, 2]] - points[rows[:, 0]])
    assert ((points[rows].mean(axis=1) - centers) * normal).sum(axis=1).min() > 0


@tack.kernel
def _rows_of(cells, out):
    for c in cells:
        for j in range(cells.NUM_POINTS):
            out[cells.entity_id(c), j] = cells.point_id(c, j)


def _cell_point_ids(data):
    """Each hexahedron's point ids, from its view: explicit or structured alike."""
    out = tack.field(tack.i32, shape=(data.num_cells, 8))
    td.for_each(_rows_of, data, "cells", out)
    return out.to_numpy()


@needs_vtk
@pytest.mark.parametrize("kind", [10, 12, 13, 14])
def test_external_faces_are_vtks(f64_backend, kind):
    from vtkmodules.vtkFiltersSources import vtkCellTypeSource

    source = vtkCellTypeSource()
    source.SetCellType(kind)
    source.SetBlocksDimensions(3, 2, 2)
    source.Update()
    grid = source.GetOutput()
    geometry = vtk.vtkGeometryFilter()
    geometry.SetInputData(grid)
    geometry.Update()
    out = geometry.GetOutput()
    polys = out.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray())
    connectivity = vtk_to_numpy(polys.GetConnectivityArray())
    # vtkGeometryFilter renumbers its points; compare faces by their points' places.
    at = np.round(vtk_to_numpy(out.GetPoints().GetData()), 6)
    theirs = {frozenset(map(tuple, at[connectivity[offsets[i]:offsets[i + 1]]]))
              for i in range(len(offsets) - 1)}
    surface = td.external_faces(vtk_to_dataset(grid, dtype=tack.f64))
    t = surface.topology
    o, c = t.offsets.to_numpy(), t.connectivity.to_numpy()
    ours_at = np.round(surface.positions(), 6)
    ours = {frozenset(map(tuple, ours_at[c[o[i]:o[i + 1]]])) for i in range(len(o) - 1)}
    assert len(ours) == surface.num_cells == len(theirs) and ours == theirs

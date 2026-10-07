"""Tests for tack.data.shapes: VTK's linear cells as template classes.

VTK is the reference. Its tables (points, centers, edges, faces) are
recorded below as literals, so every machine checks Tack against them;
where VTK is installed, the recorded tables are checked against VTK too,
and the shape functions, derivatives and world-to-parametric inversion
are compared with VTK's own. The properties that need no reference --
the shape functions sum to one, are one at their own point and zero at
the others, have the gradients finite differences give, and invert
``interpolate_point`` -- run everywhere.

Each test runs once per shape class, passing an instance to the kernel as
its template argument, as user code does.
"""

import types

import numpy as np
import pytest

import tack
from tack.data import shapes as sh
from tack.runtime.dispatch import env_flag

# Only VTK's data model: plain `import vtk` also loads its rendering
# libraries, which a headless runner may not have.
try:
    from vtkmodules import vtkCommonCore, vtkCommonDataModel
except ImportError:
    if env_flag("TACK_REQUIRE_VTK"):
        raise
    vtk = None
else:
    vtk = types.SimpleNamespace(reference=vtkCommonCore.reference,
                                **{name: getattr(vtkCommonDataModel, name)
                                   for name in dir(vtkCommonDataModel)
                                   if name.startswith("vtk")})

# CI sets TACK_REQUIRE_VTK on the job that installs VTK, so a missing or
# broken VTK fails there instead of skipping the comparisons.
needs_vtk = pytest.mark.skipif(vtk is None, reason="needs VTK, the reference")

# VTK 9.7.1's tables: GetParametricCoords, GetParametricCenter, GetEdge and
# GetFace point ids, and each face's cell type.
EXPECTED = {
    sh.Vertex: {"dim": 0, "points": [(0, 0, 0)], "center": (0, 0, 0), "edges": [], "faces": [],
                "face_shapes": []},
    sh.Line: {"dim": 1, "points": [(0, 0, 0), (1, 0, 0)], "center": (0.5, 0, 0), "edges": [],
              "faces": [], "face_shapes": []},
    sh.Triangle: {"dim": 2, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                  "center": (1 / 3, 1 / 3, 0), "edges": [(0, 1), (1, 2), (2, 0)], "faces": [],
                  "face_shapes": []},
    sh.Pixel: {"dim": 2, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
               "center": (0.5, 0.5, 0), "edges": [(0, 1), (1, 3), (2, 3), (0, 2)], "faces": [],
               "face_shapes": []},
    sh.Quad: {"dim": 2, "points": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
              "center": (0.5, 0.5, 0), "edges": [(0, 1), (1, 2), (2, 3), (3, 0)], "faces": [],
              "face_shapes": []},
    sh.Tetra: {"dim": 3, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
               "center": (0.25, 0.25, 0.25),
               "edges": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
               "faces": [(0, 1, 3), (1, 2, 3), (2, 0, 3), (0, 2, 1)],
               "face_shapes": [5, 5, 5, 5]},
    sh.Voxel: {"dim": 3, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                                    (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
               "center": (0.5, 0.5, 0.5),
               "edges": [(0, 1), (1, 3), (2, 3), (0, 2), (4, 5), (5, 7), (6, 7), (4, 6),
                         (0, 4), (1, 5), (2, 6), (3, 7)],
               "faces": [(2, 0, 6, 4), (1, 3, 5, 7), (0, 1, 4, 5), (3, 2, 7, 6), (1, 0, 3, 2),
                         (4, 5, 6, 7)],
               "face_shapes": [8] * 6},
    sh.Hexahedron: {"dim": 3, "points": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
                    "center": (0.5, 0.5, 0.5),
                    "edges": [(0, 1), (1, 2), (3, 2), (0, 3), (4, 5), (5, 6), (7, 6), (4, 7),
                              (0, 4), (1, 5), (3, 7), (2, 6)],
                    "faces": [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2),
                              (0, 3, 2, 1), (4, 5, 6, 7)],
                    "face_shapes": [9] * 6},
    sh.Wedge: {"dim": 3, "points": [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1),
                                    (0, 1, 1)],
               "center": (1 / 3, 1 / 3, 0.5),
               "edges": [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
               "faces": [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
               "face_shapes": [5, 5, 9, 9, 9]},
    sh.Pyramid: {"dim": 3, "points": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1)],
                 "center": (0.4, 0.4, 0.2),
                 "edges": [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)],
                 "faces": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
                 "face_shapes": [9, 5, 5, 5, 5]},
}

SHAPES = list(sh.SHAPES)
each_shape = pytest.mark.parametrize("shape", SHAPES, ids=lambda cls: cls.__name__)
SOLIDS = (sh.Tetra, sh.Voxel, sh.Hexahedron, sh.Wedge, sh.Pyramid)


def _vtk_cell(shape):
    cell = getattr(vtk, f"vtk{shape.__name__}")()
    for i in range(cell.GetNumberOfPoints()):
        cell.GetPointIds().SetId(i, i)
    return cell


def _vectors(a, dtype):
    a = np.asarray(a)
    f = tack.Vector.field(3, dtype, shape=a.shape[:-1])
    f.from_numpy(a.astype(np.float64 if dtype == tack.f64 else np.float32))
    return f


# ── Tables ──────────────────────────────────────────────────────────

@tack.kernel
def _describe(cell, counts, center, points, edges, faces, face_shapes):
    for i in range(counts.shape[0]):
        counts[i] = [cell.ID, cell.NUM_POINTS, cell.DIMENSION, cell.NUM_EDGES, cell.NUM_FACES]
        center[i] = cell.parametric_center()
        for j in range(cell.NUM_POINTS):
            points[j] = cell.parametric_point(j)
        for e in range(cell.NUM_EDGES):
            edges[e, 0] = cell.edge_point(e, 0)
            edges[e, 1] = cell.edge_point(e, 1)
        for f in range(cell.NUM_FACES):
            face_shapes[f] = cell.face_shape(f)
            for k in range(cell.face_num_points(f)):
                faces[f, k] = cell.face_point(f, k)


@each_shape
def test_tables_match_the_recorded_ones(backend, shape):
    want = EXPECTED[shape]
    n = len(want["points"])
    counts = tack.Vector.field(5, tack.i32, shape=(1,))
    center = tack.Vector.field(3, tack.f32, shape=(1,))
    points = tack.Vector.field(3, tack.f32, shape=(8,))
    edges = tack.full(tack.i32, (12, 2), -1)
    faces = tack.full(tack.i32, (6, 4), -1)
    face_shapes = tack.full(tack.i32, (6,), -1)
    _describe(shape(), counts, center, points, edges, faces, face_shapes)
    assert counts.to_numpy().tolist() == [
        shape.ID, n, want["dim"], len(want["edges"]), len(want["faces"])]
    np.testing.assert_allclose(center.to_numpy(), want["center"], atol=1e-7)
    np.testing.assert_array_equal(points.to_numpy(vectors=True)[:n], want["points"])
    assert [tuple(e) for e in edges.to_numpy()[:len(want["edges"])].tolist()] == want["edges"]
    got_faces = [tuple(v for v in f if v >= 0) for f in faces.to_numpy().tolist()]
    assert got_faces[:len(want["faces"])] == want["faces"]
    assert face_shapes.to_numpy()[:len(want["faces"])].tolist() == want["face_shapes"]


@needs_vtk
@each_shape
def test_the_recorded_tables_are_vtks(shape):
    cell, want = _vtk_cell(shape), EXPECTED[shape]
    assert cell.GetCellType() == shape.ID
    assert cell.GetCellDimension() == want["dim"]
    n = cell.GetNumberOfPoints()
    pc = cell.GetParametricCoords()
    np.testing.assert_array_equal(np.array([pc[k] for k in range(3 * n)]).reshape(n, 3),
                                  want["points"])
    center = [0.0, 0.0, 0.0]
    cell.GetParametricCenter(center)
    np.testing.assert_allclose(center, want["center"], atol=1e-6)   # VTK's wedge: 0.333333
    assert [tuple(cell.GetEdge(e).GetPointIds().GetId(k) for k in range(2))
            for e in range(cell.GetNumberOfEdges())] == want["edges"]
    got = []
    for f in range(cell.GetNumberOfFaces()):
        ids = cell.GetFace(f).GetPointIds()
        got.append(tuple(ids.GetId(k) for k in range(ids.GetNumberOfIds())))
    assert got == want["faces"]
    assert [cell.GetFace(f).GetCellType() for f in range(cell.GetNumberOfFaces())] \
        == want["face_shapes"]


def test_shape_class_looks_up_vtk_ids():
    assert [sh.shape_class(cls.ID) for cls in SHAPES] == SHAPES
    assert sh.shape_class(np.int32(12)) is sh.Hexahedron
    for not_linear in (0, 2, 4, 7, 15, 25, 42):
        with pytest.raises(ValueError, match="not one of the linear shapes"):
            sh.shape_class(not_linear)


# ── Shape functions ─────────────────────────────────────────────────

@tack.kernel
def _functions(cell, pcs, values, gradients):
    for i in range(pcs.shape[0]):
        for j in range(cell.NUM_POINTS):
            values[i, j] = cell.shape_function(j, pcs[i])
            gradients[i, j] = cell.shape_gradient(j, pcs[i])


def _evaluate(shape, pcs, dtype):
    pcs = np.asarray(pcs, np.float64)
    n = len(EXPECTED[shape]["points"])
    values = tack.field(dtype, shape=(len(pcs), n))
    gradients = tack.Vector.field(3, dtype, shape=(len(pcs), n))
    _functions(shape(), _vectors(pcs, dtype), values, gradients)
    return values.to_numpy(), gradients.to_numpy(vectors=True)


# Inside, on and outside every domain: the functions are polynomials and
# are defined everywhere.
_AXIS = np.array([-0.25, 0.0, 0.2, 0.5, 0.7, 1.0, 1.25])
GRID = np.stack(np.meshgrid(_AXIS, _AXIS, _AXIS, indexing="ij"), axis=-1).reshape(-1, 3)


@each_shape
def test_shape_functions_sum_to_one(backend, shape):
    values, gradients = _evaluate(shape, GRID, tack.f32)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(gradients.sum(axis=1), 0.0, atol=1e-6)


@each_shape
def test_each_function_is_one_at_its_point_and_zero_at_the_others(backend, shape):
    points = EXPECTED[shape]["points"]
    values, _ = _evaluate(shape, points, tack.f32)
    np.testing.assert_array_equal(values, np.eye(len(points)))


@each_shape
def test_gradients_are_the_functions_derivatives(f64_backend, shape):
    """Central differences of the values, in f64; exact for these polynomials up to rounding."""
    h = 1e-5
    _, gradients = _evaluate(shape, GRID, tack.f64)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        plus, _ = _evaluate(shape, GRID + step, tack.f64)
        minus, _ = _evaluate(shape, GRID - step, tack.f64)
        np.testing.assert_allclose(gradients[..., axis], (plus - minus) / (2 * h), atol=1e-8)


def _vtk_functions(shape, pcs):
    cell = _vtk_cell(shape)
    n, dim = cell.GetNumberOfPoints(), cell.GetCellDimension()
    values, gradients = np.zeros((len(pcs), n)), np.zeros((len(pcs), n, 3))
    for i, pc in enumerate(pcs):
        w = [0.0] * n
        cell.InterpolateFunctions(pc, w)
        d = [0.0] * (dim * n or 3)
        cell.InterpolateDerivs(pc, d)
        values[i] = w
        for axis in range(dim):
            gradients[i, :, axis] = d[axis * n:(axis + 1) * n]
    return values, gradients


@needs_vtk
@each_shape
def test_shape_functions_are_vtks(backend, shape):
    values, gradients = _evaluate(shape, GRID, tack.f32)
    want_v, want_g = _vtk_functions(shape, GRID)
    np.testing.assert_allclose(values, want_v, atol=1e-6)
    np.testing.assert_allclose(gradients, want_g, atol=1e-6)


@needs_vtk
@each_shape
def test_shape_functions_are_vtks_in_f64(f64_backend, shape):
    values, gradients = _evaluate(shape, GRID, tack.f64)
    want_v, want_g = _vtk_functions(shape, GRID)
    np.testing.assert_allclose(values, want_v, atol=1e-15)
    np.testing.assert_allclose(gradients, want_g, atol=1e-15)


# ── Inside ──────────────────────────────────────────────────────────

@tack.kernel
def _inside(cell, pcs, out, tol):
    for i in range(pcs.shape[0]):
        out[i] = cell.is_inside(pcs[i], tol)


def _reference_inside(shape, pc, tol):
    r, s, t = pc
    lo, hi = -tol, 1 + tol
    dim = EXPECTED[shape]["dim"]
    if shape is sh.Vertex:
        return True
    if shape in (sh.Triangle, sh.Tetra):
        coords = (r, s, t)[:dim]
        return all(c >= lo for c in coords) and sum(coords) <= hi
    if shape is sh.Wedge:
        return r >= lo and s >= lo and r + s <= hi and lo <= t <= hi
    return all(lo <= c <= hi for c in (r, s, t)[:dim])


def _is_inside(shape, pcs, tol):
    out = tack.field(tack.i32, shape=(len(pcs),))
    _inside(shape(), _vectors(pcs, tack.f32), out, tol)
    return out.to_numpy().astype(bool)


@each_shape
def test_is_inside(backend, shape):
    tol = 0.01
    want = [_reference_inside(shape, p, tol) for p in GRID]
    assert _is_inside(shape, GRID, tol).tolist() == want


# ── Cell geometry ───────────────────────────────────────────────────

def _random_cell(shape, rng, offset):
    """World points for a well-formed cell of ``shape``, one row per point.

    Simplices are affine images of the reference cell; the others are
    perturbed too, so their maps are not affine, except a pixel and a voxel,
    which VTK defines axis-aligned. 2D cells lie in a plane.
    """
    ref = np.array(EXPECTED[shape]["points"], np.float64)
    dim = EXPECTED[shape]["dim"]
    if shape in (sh.Pixel, sh.Voxel):
        scale = rng.uniform(0.5, 2.0, 3)
        if shape is sh.Pixel:
            scale[2] = 0.0
        return ref * scale + offset
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    a = q * rng.uniform(0.6, 1.5, 3)
    local = ref.copy()
    if shape in (sh.Quad, sh.Hexahedron, sh.Wedge, sh.Pyramid):
        local[:, :dim] += rng.uniform(-0.1, 0.1, (len(ref), dim))
    return local @ a.T + offset


def _random_pcs(shape, rng, count, outside=False):
    """Parametric points well inside ``shape`` (or past one face, if ``outside``)."""
    u = rng.uniform(0.1, 0.9, (count, 3))
    if shape in (sh.Triangle, sh.Tetra, sh.Wedge):
        k = 3 if shape is sh.Tetra else 2
        b = rng.dirichlet(np.ones(k + 1), count) * 0.8 + 0.2 / (k + 1)
        u[:, :k] = b[:, 1:]
    if shape is sh.Pyramid:
        u[:, 2] = rng.uniform(0.1, 0.7, count)
    if outside:
        u[:, 0] = 1.0 if shape in (sh.Triangle, sh.Tetra, sh.Wedge) else 1.15
    u[:, EXPECTED[shape]["dim"]:] = 0.0
    return u


def _normal_offset(shape, cell, rng):
    """A displacement off a 2D cell's plane or a line's direction; none for 3D."""
    dim = EXPECTED[shape]["dim"]
    if dim == 2:
        n = np.cross(cell[1] - cell[0], cell[-1] - cell[0])
        return n / np.linalg.norm(n) * rng.uniform(-0.5, 0.5)
    if dim == 1:
        side = np.cross(cell[1] - cell[0], rng.normal(size=3))
        return side / np.linalg.norm(side) * rng.uniform(0.1, 0.5)
    if dim == 0:
        return rng.normal(size=3) * 0.3
    return np.zeros(3)


def _make_cases(shape, seed, count, offset=0.0, outside=False):
    """``count`` random cells of ``shape``, a parametric point in each, and an offset off it."""
    rng = np.random.default_rng([seed, int(shape.ID)])
    cells, offsets = [], []
    for _ in range(count):
        cell = _random_cell(shape, rng, rng.uniform(-1, 1, 3) + offset)
        cells.append(cell)
        offsets.append(_normal_offset(shape, cell, rng))
    return np.array(cells), _random_pcs(shape, rng, count, outside), np.array(offsets)


@tack.kernel
def _round_trip(cell, cells, pcs, offsets, positions, found, converged):
    for i in range(pcs.shape[0]):
        pts = tack.local_array(tack.f64, 3 * cell.NUM_POINTS)
        for j in range(cell.NUM_POINTS):
            pts[3 * j], pts[3 * j + 1], pts[3 * j + 2] = cells[i, j]
        x = cell.interpolate_point(pts, pcs[i]) + offsets[i]
        positions[i] = x
        pc, ok = cell.world_to_parametric(pts, x)
        found[i] = pc
        converged[i] = ok


@tack.kernel
def _round_trip_f32(cell, cells, pcs, offsets, positions, found, converged):
    for i in range(pcs.shape[0]):
        pts = tack.local_array(tack.f32, 3 * cell.NUM_POINTS)
        for j in range(cell.NUM_POINTS):
            pts[3 * j], pts[3 * j + 1], pts[3 * j + 2] = cells[i, j]
        x = cell.interpolate_point(pts, pcs[i]) + offsets[i]
        positions[i] = x
        pc, ok = cell.world_to_parametric(pts, x)
        found[i] = pc
        converged[i] = ok


def _run_round_trip(shape, cases, dtype):
    cells, pcs, offsets = cases
    n = len(pcs)
    positions = tack.Vector.field(3, dtype, shape=(n,))
    found = tack.Vector.field(3, dtype, shape=(n,))
    converged = tack.field(tack.i32, shape=(n,))
    kernel = _round_trip if dtype == tack.f64 else _round_trip_f32
    kernel(shape(), _vectors(cells, dtype), _vectors(pcs, dtype), _vectors(offsets, dtype),
           positions, found, converged)
    return (positions.to_numpy(vectors=True), found.to_numpy(vectors=True),
            converged.to_numpy())


@each_shape
def test_world_to_parametric_inverts_interpolate_point(backend, shape):
    """f32: the parametric point that made each world point, found again."""
    cases = _make_cases(shape, seed=1, count=20)
    _, found, converged = _run_round_trip(shape, cases, tack.f32)
    assert converged.all()
    np.testing.assert_allclose(found, cases[1], atol=2e-5)


@each_shape
def test_world_to_parametric_inverts_interpolate_point_in_f64(f64_backend, shape):
    cases = _make_cases(shape, seed=2, count=20)
    _, found, converged = _run_round_trip(shape, cases, tack.f64)
    assert converged.all()
    np.testing.assert_allclose(found, cases[1], atol=1e-10)


@each_shape
def test_world_to_parametric_far_from_the_origin(backend, shape):
    """f32 cells a thousand units out: the residual is taken relative to the first point."""
    cases = _make_cases(shape, seed=3, count=20, offset=1000.0)
    _, found, converged = _run_round_trip(shape, cases, tack.f32)
    assert converged.all()
    np.testing.assert_allclose(found, cases[1], atol=5e-4)


@each_shape
def test_world_to_parametric_outside_the_cell(backend, shape):
    cases = _make_cases(shape, seed=4, count=20, outside=True)
    _, found, converged = _run_round_trip(shape, cases, tack.f32)
    assert converged.all()
    np.testing.assert_allclose(found, cases[1], atol=2e-5)
    inside = _is_inside(shape, found, 1e-3)
    assert inside.all() if shape is sh.Vertex else not inside.any()


@needs_vtk
@each_shape
def test_world_to_parametric_is_vtks(f64_backend, shape):
    """VTK's EvaluateLocation and EvaluatePosition agree with Tack's two directions."""
    cells, pcs, offsets = cases = _make_cases(shape, seed=5, count=10)
    positions, found, converged = _run_round_trip(shape, cases, tack.f64)
    assert converged.all()
    dim = EXPECTED[shape]["dim"]
    for i in range(len(pcs)):
        cell = _vtk_cell(shape)
        n = cell.GetNumberOfPoints()
        for j in range(n):
            cell.GetPoints().SetPoint(j, cells[i, j])
        x = [0.0, 0.0, 0.0]
        cell.EvaluateLocation(vtk.reference(0), pcs[i], x, [0.0] * n)
        np.testing.assert_allclose(positions[i] - offsets[i], x, atol=1e-12)
        pc, closest, dist2 = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], vtk.reference(0.0)
        cell.EvaluatePosition(positions[i], closest, vtk.reference(0), pc, dist2, [0.0] * n)
        np.testing.assert_allclose(found[i, :dim], pc[:dim], atol=1e-6)


@tack.kernel
def _geometry(cell, cells, pcs, values, interpolated, positions, jacobians):
    for i in range(pcs.shape[0]):
        pts = tack.local_array(tack.f64, 3 * cell.NUM_POINTS)
        vals = tack.local_array(tack.f64, cell.NUM_POINTS)
        for j in range(cell.NUM_POINTS):
            pts[3 * j], pts[3 * j + 1], pts[3 * j + 2] = cells[i, j]
            vals[j] = values[i, j]
        interpolated[i] = cell.interpolate(vals, pcs[i])
        positions[i] = cell.interpolate_point(pts, pcs[i])
        jacobians[i] = cell.jacobian(pts, pcs[i])


def _run_geometry(shape, cells, pcs, values):
    n = len(pcs)
    vals = tack.field(tack.f64, shape=values.shape)
    vals.from_numpy(values)
    interpolated = tack.field(tack.f64, shape=(n,))
    positions = tack.Vector.field(3, tack.f64, shape=(n,))
    jacobians = tack.Matrix.field(3, 3, tack.f64, shape=(n,))
    _geometry(shape(), _vectors(cells, tack.f64), _vectors(pcs, tack.f64), vals,
              interpolated, positions, jacobians)
    return (interpolated.to_numpy(), positions.to_numpy(vectors=True),
            jacobians.to_numpy().reshape(n, 3, 3))


@each_shape
def test_interpolation_reproduces_linear_fields(f64_backend, shape):
    """A field linear in world coordinates interpolates exactly, on every shape."""
    cells, pcs, _ = _make_cases(shape, seed=6, count=10)
    gradient, constant = np.array([0.3, -1.2, 2.0]), 0.7
    interpolated, positions, _ = _run_geometry(shape, cells, pcs, cells @ gradient + constant)
    np.testing.assert_allclose(interpolated, positions @ gradient + constant, atol=1e-12)


@each_shape
def test_jacobian_is_the_positions_derivative(f64_backend, shape):
    cells, pcs, _ = _make_cases(shape, seed=7, count=10)
    zeros = np.zeros(cells.shape[:2])
    _, _, jacobians = _run_geometry(shape, cells, pcs, zeros)
    h = 1e-6
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = h
        _, plus, _ = _run_geometry(shape, cells, pcs + step, zeros)
        _, minus, _ = _run_geometry(shape, cells, pcs - step, zeros)
        np.testing.assert_allclose(jacobians[:, :, axis], (plus - minus) / (2 * h), atol=1e-8)


@pytest.mark.parametrize("shape", SOLIDS, ids=lambda cls: cls.__name__)
def test_jacobian_of_a_well_formed_solid_has_positive_determinant(f64_backend, shape):
    cells, pcs, _ = _make_cases(shape, seed=8, count=10)
    _, _, jacobians = _run_geometry(shape, cells, pcs, np.zeros(cells.shape[:2]))
    assert (np.linalg.det(jacobians) > 0).all()

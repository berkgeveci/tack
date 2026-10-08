"""Tests for tack.interop.mfem, with MFEM itself (PyMFEM) as the reference.

Meshes are built in MFEM -- Cartesian meshes of each element type, and a
hand-built mixed one -- so the tests need no data files. MFEM's own counts
and records check the derived topology (faces, edges, the elements on each
side of each face, the boundary); its GetValue and GetGradient check fields
of every space the interop maps: H1 order 1, L2 order 0 and 1 (with jumps),
vector fields in both orderings.
"""

import numpy as np
import pytest

import tack
import tack.data as td
from tack.data import algorithms as alg
from tack.runtime.dispatch import env_flag

try:
    import mfem.ser as mfem

    from tack.interop.mfem import mfem_to_dataset
except ImportError:
    if env_flag("TACK_REQUIRE_MFEM"):
        raise
    mfem = None

pytestmark = pytest.mark.skipif(mfem is None, reason="needs PyMFEM, the reference")

KINDS = ["TETRAHEDRON", "HEXAHEDRON", "WEDGE", "PYRAMID"]
MESHES = [*KINDS, "mixed"]


def _cartesian(kind):
    mesh = mfem.Mesh.MakeCartesian3D(3, 2, 2, getattr(mfem.Element, kind), 3.0, 1.5, 2.0)
    # Not axis-aligned, so nothing passes by symmetry.
    mesh.EnsureNodes()
    mesh.Transform(_Skew(3))
    return mesh


class _Skew(mfem.VectorPyCoefficient if mfem else object):
    def EvalValue(self, x):
        return [x[0] + 0.2 * x[1], x[1] + 0.1 * x[2], x[2] + 0.15 * x[0]]


def _mixed():
    """Two hexahedra side by side, a pyramid on the first and a prism on the second."""
    mesh = mfem.Mesh(3, 15, 4, 0, 3)
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1, 2):
                mesh.AddVertex([float(x), float(y), float(z)])
    mesh.AddVertex([0.5, 0.5, 2.0])                    # 12: the pyramid's apex
    mesh.AddVertex([1.5, 0.0, 2.0])                    # 13, 14: the prism's ridge
    mesh.AddVertex([1.5, 1.0, 2.0])

    def v(x, y, z):
        return x + 3 * y + 6 * z

    for i in (0, 1):
        mesh.AddHex([v(i, 0, 0), v(i + 1, 0, 0), v(i + 1, 1, 0), v(i, 1, 0),
                     v(i, 0, 1), v(i + 1, 0, 1), v(i + 1, 1, 1), v(i, 1, 1)], 1 + i)
    mesh.AddPyramid([v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1), 12], 3)
    # The triangle (0, 1, 2) with its normal toward (3, 4, 5): Finalize's
    # orientation fix leaves prisms alone, so it has to be right here.
    mesh.AddWedge([v(1, 0, 1), 13, v(2, 0, 1), v(1, 1, 1), 14, v(2, 1, 1)], 4)
    mesh.FinalizeTopology()
    mesh.Finalize(False, True)
    return mesh


def _mfem_determinants(mesh):
    out = []
    for e in range(mesh.GetNE()):
        T = mesh.GetElementTransformation(e)
        T.SetIntPoint(mfem.Geometries.GetCenter(mesh.GetElementBaseGeometry(e)))
        out.append(np.linalg.det(np.array(T.Jacobian().GetDataArray()).reshape(3, 3)))
    return np.array(out)


def _make(name):
    return _mixed() if name == "mixed" else _cartesian(name)


def _projected(mesh, fec, function, vdim=1, ordering=0):
    fes = mfem.FiniteElementSpace(mesh, fec, vdim, ordering)
    gf = mfem.GridFunction(fes)
    if vdim == 1:
        class F(mfem.PyCoefficient):
            def EvalValue(self, x):
                return function(x)
        gf.ProjectCoefficient(F())
    else:
        class V(mfem.VectorPyCoefficient):
            def EvalValue(self, x):
                return function(x)
        gf.ProjectCoefficient(V(vdim))
    # The grid function points at its space and collection; keep them alive.
    gf.keep = (mesh, fec, fes)
    return gf


def _f(x):
    return x[0] + 2 * x[1] - x[2] + 0.5 * x[0] * x[1]


def _linear(x):
    return x[0] + 2 * x[1] - x[2]


def _centers(mesh):
    """Each element's reference center, where tack's parametric center maps to the
    same point -- every shape but the pyramid, whose two references differ."""
    return [mfem.Geometries.GetCenter(mesh.GetElementBaseGeometry(e))
            for e in range(mesh.GetNE())]


def _not_pyramids(mesh):
    return np.array([mesh.GetElementBaseGeometry(e) != 7 for e in range(mesh.GetNE())])


# ── Topology ────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MESHES)
def test_topology_matches_mfem(backend, name):
    mesh = _make(name)
    data = mfem_to_dataset(mesh)
    assert data.num_cells == mesh.GetNE() and data.num_points == mesh.GetNV()
    faces = data.topology.faces()
    assert faces.num_faces == mesh.GetNFaces()
    assert data.topology.edges().num_edges == mesh.GetNEdges()
    # The elements on each side of each face: the same pairs, side 0 aside.
    theirs = sorted(tuple(sorted(mesh.GetFaceElements(f))) for f in range(mesh.GetNFaces()))
    sides = faces.sides.to_numpy(vectors=True)
    ours = sorted(tuple(sorted((int(s[0]), int(s[2])))) for s in sides)
    assert ours == theirs
    assert faces.boundary().shape[0] == sum(1 for e1, e2 in theirs if e1 < 0)


@tack.kernel
def _determinants(cells, out):
    for c in cells:
        out[cells.entity_id(c)] = cells.geometry_jacobian(c, cells.parametric_center()).determinant()


@pytest.mark.parametrize("name", MESHES)
def test_cells_are_right_side_out(backend, name):
    """Every element MFEM sees as right side out is right side out here: a positive
    Jacobian, and faces whose outward area vectors add up to nothing."""
    mesh = _make(name)
    assert (_mfem_determinants(mesh) > 0).all()
    data = mfem_to_dataset(mesh)
    out = tack.field(tack.f32, shape=(data.num_cells,))
    td.for_each(_determinants, data, "cells", out)
    assert (out.to_numpy() > 0).all()
    normals, areas = alg.face_geometry(data)
    flux = td.Field(td.Values(data, "faces"), tack.Vector.field(3, tack.f32, shape=(areas.values.shape[0],)))
    flux.values.from_numpy((normals.values.to_numpy(vectors=True)
                            * areas.values.to_numpy()[:, None]).astype(np.float32))
    np.testing.assert_allclose(alg.divergence(data, flux).values.to_numpy(vectors=True), 0,
                               atol=1e-5)


# ── Fields ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", MESHES)
def test_h1_values_and_gradients_match_mfem(backend, name):
    mesh = _make(name)
    gf = _projected(mesh, mfem.H1_FECollection(1, 3), _f)
    data = mfem_to_dataset(mesh, {"u": gf})
    assert data.fields["u"].space is td.H1(data)
    np.testing.assert_allclose(data.fields["u"].values.to_numpy(), gf.GetDataArray(),
                               rtol=1e-6)
    ours = alg.values_at_centers(data, data.fields["u"]).values.to_numpy()
    gradients = alg.gradients(data, data.fields["u"]).values.to_numpy(vectors=True)
    gradient = mfem.Vector(3)
    for e, center in enumerate(_centers(mesh)):
        if mesh.GetElementBaseGeometry(e) == 7:
            continue
        np.testing.assert_allclose(ours[e], gf.GetValue(e, center), rtol=1e-5, atol=1e-5)
        T = mesh.GetElementTransformation(e)
        T.SetIntPoint(center)
        gf.GetGradient(T, gradient)
        np.testing.assert_allclose(gradients[e], gradient.GetDataArray(), rtol=1e-4,
                                   atol=1e-4)


@pytest.mark.parametrize("name", MESHES)
def test_l2_fields_and_their_jumps(backend, name):
    """A DG field from MFEM, each element offset by 10 times its number: values at
    the centers are MFEM's, and every interior face jumps by the offsets'
    difference. The function is linear, so its L2 projection is itself on every
    shape and the only jumps are the offsets."""
    mesh = _make(name)
    gf = _projected(mesh, mfem.L2_FECollection(1, 3), _linear)
    fes = gf.FESpace()
    values = gf.GetDataArray()
    for e in range(mesh.GetNE()):
        for d in fes.GetElementDofs(e):
            values[d] += 10.0 * e
    data = mfem_to_dataset(mesh, {"u": gf})
    u = data.fields["u"]
    assert u.space is td.L2(data)
    exact = _not_pyramids(mesh)
    ours = alg.values_at_centers(data, u).values.to_numpy()
    for e, center in enumerate(_centers(mesh)):
        if exact[e]:
            np.testing.assert_allclose(ours[e], gf.GetValue(e, center), rtol=1e-5, atol=1e-4)
    sides = data.topology.faces().sides.to_numpy(vectors=True)
    jumps = alg.jump(data, u).values.to_numpy()
    interior = sides[:, 2] >= 0
    # The function is continuous; only the offsets jump (exactly, where both
    # sides are exact).
    both = interior & exact[sides[:, 0]] & exact[np.where(interior, sides[:, 2], 0)]
    np.testing.assert_allclose(jumps[both], 10.0 * (sides[both, 2] - sides[both, 0]),
                               atol=1e-3)
    np.testing.assert_allclose(jumps[~interior], 0)


def test_cell_constants_vectors_and_attributes(backend):
    mesh = _mixed()
    constant = _projected(mesh, mfem.L2_FECollection(0, 3), _f)
    data = mfem_to_dataset(mesh, {
        "c": constant,
        "x by nodes": _projected(mesh, mfem.H1_FECollection(1, 3), lambda x: list(x), 3, 0),
        "x by vdim": _projected(mesh, mfem.H1_FECollection(1, 3), lambda x: list(x), 3, 1),
    })
    assert data.fields["c"].space is td.Constant(data)
    np.testing.assert_allclose(data.fields["c"].values.to_numpy(), constant.GetDataArray(),
                               rtol=1e-6)
    for name in ("x by nodes", "x by vdim"):
        np.testing.assert_allclose(data.fields[name].values.to_numpy(vectors=True),
                                   data.positions(), atol=1e-6)
    np.testing.assert_array_equal(data.fields["attribute"].values.to_numpy(), [1, 2, 3, 4])


def test_what_is_not_mapped_yet(backend):
    mesh = _cartesian("HEXAHEDRON")
    cubic = _projected(mesh, mfem.H1_FECollection(3, 3), _f)
    with pytest.raises(NotImplementedError, match="H1 order 3"):
        mfem_to_dataset(mesh, {"u": cubic})
    curved = _cartesian("HEXAHEDRON")
    curved.SetCurvature(3)
    with pytest.raises(NotImplementedError, match="order above 2"):
        mfem_to_dataset(curved)
    pyramids = _cartesian("PYRAMID")
    with pytest.raises(NotImplementedError, match="Pyramid"):
        mfem_to_dataset(pyramids, {"u": _projected(pyramids, mfem.H1_FECollection(2, 3), _f)})


# ── Order 2: quadratic fields and curved meshes ─────────────────────

ORDER2 = ["TETRAHEDRON", "HEXAHEDRON", "WEDGE"]


class _Bend(mfem.VectorPyCoefficient if mfem else object):
    """A smooth, curved map: the cells' faces bend."""

    def EvalValue(self, x):
        return [x[0] + 0.15 * np.sin(1.3 * x[1]), x[1] + 0.1 * x[0] * x[2],
                x[2] + 0.12 * np.cos(x[0])]


def _curved(kind):
    mesh = mfem.Mesh.MakeCartesian3D(3, 2, 2, getattr(mfem.Element, kind), 3.0, 1.5, 2.0)
    mesh.SetCurvature(2)
    mesh.Transform(_Bend(3))
    return mesh


def _inside(geometry, rng):
    """A random point inside MFEM's (and tack's) reference element of ``geometry``."""
    if geometry == 4:                                   # tetrahedron
        b = rng.dirichlet(np.ones(4))
        return b[1:]
    if geometry == 6:                                   # prism
        b = rng.dirichlet(np.ones(3))
        return np.array([b[1], b[2], rng.uniform()])
    return rng.uniform(size=3)


@tack.kernel
def _evaluate(cells, u, pcs, values, positions, gradients):
    for c in cells:
        e = cells.entity_id(c)
        pc = pcs[e]
        values[e] = u.value(c, pc)
        positions[e] = cells.position(c, pc)
        gradients[e] = (cells.geometry_jacobian(c, pc).inverse().transpose()
                        @ u.parametric_gradient(c, pc))


def _compare(mesh, data, field, gf, rng, tolerance):
    """``field`` and the geometry at a random point of every element against MFEM."""
    n = mesh.GetNE()
    points = np.array([_inside(mesh.GetElementBaseGeometry(e), rng) for e in range(n)])
    pcs = tack.Vector.field(3, tack.f32, shape=(n,))
    pcs.from_numpy(points.astype(np.float32))
    values = tack.field(tack.f32, shape=(n,))
    positions = tack.Vector.field(3, tack.f32, shape=(n,))
    gradients = tack.Vector.field(3, tack.f32, shape=(n,))
    td.for_each(_evaluate, data, "cells", field, pcs, values, positions, gradients)
    values, positions = values.to_numpy(), positions.to_numpy(vectors=True)
    gradients = gradients.to_numpy(vectors=True)
    gradient = mfem.Vector(3)
    for e in range(n):
        ip = mfem.IntegrationPoint()
        ip.Set3(*(float(x) for x in points[e].astype(np.float32)))
        T = mesh.GetElementTransformation(e)
        T.SetIntPoint(ip)
        np.testing.assert_allclose(positions[e], T.Transform(ip), atol=tolerance)
        np.testing.assert_allclose(values[e], gf.GetValue(e, ip), rtol=tolerance,
                                   atol=tolerance)
        gf.GetGradient(T, gradient)
        np.testing.assert_allclose(gradients[e], gradient.GetDataArray(), rtol=10 * tolerance,
                                   atol=10 * tolerance)


def _quadratic(x):
    return x[0] * x[1] - 0.5 * x[2] ** 2 + 2.0 * x[0] + np.sin(x[1])


@pytest.mark.parametrize("kind", ORDER2)
def test_quadratic_fields_on_straight_meshes(backend, kind):
    mesh = _cartesian(kind)
    gf = _projected(mesh, mfem.H1_FECollection(2, 3), _quadratic)
    data = mfem_to_dataset(mesh, {"u": gf})
    u = data.fields["u"]
    assert u.space is td.H1(data, order=2)
    assert u.space.size == gf.FESpace().GetNDofs()
    _compare(mesh, data, u, gf, np.random.default_rng(1), 2e-5)


@pytest.mark.parametrize("kind", ORDER2)
@pytest.mark.parametrize("order", [1, 2])
def test_fields_on_curved_meshes(backend, kind, order):
    """A curved geometry (order 2) under a field of order 1 or 2: positions, values
    and gradients -- the Jacobian now varies within each cell -- are MFEM's."""
    mesh = _curved(kind)
    gf = _projected(mesh, mfem.H1_FECollection(order, 3), _quadratic)
    data = mfem_to_dataset(mesh, {"u": gf})
    assert data.geometry.space is td.H1(data, order=2)
    _compare(mesh, data, data.fields["u"], gf, np.random.default_rng(order), 3e-5)
    out = tack.field(tack.f32, shape=(data.num_cells,))
    td.for_each(_determinants, data, "cells", out)
    np.testing.assert_allclose(out.to_numpy(), _mfem_determinants(mesh), rtol=1e-4)

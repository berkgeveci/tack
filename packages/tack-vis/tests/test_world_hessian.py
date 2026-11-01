"""``world_hessian``: a field's second derivatives in the world from its basis' second
derivatives and the geometry's. On affine cells, against the exact Hessians of
polynomials the spaces hold; on distorted and curved cells, against central
differences of ``world_gradient`` in float64, which differentiate the same field."""

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes
from tack.dataset.algorithms.gradients import _world_gradient
from tack.dataset.topology import uniform_cells


@tack.kernel
def _hessians(cells, u, out):
    for c in cells:
        h = td.world_hessian(cells, u, c, cells.parametric_center())
        e = cells.entity_id(c)
        for a in range(3):
            for b in range(3):
                out[9 * e + 3 * a + b] = h[a, b]


@tack.kernel
def _differenced(cells, u, out, step):
    for c in cells:
        pc = cells.parametric_center()
        d = tack.mat3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for k in range(3):
            e = tack.vec3(1.0 if k == 0 else 0.0, 1.0 if k == 1 else 0.0, 1.0 if k == 2 else 0.0)
            ahead = _world_gradient(cells, u, c, pc + step * e)
            behind = _world_gradient(cells, u, c, pc - step * e)
            d += ((ahead - behind) / (2.0 * step)).outer_product(e)
        h = d @ cells.geometry_jacobian(c, pc).inverse()
        i = cells.entity_id(c)
        for a in range(3):
            for b in range(3):
                out[9 * i + 3 * a + b] = 0.5 * (h[a, b] + h[b, a])


def _run(data, kernel, *args):
    out = tack.zeros((9 * data.num_cells,), data.dtype)
    td.for_each(kernel, data, "cells", data.fields["u"], out, *args)
    return out.numpy().reshape(-1, 3, 3)


@tack.func
def _p1(x):
    return x[0] * x[1] + 2.0 * x[1] * x[2] - x[0] * x[2] + 3.0 * x[0]


@tack.func
def _p1_wedge(x):
    return x[0] * x[2] + 2.0 * x[1] * x[2] + 3.0 * x[0]


@tack.func
def _p2(x):
    return (x[0] * x[0] - 2.0 * x[1] * x[1] + 0.5 * x[2] * x[2] + x[0] * x[1] - x[1] * x[2]
            + 3.0 * x[0] * x[2])


@tack.func
def _p3(x):
    return x[0] * x[0] * x[0] + x[0] * x[1] * x[2] - x[1] * x[1] * x[2] + x[2] * x[2]


@tack.func
def _p4(x):
    return x[0] * x[0] * x[1] * x[1] + x[2] * x[2] * x[2] * x[2] - x[0] * x[2]


def _h1(x):
    return np.broadcast_to([[0.0, 1.0, -1.0], [1.0, 0.0, 2.0], [-1.0, 2.0, 0.0]], (len(x), 3, 3))


def _h1_wedge(x):
    return np.broadcast_to([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [1.0, 2.0, 0.0]], (len(x), 3, 3))


def _h2(x):
    return np.broadcast_to([[2.0, 1.0, 3.0], [1.0, -4.0, -1.0], [3.0, -1.0, 1.0]], (len(x), 3, 3))


def _h3(x):
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    return np.stack([np.stack([6 * a, c, b], -1), np.stack([c, -2 * c, a - 2 * b], -1),
                     np.stack([b, a - 2 * b, np.full_like(a, 2.0)], -1)], -2)


def _h4(x):
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    z = np.zeros_like(a)
    return np.stack([np.stack([2 * b * b, 4 * a * b, z - 1], -1), np.stack([4 * a * b, 2 * a * a, z], -1),
                     np.stack([z - 1, z, 12 * c * c], -1)], -2)


def _voxels():
    return td.uniform_grid((5, 4, 4), origin=(0.1, -0.2, 0.3), spacing=(0.3, 0.25, 0.4), dtype=tack.f64)


def _tetrahedra():
    return td.tetrahedralize(_voxels())


def _wedges():
    """A unit cube as two right prisms, each triangle (0, 1, 2) turned toward (3, 4, 5)."""
    corners = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], dtype=np.float64)
    cells = np.array([[0, 1, 2, 4, 5, 6], [1, 3, 2, 5, 7, 6]], dtype=np.int32)
    return _mesh(corners * [0.7, 0.9, 0.6] + [0.2, 0.1, 0.4], cells, shapes.WEDGE)


def _pyramids():
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.45, 0.6, 0.9]], dtype=np.float64)
    return _mesh(corners + np.array([0.3, 0.2, 0.1]), np.array([[0, 1, 2, 3, 4]], dtype=np.int32), shapes.PYRAMID)


def _mesh(points, cells, shape):
    types, offsets = uniform_cells(int(shape), len(cells), tack.i32)
    topology = td.UnstructuredTopology(types, offsets, tack.array(cells.reshape(-1)), num_points=len(points))
    positions = tack.empty((len(points),), tack.vec3(tack.f64))
    tack.copyto(positions, points)
    return td.DataSet(topology, positions)


EXACT = [
    ("voxels H1 order 1", _voxels, lambda d: td.H1(d), _p1, _h1),
    ("voxels H1 order 2", _voxels, lambda d: td.H1(d, order=2), _p2, _h2),
    ("voxels L2 order 2", _voxels, lambda d: td.L2(d, order=2), _p2, _h2),
    ("voxels H1 order 3", _voxels, lambda d: td.H1(d, order=3), _p3, _h3),
    ("voxels H1 order 3 Bernstein", _voxels, lambda d: td.H1(d, order=3, basis="bernstein"), _p3, _h3),
    ("voxels H1 order 4", _voxels, lambda d: td.H1(d, order=4), _p4, _h4),
    ("tetrahedra H1 order 2", _tetrahedra, lambda d: td.H1(d, order=2), _p2, _h2),
    ("tetrahedra H1 order 3", _tetrahedra, lambda d: td.H1(d, order=3, basis="equispaced"), _p3, _h3),
    ("tetrahedra L2 order 3", _tetrahedra, lambda d: td.L2(d, order=3, basis="equispaced"), _p3, _h3),
    ("tetrahedra H1 order 3 Bernstein", _tetrahedra, lambda d: td.H1(d, order=3, basis="bernstein"), _p3, _h3),
    ("tetrahedra H1 order 4", _tetrahedra, lambda d: td.H1(d, order=4, basis="equispaced"), _p4, _h4),
    ("wedges H1 order 1", _wedges, lambda d: td.H1(d), _p1_wedge, _h1_wedge),
    ("wedges H1 order 2", _wedges, lambda d: td.H1(d, order=2), _p2, _h2),
    ("wedges H1 order 3", _wedges, lambda d: td.H1(d, order=3, basis="equispaced"), _p3, _h3),
]


@pytest.mark.parametrize(("make", "space", "f", "exact"), [c[1:] for c in EXACT], ids=[c[0] for c in EXACT])
def test_exact_on_affine_cells(f64_backend, make, space, f, exact):
    data = make()
    data.fields["u"] = td.interpolate(data, f, space(data))
    got = _run(data, _hessians)
    want = exact(td.cell_centers(data).values.numpy())
    np.testing.assert_allclose(got, want, atol=1e-9 * max(1.0, np.abs(want).max()))


@tack.func
def _bend(x):
    return tack.vec3(x[0] + 0.12 * x[1] * x[1], x[1] + 0.1 * x[0] * x[2], x[2] + 0.08 * x[0] * x[1])


def _bent(make, order):
    """The cells of ``make`` with their geometry bent at ``order`` (1: the corners moved,
    so voxels stop being parallelepipeds)."""
    data = make()
    space = td.H1(data, order=order, basis="equispaced") if order > 2 else td.H1(data, order=order)
    data.geometry = td.interpolate(data, _bend, space, dtype=tack.vec3(tack.f64))
    return data


CURVED = [
    ("voxels moved, H1 order 2", lambda: _bent(_voxels, 1), lambda d: td.H1(d, order=2)),
    ("voxels moved, H1 order 3", lambda: _bent(_voxels, 1), lambda d: td.H1(d, order=3)),
    ("voxels curved at order 2", lambda: _bent(_voxels, 2), lambda d: td.H1(d, order=2)),
    ("voxels curved at order 3", lambda: _bent(_voxels, 3), lambda d: td.H1(d, order=3)),
    ("tetrahedra curved at order 2", lambda: _bent(_tetrahedra, 2), lambda d: td.H1(d, order=2)),
    ("tetrahedra curved at order 3", lambda: _bent(_tetrahedra, 3),
     lambda d: td.H1(d, order=3, basis="equispaced")),
    ("wedges curved at order 3", lambda: _bent(_wedges, 3), lambda d: td.H1(d, order=3, basis="equispaced")),
    ("pyramids, H1 order 1", _pyramids, lambda d: td.H1(d)),
]


@pytest.mark.parametrize(("make", "space"), [c[1:] for c in CURVED], ids=[c[0] for c in CURVED])
def test_curved_cells_match_differences_of_the_world_gradient(f64_backend, make, space):
    data = make()
    data.fields["u"] = td.interpolate(data, _p3, space(data))
    got = _run(data, _hessians)
    want = _run(data, _differenced, 1e-4)
    np.testing.assert_allclose(got, want, atol=1e-6 * max(1.0, np.abs(want).max()))


def test_float32(backend):
    data = td.uniform_grid((5, 4, 4), origin=(0.1, -0.2, 0.3), spacing=(0.3, 0.25, 0.4))
    data.fields["u"] = td.interpolate(data, _p3, td.H1(data, order=3))
    got = _run(data, _hessians)
    want = _h3(td.cell_centers(data).values.numpy().astype(np.float64))
    np.testing.assert_allclose(got, want, atol=2e-4 * np.abs(want).max())


def test_surfaces_have_none(backend):
    data = td.uniform_grid((5, 4))
    data.fields["u"] = td.interpolate(data, _p2, td.H1(data, order=2))
    assert not _run(data, _hessians).any()


@tack.kernel
def _jets(cells, u, out):
    for c in cells:
        pc = cells.parametric_center()
        v, g, h = td.world_jet(cells, u, c, pc)
        e = cells.entity_id(c)
        out[13 * e] = v - u.value(c, pc)
        for a in range(3):
            out[13 * e + 1 + a] = g[a] - _world_gradient(cells, u, c, pc)[a]
            for b in range(3):
                out[13 * e + 4 + 3 * a + b] = h[a, b]


@pytest.mark.parametrize(("make", "space"), [(c[1], c[2]) for c in EXACT[::2]], ids=[c[0] for c in EXACT[::2]])
def test_jets_are_the_value_gradient_and_hessian(f64_backend, make, space):
    data = make()
    data.fields["u"] = td.interpolate(data, _p2, space(data))
    out = tack.zeros((13 * data.num_cells,), data.dtype)
    td.for_each(_jets, data, "cells", data.fields["u"], out)
    out = out.numpy().reshape(-1, 13)
    np.testing.assert_allclose(out[:, :4], 0.0, atol=1e-10)
    np.testing.assert_allclose(out[:, 4:].reshape(-1, 3, 3), _run(data, _hessians), atol=1e-10)

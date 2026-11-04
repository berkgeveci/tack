# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""HDiv and HCurl reproduce the fields their spaces hold, through their Piola maps."""

import itertools

import numpy as np
import pytest

import tack
import tack.dataset as td
from tack.dataset import shapes


@tack.func
def _normal_linear(x):
    # Each component depends on its own coordinate alone: in lowest-order HDiv.
    return tack.vec3(1.0 + 2.0 * x[0], 3.0 - x[1], 0.5 + 4.0 * x[2])


@tack.func
def _constant(x):
    return tack.vec3(1.0, -2.0, 0.5)


@tack.kernel
def _face_fluxes(faces, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        out[e] = _normal_linear(centers[e]).dot(normals[e]) * areas[e]


@tack.kernel
def _edge_integrals(edges, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        out[edges.entity_id(i)] = _constant(0.5 * (x0 + x1)).dot(x1 - x0)


@tack.kernel
def _hdiv_at(cells, u, px, py, pz, values, positions):
    for c in cells:
        pc = tack.vec3(px, py, pz)
        values[cells.entity_id(c)] = td.hdiv_value(cells, u, c, pc)
        positions[cells.entity_id(c)] = cells.position(c, pc)


@tack.kernel
def _hcurl_at(cells, u, px, py, pz, values, positions):
    for c in cells:
        pc = tack.vec3(px, py, pz)
        values[cells.entity_id(c)] = td.hcurl_value(cells, u, c, pc)
        positions[cells.entity_id(c)] = cells.position(c, pc)


def _grid():
    return td.uniform_grid((5, 4, 3), origin=(0.3, -0.2, 1.0), spacing=(0.5, 0.3, 0.7))


POINTS = [(0.5, 0.5, 0.5), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.2, 0.7, 0.4)]


@pytest.mark.parametrize("pc", POINTS)
def test_hdiv_reproduces_its_fields(backend, pc):
    data = _grid()
    space = td.HDiv(data)
    assert space.size == data.topology.faces().num_faces
    geometry = td.face_geometry(data)
    fluxes = tack.empty((space.size,), data.dtype)
    td.for_each(_face_fluxes, data, "faces", td.face_centers(data).values,
                geometry.normal.values, geometry.area.values, fluxes)
    u = td.Field(space, fluxes)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hdiv_at, data, "cells", u, *pc, values, positions)
    x = positions.numpy()
    expected = np.stack([1 + 2 * x[:, 0], 3 - x[:, 1], 0.5 + 4 * x[:, 2]], axis=1)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("pc", POINTS)
def test_hcurl_reproduces_constants(backend, pc):
    data = _grid()
    space = td.HCurl(data)
    assert space.size == data.topology.edges().num_edges
    integrals = tack.empty((space.size,), data.dtype)
    td.for_each(_edge_integrals, data, "edges", integrals)
    u = td.Field(space, integrals)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hcurl_at, data, "cells", u, *pc, values, positions)
    expected = np.tile([1.0, -2.0, 0.5], (data.num_cells, 1))
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-5, atol=1e-5)


@tack.func
def _rotation(x):
    return tack.vec3(-x[1], x[0], 0.0)


@tack.func
def _bilinear(x):
    return tack.vec3(x[1] * x[2], 0.0, 0.0)


@tack.kernel
def _edge_integrals_of(edges, which, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        m = 0.5 * (x0 + x1)
        f = _rotation(m) if which == 0 else _bilinear(m)
        out[edges.entity_id(i)] = f.dot(x1 - x0)


@tack.kernel
def _hcurl_curl_at(cells, u, px, py, pz, values, positions):
    for c in cells:
        pc = tack.vec3(px, py, pz)
        values[cells.entity_id(c)] = td.hcurl_curl(cells, u, c, pc)
        positions[cells.entity_id(c)] = cells.position(c, pc)


@pytest.mark.parametrize("pc", POINTS)
@pytest.mark.parametrize("which", [0, 1])
def test_hcurl_values_and_curls(backend, pc, which):
    """(-y, x, 0), curl (0, 0, 2), and (y z, 0, 0), curl (0, y, -z): both in the space."""
    data = _grid()
    space = td.HCurl(data)
    integrals = tack.empty((space.size,), data.dtype)
    td.for_each(_edge_integrals_of, data, "edges", which, integrals)
    u = td.Field(space, integrals)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    curls = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hcurl_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hcurl_curl_at, data, "cells", u, *pc, curls, positions)
    x = positions.numpy()
    zero = np.zeros(len(x))
    if which == 0:
        value = np.stack([-x[:, 1], x[:, 0], zero], axis=1)
        curl = np.tile([0.0, 0.0, 2.0], (len(x), 1))
    else:
        value = np.stack([x[:, 1] * x[:, 2], zero, zero], axis=1)
        curl = np.stack([zero, x[:, 1], -x[:, 2]], axis=1)
    np.testing.assert_allclose(values.numpy(), value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(curls.numpy(), curl, rtol=1e-5, atol=1e-5)


def test_h1_node_queries():
    data = td.uniform_grid((3, 3, 3))
    hexa = next(g.shape for g in data.topology.groups() if g.count)
    assert td.H1(data).quadratic_count(hexa) == hexa.NUM_POINTS
    assert td.H1(data, order=2).quadratic_count(hexa) == hexa.NUM_QUADRATIC
    assert td.H1(data, order=2, basis="serendipity").quadratic_count(hexa) == hexa.NUM_POINTS + hexa.NUM_EDGES
    with pytest.raises(ValueError):
        td.H1(data, order=3).quadratic_count(hexa)


@tack.func
def _rt(x):
    return tack.vec3(1.0 + 2.0 * x[0], 3.0 + 2.0 * x[1], 0.5 + 2.0 * x[2])


@tack.func
def _nedelec(x):
    return tack.vec3(1.0, -2.0, 0.5) + tack.vec3(0.3, -0.2, 0.7).cross(x)


@tack.kernel
def _fluxes_of(faces, which, centers, normals, areas, out):
    for f in faces:
        e = faces.entity_id(f)
        v = _rt(centers[e]) if which == 0 else _normal_linear(centers[e])
        out[e] = v.dot(normals[e]) * areas[e]


@tack.kernel
def _nedelec_integrals(edges, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        out[edges.entity_id(i)] = _nedelec(0.5 * (x0 + x1)).dot(x1 - x0)


@tack.kernel
def _hdiv_div_at(cells, u, px, py, pz, values):
    for c in cells:
        values[cells.entity_id(c)] = td.hdiv_div(cells, u, c, tack.vec3(px, py, pz))


def _tetrahedra():
    pytest.importorskip("vtkmodules.vtkFiltersGeneral")
    from vtkmodules.vtkCommonDataModel import vtkImageData
    from vtkmodules.vtkFiltersGeneral import vtkDataSetTriangleFilter

    from tack.interop.vtk import vtk_to_dataset

    image = vtkImageData(dimensions=(5, 4, 3), origin=(0.3, -0.2, 1.0), spacing=(0.5, 0.3, 0.7))
    return vtk_to_dataset(vtkDataSetTriangleFilter()(image))


TET_POINTS = [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.6, 0.2, 0.1)]


@pytest.mark.parametrize("pc", TET_POINTS)
def test_hdiv_on_tetrahedra(backend, pc):
    data = _tetrahedra()
    space = td.HDiv(data)
    geometry = td.face_geometry(data)
    fluxes = tack.empty((space.size,), data.dtype)
    td.for_each(_fluxes_of, data, "faces", 0, td.face_centers(data).values, geometry.normal.values,
                geometry.area.values, fluxes)
    u = td.Field(space, fluxes)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    divergence = tack.empty((data.num_cells,), data.dtype)
    td.for_each(_hdiv_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hdiv_div_at, data, "cells", u, *pc, divergence)
    x = positions.numpy()
    np.testing.assert_allclose(values.numpy(), np.stack([1 + 2 * x[:, 0], 3 + 2 * x[:, 1], 0.5 + 2 * x[:, 2]], 1),
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(divergence.numpy(), 6.0, rtol=1e-5)


@pytest.mark.parametrize("pc", TET_POINTS)
def test_hcurl_on_tetrahedra(backend, pc):
    data = _tetrahedra()
    space = td.HCurl(data)
    integrals = tack.empty((space.size,), data.dtype)
    td.for_each(_nedelec_integrals, data, "edges", integrals)
    u = td.Field(space, integrals)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    curls = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hcurl_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hcurl_curl_at, data, "cells", u, *pc, curls, positions)
    x = positions.numpy()
    expected = np.array([1.0, -2.0, 0.5]) + np.cross(np.array([0.3, -0.2, 0.7]), x)
    np.testing.assert_allclose(values.numpy(), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(curls.numpy(), np.tile([0.6, -0.4, 1.4], (len(x), 1)), rtol=1e-5, atol=1e-5)


def test_hdiv_divergence_on_hexahedra(backend):
    data = _grid()
    geometry = td.face_geometry(data)
    fluxes = tack.empty((td.HDiv(data).size,), data.dtype)
    td.for_each(_fluxes_of, data, "faces", 1, td.face_centers(data).values, geometry.normal.values,
                geometry.area.values, fluxes)
    divergence = tack.empty((data.num_cells,), data.dtype)
    td.for_each(_hdiv_div_at, data, "cells", td.Field(td.HDiv(data), fluxes), 0.3, 0.6, 0.2, divergence)
    np.testing.assert_allclose(divergence.numpy(), 5.0, rtol=1e-5)


@tack.func
def _rt_order(x, p):
    """In RT of order p: component a of degree p along axis a and p - 1 across."""
    if p == 2:
        return tack.vec3(x[0] * x[0] * x[1] + x[2], x[1] * x[1] * x[2] + x[0], x[2] * x[2] * x[0] + x[1])
    return tack.vec3(x[0] * x[0] * x[0] * x[1] * x[1] + x[2] * x[2],
                     x[1] * x[1] * x[1] * x[2] * x[2] + x[0] * x[0],
                     x[2] * x[2] * x[2] * x[0] * x[0] + x[1] * x[1])


def _rt_order_numpy(x, p):
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    if p == 2:
        value = np.stack([X * X * Y + Z, Y * Y * Z + X, Z * Z * X + Y], 1)
        div = 2 * X * Y + 2 * Y * Z + 2 * Z * X
    else:
        value = np.stack([X ** 3 * Y ** 2 + Z ** 2, Y ** 3 * Z ** 2 + X ** 2, Z ** 3 * X ** 2 + Y ** 2], 1)
        div = 3 * X * X * Y * Y + 3 * Y * Y * Z * Z + 3 * Z * Z * X * X
    return value, div


@tack.kernel
def _face_densities(faces, normals, gauss, p, out):
    for f in faces:
        e = faces.entity_id(f)
        x0 = faces.corner_position(f, 0)
        x1 = faces.corner_position(f, 1)
        x2 = faces.corner_position(f, 2)
        x3 = faces.corner_position(f, 3)
        for m in range(p):
            for q in range(p):
                u = gauss[m]
                w = gauss[q]
                x = x0 * ((1.0 - u) * (1.0 - w)) + x1 * (u * (1.0 - w)) + x2 * (u * w) + x3 * ((1.0 - u) * w)
                du = (x1 - x0) * (1.0 - w) + (x2 - x3) * w
                dw = (x3 - x0) * (1.0 - u) + (x2 - x1) * u
                n = du.cross(dw)
                n = n if n.dot(normals[e]) > 0.0 else -n
                out[e * p * p + m + p * q] = _rt_order(x, p).dot(n)


@tack.kernel
def _interior_values(cells, gauss, lobatto, p, base, out):
    for c in cells:
        for a in range(3):
            b = (a + 1) % 3
            for i in range(1, p):
                for j in range(p):
                    for k in range(p):
                        t0 = lobatto[i] if a == 0 else (gauss[j] if b == 0 else gauss[k])
                        t1 = lobatto[i] if a == 1 else (gauss[j] if b == 1 else gauss[k])
                        t2 = lobatto[i] if a == 2 else (gauss[j] if b == 2 else gauss[k])
                        pc = tack.vec3(t0, t1, t2)
                        J = cells.geometry_jacobian(c, pc)
                        reference = J.determinant() * (J.inverse() @ _rt_order(cells.position(c, pc), p))
                        index = (base + cells.entity_id(c) * 3 * (p - 1) * p * p + a * (p - 1) * p * p
                                 + (i - 1) * p * p + j * p + k)
                        out[index] = reference[a]


def _hdiv_of_order(data, p):
    from tack.dataset.vector_spaces import gauss_nodes, lobatto_nodes
    space = td.HDiv(data, order=p)
    values = tack.zeros((space.size,), data.dtype)
    gauss = tack.array(gauss_nodes(p).astype(np.float32))
    lobatto = tack.array(lobatto_nodes(p + 1).astype(np.float32))
    td.for_each(_face_densities, data, "faces", td.face_geometry(data).normal.values, gauss, p, values)
    td.for_each(_interior_values, data, "cells", gauss, lobatto, p, data.topology.faces().num_faces * p * p,
                values)
    return td.Field(space, values)


@pytest.mark.parametrize("pc", POINTS)
@pytest.mark.parametrize("p", [2, 3])
def test_hdiv_of_higher_order_reproduces_its_fields(backend, pc, p):
    data = _grid()
    u = _hdiv_of_order(data, p)
    assert u.space.size == data.topology.faces().num_faces * p * p + data.num_cells * 3 * (p - 1) * p * p
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    divergence = tack.empty((data.num_cells,), data.dtype)
    td.for_each(_hdiv_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hdiv_div_at, data, "cells", u, *pc, divergence)
    value, div = _rt_order_numpy(positions.numpy().astype(np.float64), p)
    scale = max(1.0, float(np.abs(value).max()))
    np.testing.assert_allclose(values.numpy(), value, rtol=2e-5, atol=2e-5 * scale)
    np.testing.assert_allclose(divergence.numpy(), div, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(div).max())))


@tack.func
def _ned_order(x, p):
    """In Nedelec (first kind) of order p: component a of degree p - 1 along axis a, p across."""
    if p == 2:
        return tack.vec3(x[0] * x[1] * x[1] + x[2] * x[2], x[1] * x[2] * x[2] + x[0] * x[0],
                         x[2] * x[0] * x[0] + x[1] * x[1])
    return tack.vec3(x[0] * x[0] * x[1] * x[1] * x[1] + x[2] * x[2] * x[2],
                     x[1] * x[1] * x[2] * x[2] * x[2] + x[0] * x[0] * x[0],
                     x[2] * x[2] * x[0] * x[0] * x[0] + x[1] * x[1] * x[1])


def _ned_order_numpy(x, p):
    def f(y):
        X, Y, Z = y[..., 0], y[..., 1], y[..., 2]
        if p == 2:
            return np.stack([X * Y * Y + Z * Z, Y * Z * Z + X * X, Z * X * X + Y * Y], -1)
        return np.stack([X * X * Y ** 3 + Z ** 3, Y * Y * Z ** 3 + X ** 3, Z * Z * X ** 3 + Y ** 3], -1)

    h = 1e-5
    J = np.stack([(f(x + h * e) - f(x - h * e)) / (2 * h) for e in np.eye(3)], -1)
    curl = np.stack([J[:, 2, 1] - J[:, 1, 2], J[:, 0, 2] - J[:, 2, 0], J[:, 1, 0] - J[:, 0, 1]], 1)
    return f(x), curl


@tack.kernel
def _ned_edges(edges, gauss, p, out):
    for i in edges:
        x0 = edges.corner_position(i, 0)
        x1 = edges.corner_position(i, 1)
        for m in range(p):
            x = x0 + (x1 - x0) * gauss[m]
            out[edges.entity_id(i) * p + m] = _ned_order(x, p).dot(x1 - x0)


@tack.kernel
def _ned_faces(faces, gauss, lobatto, p, base, out):
    for f in faces:
        e = faces.entity_id(f)
        x0 = faces.corner_position(f, 0)
        x1 = faces.corner_position(f, 1)
        x2 = faces.corner_position(f, 2)
        x3 = faces.corner_position(f, 3)
        for m in range(p):
            for q in range(1, p):
                for which in range(2):
                    u = gauss[m] if which == 0 else lobatto[q]
                    w = lobatto[q] if which == 0 else gauss[m]
                    x = x0 * ((1.0 - u) * (1.0 - w)) + x1 * (u * (1.0 - w)) + x2 * (u * w) + x3 * ((1.0 - u) * w)
                    du = (x1 - x0) * (1.0 - w) + (x2 - x3) * w
                    dw = (x3 - x0) * (1.0 - u) + (x2 - x1) * u
                    t = du if which == 0 else dw
                    out[base + e * 2 * p * (p - 1) + which * p * (p - 1) + m + p * (q - 1)] = _ned_order(x, p).dot(t)


@tack.kernel
def _ned_interior(cells, gauss, lobatto, p, base, out):
    for c in cells:
        for a in range(3):
            b = (a + 1) % 3
            for i in range(p):
                for j in range(1, p):
                    for k in range(1, p):
                        t0 = gauss[i] if a == 0 else (lobatto[j] if b == 0 else lobatto[k])
                        t1 = gauss[i] if a == 1 else (lobatto[j] if b == 1 else lobatto[k])
                        t2 = gauss[i] if a == 2 else (lobatto[j] if b == 2 else lobatto[k])
                        pc = tack.vec3(t0, t1, t2)
                        J = cells.geometry_jacobian(c, pc)
                        reference = J.transpose() @ _ned_order(cells.position(c, pc), p)
                        index = (base + cells.entity_id(c) * 3 * p * (p - 1) * (p - 1) + a * p * (p - 1) * (p - 1)
                                 + i * (p - 1) * (p - 1) + (j - 1) * (p - 1) + (k - 1))
                        out[index] = reference[a]


def _hcurl_of_order(data, p):
    from tack.dataset.vector_spaces import gauss_nodes, lobatto_nodes
    space = td.HCurl(data, order=p)
    values = tack.zeros((space.size,), data.dtype)
    gauss = tack.array(gauss_nodes(p).astype(np.float32))
    lobatto = tack.array(lobatto_nodes(p + 1).astype(np.float32))
    edges = data.topology.edges().num_edges
    faces = data.topology.faces().num_faces
    td.for_each(_ned_edges, data, "edges", gauss, p, values)
    td.for_each(_ned_faces, data, "faces", gauss, lobatto, p, edges * p, values)
    td.for_each(_ned_interior, data, "cells", gauss, lobatto, p, edges * p + faces * 2 * p * (p - 1), values)
    return td.Field(space, values)


@pytest.mark.parametrize("pc", POINTS)
@pytest.mark.parametrize("p", [2, 3])
def test_hcurl_of_higher_order_reproduces_its_fields(backend, pc, p):
    data = _grid()
    u = _hcurl_of_order(data, p)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    curls = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    td.for_each(_hcurl_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hcurl_curl_at, data, "cells", u, *pc, curls, positions)
    value, curl = _ned_order_numpy(positions.numpy().astype(np.float64), p)
    np.testing.assert_allclose(values.numpy(), value, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(value).max())))
    np.testing.assert_allclose(curls.numpy(), curl, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(curl).max())))


# Continuity across faces, on meshes whose cells number their corners every way.

_HEX_BITS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
_VOXEL_BITS = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)])
_BLOCK = (4, 3, 3)
_SPACING = np.array([0.5, 0.3, 0.7])
_ORIGIN = np.array([0.3, -0.2, 1.0])


def _symmetries(reflections):
    """The cube's 24 rotations, or with ``reflections`` all 48 of its symmetries."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), int)
            for r, (p, s) in enumerate(zip(perm, signs)):
                R[r, p] = s
            if reflections or round(np.linalg.det(R)) == 1:
                out.append(R)
    return out


def _unstructured(points, types, connectivity, per_cell):
    offsets = np.arange(len(types) + 1, dtype=np.int64) * per_cell
    topology = td.UnstructuredTopology(tack.array(np.asarray(types, np.uint8)), tack.array(offsets),
                                       tack.array(np.asarray(connectivity, np.int64)), num_points=len(points))
    positions = tack.empty((len(points),), tack.vec3(tack.f32))
    tack.copyto(positions, points.astype(np.float32))
    return td.DataSet(topology, positions)


def _block(numbering="natural", voxels=None, jitter=0.0, seed=3):
    """A 4 x 3 x 3 block of hexahedra numbered ``natural``ly, from random ``rotated`` cubes,
    or from any of the cube's symmetries (``reflected``: some cells inside out). ``voxels``
    (i, j, k) picks the cells that are voxels instead; ``jitter`` moves the interior
    points, so the cells are not affine."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = _BLOCK
    ids = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nz + 1, ny + 1, nx + 1)
    k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    points = _ORIGIN + _SPACING * np.stack([i, j, k], -1).reshape(-1, 3)
    if jitter:
        inside = ((i > 0) & (i < nx) & (j > 0) & (j < ny) & (k > 0) & (k < nz)).reshape(-1)
        points[inside] += jitter * _SPACING * rng.uniform(-1.0, 1.0, (int(inside.sum()), 3))
    symmetries = _symmetries(numbering == "reflected")
    types, connectivity = [], []
    for c in range(nz):
        for b in range(ny):
            for a in range(nx):
                if voxels is not None and voxels(a, b, c):
                    types.append(int(shapes.VOXEL))
                    connectivity += [ids[c + o[2], b + o[1], a + o[0]] for o in _VOXEL_BITS]
                    continue
                R = np.eye(3, dtype=int) if numbering == "natural" else symmetries[rng.integers(len(symmetries))]
                types.append(int(shapes.HEXAHEDRON))
                for bit in _HEX_BITS:
                    o = np.rint(R @ (bit - 0.5) + 0.5).astype(int)
                    connectivity.append(ids[c + o[2], b + o[1], a + o[0]])
    return _unstructured(points, types, connectivity, 8)


def _kuhn_tetrahedra(odd, seed=5):
    """The block's cubes as six tetrahedra each (Kuhn's), each numbering its corners from a
    random even permutation, or with ``odd`` any permutation, which turns some inside out."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = _BLOCK
    ids = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nz + 1, ny + 1, nx + 1)
    k, j, i = np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij")
    points = _ORIGIN + _SPACING * np.stack([i, j, k], -1).reshape(-1, 3)
    perms = list(itertools.permutations(range(4)))
    even = [p for p in perms if round(np.linalg.det(np.eye(4)[list(p)])) == 1]
    connectivity = []
    for c in range(nz):
        for b in range(ny):
            for a in range(nx):
                for axes in itertools.permutations(range(3)):
                    o = np.zeros(3, int)
                    corners = [ids[c, b, a]]
                    for axis in axes:
                        o[axis] = 1
                        corners.append(ids[c + o[2], b + o[1], a + o[0]])
                    x = points[corners]
                    if np.linalg.det(x[1:] - x[0]) < 0:
                        corners[2], corners[3] = corners[3], corners[2]
                    p = (perms if odd else even)[rng.integers(len(perms) if odd else len(even))]
                    connectivity += [corners[q] for q in p]
    return _unstructured(points, [int(shapes.TETRA)] * (len(connectivity) // 4), connectivity, 4)


def _rotated_hexahedra(seed=3):
    """A 4 x 3 x 3 block of hexahedra whose corners each cell numbers from a random one of
    the cube's 24 rotations, so shared faces and edges have different local frames."""
    return _block("rotated", seed=seed)


REPRODUCING = {"rotated hexahedra": _rotated_hexahedra,
               "voxels": lambda: _block(voxels=lambda a, b, c: True),
               "voxels and rotated hexahedra": lambda: _block("rotated", voxels=lambda a, b, c: (a + b) % 2 == 0)}


@pytest.mark.parametrize("mesh", list(REPRODUCING))
@pytest.mark.parametrize("p", [1, 2, 3])
def test_vector_spaces_on_rotated_hexahedra(backend, p, mesh):
    data = REPRODUCING[mesh]()
    pc = (0.3, 0.6, 0.2)
    values = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    other = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    positions = tack.empty((data.num_cells,), tack.vec3(data.dtype))
    divergence = tack.empty((data.num_cells,), data.dtype)
    if p == 1:
        geometry = td.face_geometry(data)
        fluxes = tack.empty((td.HDiv(data).size,), data.dtype)
        td.for_each(_fluxes_of, data, "faces", 1, td.face_centers(data).values, geometry.normal.values,
                    geometry.area.values, fluxes)
        u = td.Field(td.HDiv(data), fluxes)
        integrals = tack.empty((td.HCurl(data).size,), data.dtype)
        td.for_each(_edge_integrals_of, data, "edges", 0, integrals)
        w = td.Field(td.HCurl(data), integrals)
    else:
        u = _hdiv_of_order(data, p)
        w = _hcurl_of_order(data, p)
    td.for_each(_hdiv_at, data, "cells", u, *pc, values, positions)
    td.for_each(_hdiv_div_at, data, "cells", u, *pc, divergence)
    x = positions.numpy().astype(np.float64)
    if p == 1:
        value, div = np.stack([1 + 2 * x[:, 0], 3 - x[:, 1], 0.5 + 4 * x[:, 2]], 1), np.full(len(x), 5.0)
    else:
        value, div = _rt_order_numpy(x, p)
    np.testing.assert_allclose(values.numpy(), value, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(value).max())))
    np.testing.assert_allclose(divergence.numpy(), div, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(div).max())))
    td.for_each(_hcurl_at, data, "cells", w, *pc, values, positions)
    td.for_each(_hcurl_curl_at, data, "cells", w, *pc, other, positions)
    if p == 1:
        value = np.stack([-x[:, 1], x[:, 0], 0 * x[:, 0]], 1)
        curl = np.tile([0.0, 0.0, 2.0], (len(x), 1))
    else:
        value, curl = _ned_order_numpy(x, p)
    np.testing.assert_allclose(values.numpy(), value, rtol=2e-5, atol=2e-5 * max(1.0, float(np.abs(value).max())))
    np.testing.assert_allclose(other.numpy(), curl, rtol=1e-4, atol=1e-4 * max(1.0, float(np.abs(curl).max())))


def test_vector_spaces_refuse_other_shapes(backend):
    data = td.uniform_grid((4, 3, 1))           # quads
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        values = tack.zeros((td.HDiv(data).size,), data.dtype)
        u = td.Field(td.HDiv(data), values)
        out = tack.empty((data.num_cells,), tack.vec3(data.dtype))
        td.for_each(_hdiv_at, data, "cells", u, 0.5, 0.5, 0.5, out, out)


CONTINUITY = {
    "hexahedra": (lambda: _block(), (1, 2, 3)),
    "rotated hexahedra": (lambda: _block("rotated"), (1, 2, 3)),
    "reflected hexahedra": (lambda: _block("reflected"), (1, 2, 3)),
    "reflected hexahedra, not affine": (lambda: _block("reflected", jitter=0.2), (1, 2, 3)),
    "voxels": (lambda: _block(voxels=lambda a, b, c: True), (1, 2, 3)),
    "voxels and rotated hexahedra": (lambda: _block("rotated", voxels=lambda a, b, c: (a + b + c) % 2 == 0),
                                     (1, 2, 3)),
    "tetrahedra": (lambda: _kuhn_tetrahedra(odd=False), (1,)),
    "inside-out tetrahedra": (lambda: _kuhn_tetrahedra(odd=True), (1,)),
}

_FACE_POINTS = [(1 / 3, 1 / 3), (0.2, 0.3), (0.6, 0.2), (0.1, 0.7), (0.25, 0.15)]
"""Points in a face's own frame, inside triangles and quadrilaterals alike."""


@tack.func
def _face_frame(cells, c, lf):
    """The parametric points of local face ``lf``'s own corners 0, 1 and its last (3 on a
    quadrilateral, 2 on a triangle), the frame both cells of the face share."""
    m = cells.face_num_points(lf)
    p0 = tack.vec3(0.0, 0.0, 0.0)
    p1 = tack.vec3(0.0, 0.0, 0.0)
    p2 = tack.vec3(0.0, 0.0, 0.0)
    for i in range(m):
        where = cells.face_position(c, lf, i)
        point = cells.parametric_point(cells.face_corner(lf, i))
        if where == 0:
            p0 = point
        if where == 1:
            p1 = point
        if where == m - 1:
            p2 = point
    return p0, p1 - p0, p2 - p0


@tack.kernel
def _normal_traces(cells, u, s, t, count, out):
    for c in cells:
        for lf in range(cells.NUM_FACES):
            p0, du, dw = _face_frame(cells, c, lf)
            row = 2 * cells.face_id(c, lf) + cells.face_side(c, lf)
            for k in range(count):
                pc = p0 + s[k] * du + t[k] * dw
                J = cells.geometry_jacobian(c, pc)
                n = (J @ du).cross(J @ dw)
                out[2 * (row * count + k)] = td.hdiv_value(cells, u, c, pc).dot(n) / n.norm()


@tack.kernel
def _tangential_traces(cells, u, s, t, count, out):
    for c in cells:
        for lf in range(cells.NUM_FACES):
            p0, du, dw = _face_frame(cells, c, lf)
            row = 2 * cells.face_id(c, lf) + cells.face_side(c, lf)
            for k in range(count):
                pc = p0 + s[k] * du + t[k] * dw
                J = cells.geometry_jacobian(c, pc)
                v = td.hcurl_value(cells, u, c, pc)
                out[2 * (row * count + k)] = v.dot(J @ du) / (J @ du).norm()
                out[2 * (row * count + k) + 1] = v.dot(J @ dw) / (J @ dw).norm()


def _jumps(data, space, kernel, seed):
    """The largest difference between the two sides' traces of a field of random values
    on every interior face, and the largest trace."""
    values = np.random.default_rng(seed).uniform(-1.0, 1.0, space.size).astype(np.float32)
    u = td.Field(space, tack.array(values))
    count = len(_FACE_POINTS)
    s = tack.array(np.array([p[0] for p in _FACE_POINTS], np.float32))
    t = tack.array(np.array([p[1] for p in _FACE_POINTS], np.float32))
    out = tack.array(np.full(2 * 2 * data.topology.faces().num_faces * count * 2, np.nan, np.float32))
    td.for_each(kernel, data, "cells", u, s, t, count, out)
    out = out.numpy().reshape(-1, 2, count, 2)
    inside = ~np.isnan(out[:, 1, 0, 0])
    if kernel is _normal_traces:
        out = out[..., :1]
    both = out[inside]
    assert len(both) > 0
    return float(np.abs(both[:, 0] - both[:, 1]).max()), float(np.abs(both).max())


@pytest.mark.parametrize(("make", "p"), [(m, p) for m, (_, orders) in CONTINUITY.items() for p in orders],
                         ids=[f"{m}-p{p}" for m, (_, orders) in CONTINUITY.items() for p in orders])
def test_normal_and_tangential_components_are_continuous(backend, make, p):
    data = CONTINUITY[make][0]()
    for space, kernel in ((td.HDiv(data, order=p), _normal_traces), (td.HCurl(data, order=p), _tangential_traces)):
        jump, largest = _jumps(data, space, kernel, seed=p)
        assert jump <= 2e-5 * largest, (type(space).__name__, jump, largest)

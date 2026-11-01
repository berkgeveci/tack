"""HDiv and HCurl reproduce the fields their spaces hold, through their Piola maps."""

import importlib

import numpy as np
import pytest

import tack
import tack.dataset as td


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
    two = td.H1(data, order=2)
    assert not two.reads_any_order(hexa)
    assert two.quadratic_count(hexa) == hexa.NUM_QUADRATIC
    assert td.H1(data, order=3).reads_any_order(hexa)


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

    image = vtkImageData()
    image.SetDimensions(5, 4, 3)
    image.SetOrigin(0.3, -0.2, 1.0)
    image.SetSpacing(0.5, 0.3, 0.7)
    tets = vtkDataSetTriangleFilter()
    tets.SetInputData(image)
    tets.Update()
    return vtk_to_dataset(tets.GetOutput())


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


def _rotated_hexahedra(seed=3):
    """A 4 x 3 x 3 block of hexahedra whose corners each cell numbers from a random one of
    the cube's 24 rotations, so shared faces and edges have different local frames."""
    pytest.importorskip("vtkmodules.vtkCommonDataModel")
    import itertools

    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import VTK_HEXAHEDRON, vtkUnstructuredGrid

    from tack.interop.vtk import vtk_to_dataset

    rng = np.random.default_rng(seed)
    n = (4, 3, 3)
    spacing = np.array([0.5, 0.3, 0.7])
    origin = np.array([0.3, -0.2, 1.0])
    ids = np.arange((n[0] + 1) * (n[1] + 1) * (n[2] + 1)).reshape(n[2] + 1, n[1] + 1, n[0] + 1)
    grid = np.stack(np.meshgrid(np.arange(n[0] + 1), np.arange(n[1] + 1), np.arange(n[2] + 1), indexing="xy"), -1)
    positions = origin + spacing * np.stack([grid[..., 0], grid[..., 1], grid[..., 2]], -1).transpose(2, 0, 1, 3).reshape(-1, 3)
    bits = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            R = np.zeros((3, 3), int)
            for r, (p, s) in enumerate(zip(perm, signs)):
                R[r, p] = s
            if round(np.linalg.det(R)) == 1:
                rotations.append(R)
    cells = []
    for k in range(n[2]):
        for j in range(n[1]):
            for i in range(n[0]):
                R = rotations[rng.integers(len(rotations))]
                corners = []
                for b in bits:
                    o = np.rint(R @ (b - 0.5) + 0.5).astype(int)
                    corners.append(ids[k + o[2], j + o[1], i + o[0]])
                cells.append(corners)
    ug = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetData(numpy_to_vtk(positions.astype(np.float64), deep=True))
    ug.SetPoints(points)
    for corners in cells:
        ug.InsertNextCell(VTK_HEXAHEDRON, 8, [int(c) for c in corners])
    return vtk_to_dataset(ug)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_vector_spaces_on_rotated_hexahedra(backend, p):
    data = _rotated_hexahedra()
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


def test_world_gradient_is_public():
    module = importlib.import_module("tack.dataset.algorithms.gradients")
    assert td.world_gradient is module._world_gradient


def test_vector_spaces_refuse_other_shapes(backend):
    data = td.uniform_grid((4, 3, 1))           # quads
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        values = tack.zeros((td.HDiv(data).size,), data.dtype)
        u = td.Field(td.HDiv(data), values)
        out = tack.empty((data.num_cells,), tack.vec3(data.dtype))
        td.for_each(_hdiv_at, data, "cells", u, 0.5, 0.5, 0.5, out, out)


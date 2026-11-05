# SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""A field given its gradient: world_gradient, world_jet and gradients read it instead of
differentiating the basis."""

import numpy as np
import pytest

import tack
import tack.dataset as td


@tack.func
def _u(x):
    return x[0] * x[0] + 2.0 * x[1] - x[2]


@tack.func
def _v(x):
    return tack.vec3(x[0] * x[1], x[2], -x[0])


@tack.kernel
def _gradients_at(cells, u, out):
    for c in cells:
        out[cells.entity_id(c)] = td.world_gradient(cells, u, c, cells.parametric_center())


@tack.kernel
def _jets_at(cells, u, values, grads, hess):
    for c in cells:
        v, g, h = td.world_jet(cells, u, c, cells.parametric_center())
        e = cells.entity_id(c)
        values[e] = v
        grads[e] = g
        hess[e] = h[0, 0]


def _data():
    return td.uniform_grid((5, 4, 3), origin=(0.1, 0.2, 0.3), spacing=(0.3, 0.25, 0.4))


def _vectors(rows):
    rows = np.asarray(rows, np.float32)
    out = tack.empty((len(rows),), tack.vec(rows.shape[1], tack.f32))
    tack.copyto(out, rows)
    return out


def _per_cell(data, rows):
    return td.Field(td.Constant(data), _vectors(rows))


def test_world_gradient_and_gradients_read_a_given_gradient(backend):
    data = _data()
    given = np.random.default_rng(1).uniform(-1, 1, (data.num_cells, 3)).astype(np.float32)
    u = td.interpolate(data, _u, td.H1(data))
    data.fields["u"] = td.Field(u.space, u.values, gradient=_per_cell(data, given))
    out = tack.zeros((data.num_cells,), tack.vec3(tack.f32))
    td.for_each(_gradients_at, data, "cells", data.fields["u"], out)
    np.testing.assert_allclose(out.numpy(), given)
    np.testing.assert_allclose(td.gradients(data, "u").values.numpy(), given)
    plain = td.gradients(data, td.Field(u.space, u.values), at="points").values.numpy()
    at_points = td.gradients(data, "u", at="points").values.numpy()
    assert not np.allclose(plain, at_points)


def test_world_jet_pairs_the_given_gradient_with_the_value(backend):
    data = _data()
    given = np.tile(np.float32([1.0, 2.0, 3.0]), (data.num_cells, 1))
    u = td.interpolate(data, _u, td.H1(data))
    field = td.Field(u.space, u.values, gradient=_per_cell(data, given))
    values = tack.zeros((data.num_cells,), tack.f32)
    grads = tack.zeros((data.num_cells,), tack.vec3(tack.f32))
    hess = tack.zeros((data.num_cells,), tack.f32)
    td.for_each(_jets_at, data, "cells", field, values, grads, hess)
    centers = td.cell_centers(data).values.numpy()
    np.testing.assert_allclose(grads.numpy(), given)
    np.testing.assert_allclose(values.numpy(), _u_numpy(centers), atol=0.05)
    assert np.isnan(hess.numpy()).all()


def _u_numpy(x):
    return x[:, 0] ** 2 + 2.0 * x[:, 1] - x[:, 2]


def test_a_vector_field_gives_its_rows(backend):
    data = _data()
    jacobian = np.random.default_rng(2).uniform(-1, 1, (data.num_cells, 9)).astype(np.float32)
    v = td.interpolate(data, _v, td.H1(data), dtype=tack.vec3(tack.f32))
    data.fields["v"] = td.Field(v.space, v.values,
                                gradient=td.Field(td.Constant(data), _vectors(jacobian)))
    np.testing.assert_allclose(td.gradients(data, "v").values.numpy(), jacobian)
    q = td.flow_quantities(data, td.gradients(data, "v"))
    J = jacobian.reshape(-1, 3, 3)
    np.testing.assert_allclose(q[0].values.numpy(), np.trace(J, axis1=1, axis2=2), rtol=1e-5, atol=1e-6)


def test_a_given_gradient_is_checked(backend):
    data = _data()
    u = td.interpolate(data, _u, td.H1(data))
    with pytest.raises(ValueError):
        td.Field(u.space, u.values, gradient=td.Field(td.Constant(data), tack.zeros((data.num_cells,), tack.f32)))
    with pytest.raises(ValueError):
        other = _data()
        td.Field(u.space, u.values, gradient=_per_cell(other, np.zeros((other.num_cells, 3))))
    with pytest.raises(TypeError):
        td.Field(u.space, u.values, gradient=np.zeros(3))


def test_point_data_keeps_its_given_gradient(backend):
    data = _data()
    u = td.interpolate(data, _u, td.H1(data))
    given = np.tile(np.float32([4.0, 5.0, 6.0]), (data.num_cells, 1))
    data.fields["u"] = td.Field(td.Values(data, "points"), u.values, gradient=_per_cell(data, given))
    np.testing.assert_allclose(td.gradients(data, "u").values.numpy(), given)
